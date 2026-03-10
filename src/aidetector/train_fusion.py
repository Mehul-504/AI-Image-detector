from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .detectors.signal_utils import infer_transport_compression_signals
from .eval_dataset import iter_labelled_images
from .learned_fusion import (
    apply_platt,
    clamp01,
    confidence_from_probability,
    compression_bucket,
    extract_fusion_features,
    standardize_vector,
    vectorize_features,
)
from .pipeline import AuthenticityPipeline
from .schemas import AnalysisRequest, MediaType, Verdict


CLASS_NAMES = ("real", "ai_generated", "ai_edited")
SUBTYPE_FEATURE_NAMES = (
    "frequency_risk",
    "forensic_risk",
    "clip_risk",
    "tamper_risk",
    "spatial_risk",
    "compression_score",
    "metadata_transport_hint",
    "metadata_camera_hint",
    "watermark_risk",
    "prnu_risk",
    "clip_forensic_gap",
    "prnu_frequency_gap",
)


@dataclass(frozen=True)
class TrainSample:
    class_name: str
    binary_label: int
    subtype_label: int | None
    media_uri: str
    feature_map: dict[str, float]
    compression_score: float
    confidence_hint: float


@dataclass(frozen=True)
class LinearModel:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float


def _binary_label(name: str) -> int:
    return 0 if name == "real" else 1


def _subtype_label(name: str) -> int | None:
    if name == "ai_generated":
        return 1
    if name == "ai_edited":
        return 0
    return None


def _safe_probability(probability: float) -> float:
    return max(1e-6, min(1.0 - 1e-6, float(probability)))


def _stratified_split(
    samples: list[TrainSample],
    val_ratio: float,
    seed: int,
) -> tuple[list[TrainSample], list[TrainSample]]:
    rng = random.Random(seed)
    by_label: dict[str, list[TrainSample]] = defaultdict(list)
    for sample in samples:
        by_label[sample.class_name].append(sample)

    train: list[TrainSample] = []
    val: list[TrainSample] = []
    for items in by_label.values():
        shuffled = list(items)
        rng.shuffle(shuffled)
        n = len(shuffled)
        if n <= 1:
            train.extend(shuffled)
            continue
        if n == 2:
            val_count = 1
        else:
            val_count = max(1, int(round(n * val_ratio)))
            val_count = min(val_count, n - 1)
        val.extend(shuffled[:val_count])
        train.extend(shuffled[val_count:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _fit_logistic_regression(
    x_train,
    y_train,
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
):
    np = __import__("numpy")
    n_samples, n_features = x_train.shape
    weights = np.zeros((n_features,), dtype=np.float64)
    bias = 0.0
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    w_pos = float(n_samples / max(1.0, 2.0 * pos))
    w_neg = float(n_samples / max(1.0, 2.0 * neg))
    sample_weights = np.where(y_train > 0.5, w_pos, w_neg).astype(np.float64)

    for step in range(epochs):
        logits = x_train @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -32.0, 32.0)))
        error = (probs - y_train) * sample_weights
        grad_w = (x_train.T @ error) / float(n_samples) + l2 * weights
        grad_b = float(np.mean(error))
        lr = learning_rate * (0.985 ** (step / max(25, epochs)))
        weights -= lr * grad_w
        bias -= lr * grad_b
    return weights, float(bias)


def _fit_platt_scaler(
    logits,
    labels,
    *,
    epochs: int = 1200,
    learning_rate: float = 0.05,
    l2: float = 0.0005,
) -> tuple[float, float]:
    np = __import__("numpy")
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.size < 8:
        return 1.0, 0.0
    if len(set(int(v) for v in y.tolist())) < 2:
        return 1.0, 0.0

    n = float(len(y))
    pos = float(np.sum(y == 1.0))
    neg = float(np.sum(y == 0.0))
    w_pos = n / max(1.0, 2.0 * pos)
    w_neg = n / max(1.0, 2.0 * neg)
    sample_weights = np.where(y > 0.5, w_pos, w_neg)

    a = 1.0
    b = 0.0
    for step in range(max(200, epochs)):
        z = a * x + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
        error = (p - y) * sample_weights
        grad_a = float(np.mean(error * x) + l2 * (a - 1.0))
        grad_b = float(np.mean(error) + l2 * b)
        lr = learning_rate * (0.985 ** (step / max(20, epochs)))
        a -= lr * grad_a
        b -= lr * grad_b

    return float(a), float(b)


def _ece_binary(probabilities: list[float], labels: list[int], num_bins: int = 10) -> float:
    if not probabilities:
        return 0.0
    bins: list[list[tuple[float, int]]] = [[] for _ in range(num_bins)]
    for probability, label in zip(probabilities, labels):
        index = min(num_bins - 1, int(float(probability) * num_bins))
        bins[index].append((float(probability), int(label)))

    total = float(len(probabilities))
    ece = 0.0
    for group in bins:
        if not group:
            continue
        conf = sum(item[0] for item in group) / len(group)
        acc = sum(item[1] for item in group) / len(group)
        ece += abs(conf - acc) * (len(group) / total)
    return float(ece)


def _matrix_verdict(probability: float, confidence: float, thresholds: dict[str, float]) -> Verdict:
    if confidence < thresholds["min_confidence"]:
        return Verdict.SUSPICIOUS
    if probability < thresholds["risk_low"]:
        return Verdict.LIKELY_AUTHENTIC
    if probability >= thresholds["risk_high"]:
        return Verdict.LIKELY_SYNTHETIC_OR_EDITED
    return Verdict.SUSPICIOUS


def _evaluate_thresholds(
    probs: list[float],
    confs: list[float],
    labels: list[int],
    thresholds: dict[str, float],
) -> tuple[float, dict[str, float]]:
    tp = tn = fp = fn = 0
    suspicious = 0
    for probability, confidence, label in zip(probs, confs, labels):
        verdict = _matrix_verdict(probability, confidence, thresholds)
        if verdict == Verdict.SUSPICIOUS:
            suspicious += 1
        pred_ai = verdict != Verdict.LIKELY_AUTHENTIC
        if label == 1 and pred_ai:
            tp += 1
        elif label == 1 and not pred_ai:
            fn += 1
        elif label == 0 and pred_ai:
            fp += 1
        else:
            tn += 1

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    suspicious_rate = suspicious / max(1, len(labels))
    metrics = {
        "recall_ai": recall,
        "specificity_real": specificity,
        "accuracy": accuracy,
        "suspicious_rate": suspicious_rate,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    score = (
        0.42 * recall
        + 0.33 * specificity
        + 0.25 * accuracy
        - 0.08 * max(0.0, suspicious_rate - 0.50)
    )
    return score, metrics


def _optimize_thresholds(
    probs: list[float],
    confs: list[float],
    labels: list[int],
) -> tuple[dict[str, float], dict[str, float]]:
    best_score = -1.0
    best_thresholds = {"risk_low": 0.40, "risk_high": 0.56, "min_confidence": 0.38}
    best_metrics: dict[str, float] = {}
    for risk_low in [0.28, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44]:
        for risk_high in [0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66]:
            if risk_high <= risk_low + 0.10:
                continue
            for min_conf in [0.22, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46]:
                thresholds = {
                    "risk_low": risk_low,
                    "risk_high": risk_high,
                    "min_confidence": min_conf,
                }
                score, metrics = _evaluate_thresholds(probs, confs, labels, thresholds)
                if score > best_score:
                    best_score = score
                    best_thresholds = thresholds
                    best_metrics = metrics
    return best_thresholds, best_metrics


def _linear_logits(samples: list[TrainSample], model: LinearModel):
    np = __import__("numpy")
    vectors = np.asarray(
        [vectorize_features(sample.feature_map, model.feature_names) for sample in samples],
        dtype=np.float64,
    )
    means = np.asarray(model.means, dtype=np.float64)
    stds = np.asarray(model.stds, dtype=np.float64)
    weights = np.asarray(model.weights, dtype=np.float64)
    normalized = (vectors - means) / stds
    logits = normalized @ weights + float(model.bias)
    return logits


def _fit_linear_model(
    samples: list[TrainSample],
    labels: list[int],
    feature_names: tuple[str, ...],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> LinearModel:
    np = __import__("numpy")
    x_train = np.asarray(
        [vectorize_features(sample.feature_map, feature_names) for sample in samples],
        dtype=np.float64,
    )
    y_train = np.asarray(labels, dtype=np.float64)
    means = x_train.mean(axis=0)
    stds = np.maximum(1e-6, x_train.std(axis=0))
    x_norm = (x_train - means) / stds
    weights, bias = _fit_logistic_regression(
        x_norm,
        y_train,
        epochs=max(200, epochs),
        learning_rate=max(0.01, learning_rate),
        l2=max(0.0, l2),
    )
    return LinearModel(
        feature_names=feature_names,
        means=tuple(float(v) for v in means.tolist()),
        stds=tuple(float(v) for v in stds.tolist()),
        weights=tuple(float(v) for v in weights.tolist()),
        bias=float(bias),
    )


def _heuristic_subtype_generated_probability(feature_map: dict[str, float]) -> float:
    generated_signal = (
        0.42 * feature_map.get("frequency_risk", 0.5)
        + 0.32 * feature_map.get("forensic_risk", 0.5)
        + 0.26 * feature_map.get("clip_risk", 0.5)
    )
    edited_signal = (
        0.38 * feature_map.get("tamper_risk", 0.5)
        + 0.32 * feature_map.get("compression_score", 0.0)
        + 0.30 * feature_map.get("metadata_transport_hint", 0.0)
    )
    return clamp01(1.0 / (1.0 + math.exp(-4.0 * (generated_signal - edited_signal))))


def _three_way_probabilities(prob_ai: float, prob_generated_if_ai: float) -> dict[str, float]:
    p_real = clamp01(1.0 - prob_ai)
    p_generated = clamp01(prob_ai * prob_generated_if_ai)
    p_edited = clamp01(prob_ai * (1.0 - prob_generated_if_ai))
    total = p_real + p_generated + p_edited
    if total <= 0:
        return {"real": 1.0, "ai_generated": 0.0, "ai_edited": 0.0}
    return {
        "real": p_real / total,
        "ai_generated": p_generated / total,
        "ai_edited": p_edited / total,
    }


def _evaluate_three_way(
    truth: list[str],
    predicted: list[str],
    probabilities: list[dict[str, float]],
) -> dict[str, object]:
    confusion: Counter[tuple[str, str]] = Counter()
    for t, p in zip(truth, predicted):
        confusion[(t, p)] += 1

    per_class: dict[str, dict[str, float]] = {}
    f1_sum = precision_sum = recall_sum = 0.0
    for class_name in CLASS_NAMES:
        tp = confusion[(class_name, class_name)]
        fp = sum(confusion[(other, class_name)] for other in CLASS_NAMES if other != class_name)
        fn = sum(confusion[(class_name, other)] for other in CLASS_NAMES if other != class_name)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        per_class[class_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(sum(confusion[(class_name, p)] for p in CLASS_NAMES)),
        }

    total = len(truth)
    correct = sum(confusion[(class_name, class_name)] for class_name in CLASS_NAMES)
    accuracy = correct / total if total else 0.0

    brier = 0.0
    for class_name, probs in zip(truth, probabilities):
        row = 0.0
        for key in CLASS_NAMES:
            target = 1.0 if key == class_name else 0.0
            row += (float(probs.get(key, 0.0)) - target) ** 2
        brier += row
    brier /= max(1, total)

    matrix = {
        truth_label: {
            pred_label: int(confusion[(truth_label, pred_label)])
            for pred_label in CLASS_NAMES
        }
        for truth_label in CLASS_NAMES
    }
    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(precision_sum / len(CLASS_NAMES), 4),
        "macro_recall": round(recall_sum / len(CLASS_NAMES), 4),
        "macro_f1": round(f1_sum / len(CLASS_NAMES), 4),
        "brier_3way": round(brier, 4),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def _prepare_samples(dataset_root: Path) -> tuple[list[TrainSample], Counter[str]]:
    pipeline = AuthenticityPipeline()
    samples: list[TrainSample] = []
    counts = Counter()

    for class_name, path in iter_labelled_images(dataset_root):
        response = pipeline.analyze(
            AnalysisRequest(
                media_type=MediaType.IMAGE,
                media_uri=str(path),
                provided_scores={},
            )
        )
        compression = infer_transport_compression_signals(str(path))
        compression_score = float(compression.get("score", 0.0))
        features = extract_fusion_features(response.category_scores, compression_score=compression_score)
        samples.append(
            TrainSample(
                class_name=class_name,
                binary_label=_binary_label(class_name),
                subtype_label=_subtype_label(class_name),
                media_uri=str(path),
                feature_map=features,
                compression_score=compression_score,
                confidence_hint=float(features.get("confidence_mean", 0.2)),
            )
        )
        counts[class_name] += 1

    return samples, counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Train learned fusion model from local labeled dataset.")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--out", default="models/fusion_model.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.18)
    parser.add_argument("--l2", type=float, default=0.002)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        raise SystemExit(f"dataset root not found: {dataset_root}")
    samples, counts = _prepare_samples(dataset_root)
    if len(samples) < 24:
        raise SystemExit("need at least 24 images for training")

    train_samples, val_samples = _stratified_split(samples, val_ratio=args.val_ratio, seed=args.seed)
    if not train_samples or not val_samples:
        raise SystemExit("training/validation split failed; add more data")

    # Binary AI-vs-real model.
    binary_feature_names = tuple(sorted(train_samples[0].feature_map.keys()))
    binary_model = _fit_linear_model(
        train_samples,
        [sample.binary_label for sample in train_samples],
        binary_feature_names,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )

    np = __import__("numpy")
    val_logits = _linear_logits(val_samples, binary_model)
    val_binary_labels = [sample.binary_label for sample in val_samples]
    calibrator_a, calibrator_b = _fit_platt_scaler(val_logits, val_binary_labels)

    val_probs: list[float] = []
    val_confs: list[float] = []
    val_labels: list[int] = []
    val_buckets: list[str] = []
    for sample, logit in zip(val_samples, val_logits.tolist()):
        probability = apply_platt(float(logit), calibrator_a, calibrator_b)
        confidence = confidence_from_probability(
            probability_01=probability,
            confidence_hint_01=sample.confidence_hint,
            compression_score=sample.compression_score,
        )
        val_probs.append(probability)
        val_confs.append(confidence)
        val_labels.append(sample.binary_label)
        val_buckets.append(compression_bucket(sample.compression_score))

    default_thresholds, default_metrics = _optimize_thresholds(val_probs, val_confs, val_labels)
    thresholds_by_bucket: dict[str, dict[str, float]] = {"default": default_thresholds}
    bucket_metrics: dict[str, dict[str, float]] = {"default": default_metrics}
    for bucket_name in ("low", "medium", "high"):
        idx = [index for index, value in enumerate(val_buckets) if value == bucket_name]
        if len(idx) < 10:
            continue
        labels_bucket = [val_labels[i] for i in idx]
        if len(set(labels_bucket)) < 2:
            continue
        probs_bucket = [val_probs[i] for i in idx]
        confs_bucket = [val_confs[i] for i in idx]
        best_thresholds, best_metrics = _optimize_thresholds(probs_bucket, confs_bucket, labels_bucket)
        thresholds_by_bucket[bucket_name] = best_thresholds
        bucket_metrics[bucket_name] = best_metrics

    # Subtype model: ai_generated (1) vs ai_edited (0), trained only on AI samples.
    subtype_artifact: dict[str, object] = {}
    ai_train = [sample for sample in train_samples if sample.subtype_label is not None]
    ai_val = [sample for sample in val_samples if sample.subtype_label is not None]
    if len(ai_train) >= 16 and len(ai_val) >= 8:
        subtype_labels_train = [int(sample.subtype_label or 0) for sample in ai_train]
        subtype_labels_val = [int(sample.subtype_label or 0) for sample in ai_val]
        if len(set(subtype_labels_train)) == 2 and len(set(subtype_labels_val)) == 2:
            subtype_model = _fit_linear_model(
                ai_train,
                subtype_labels_train,
                SUBTYPE_FEATURE_NAMES,
                epochs=max(300, int(args.epochs * 0.7)),
                learning_rate=max(0.02, args.learning_rate * 0.7),
                l2=max(0.0001, args.l2 * 1.5),
            )
            subtype_logits = _linear_logits(ai_val, subtype_model)
            subtype_a, subtype_b = _fit_platt_scaler(subtype_logits, subtype_labels_val)
            subtype_probs = [apply_platt(float(v), subtype_a, subtype_b) for v in subtype_logits.tolist()]
            subtype_truth = subtype_labels_val
            subtype_pred = [1 if p >= 0.5 else 0 for p in subtype_probs]
            subtype_acc = sum(int(t == p) for t, p in zip(subtype_truth, subtype_pred)) / max(1, len(subtype_truth))
            subtype_ece = _ece_binary(subtype_probs, subtype_truth)
            subtype_artifact = {
                "feature_names": list(subtype_model.feature_names),
                "means": [float(v) for v in subtype_model.means],
                "stds": [float(v) for v in subtype_model.stds],
                "weights": [float(v) for v in subtype_model.weights],
                "bias": float(subtype_model.bias),
                "calibrator": {"a": float(subtype_a), "b": float(subtype_b)},
                "validation": {
                    "accuracy": round(float(subtype_acc), 4),
                    "ece": round(float(subtype_ece), 4),
                    "num_val": len(subtype_truth),
                },
            }

    # 3-way validation metrics from calibrated probabilities.
    val_truth: list[str] = []
    val_pred: list[str] = []
    val_prob_rows: list[dict[str, float]] = []
    for sample, binary_logit in zip(val_samples, val_logits.tolist()):
        p_ai = apply_platt(float(binary_logit), calibrator_a, calibrator_b)
        p_generated = _heuristic_subtype_generated_probability(sample.feature_map)
        if subtype_artifact:
            subtype_vector = vectorize_features(sample.feature_map, subtype_artifact["feature_names"])
            subtype_norm = standardize_vector(
                subtype_vector,
                subtype_artifact["means"],
                subtype_artifact["stds"],
            )
            subtype_logit = float(subtype_artifact["bias"]) + sum(
                float(weight) * float(value)
                for weight, value in zip(subtype_artifact["weights"], subtype_norm)
            )
            subtype_cal = subtype_artifact.get("calibrator", {})
            subtype_a = float(subtype_cal.get("a", 1.0))
            subtype_b = float(subtype_cal.get("b", 0.0))
            p_generated = apply_platt(subtype_logit, subtype_a, subtype_b)

        probs = _three_way_probabilities(p_ai, p_generated)
        predicted_class = max(probs.items(), key=lambda item: item[1])[0]
        val_truth.append(sample.class_name)
        val_pred.append(predicted_class)
        val_prob_rows.append(probs)

    three_way_metrics = _evaluate_three_way(val_truth, val_pred, val_prob_rows)
    binary_ece = _ece_binary(val_probs, val_labels)

    artifact = {
        "model_id": "fusion_logreg_v2",
        "feature_names": list(binary_model.feature_names),
        "means": [float(v) for v in binary_model.means],
        "stds": [float(v) for v in binary_model.stds],
        "weights": [float(v) for v in binary_model.weights],
        "bias": float(binary_model.bias),
        "calibrator": {
            "a": float(calibrator_a),
            "b": float(calibrator_b),
        },
        "thresholds": thresholds_by_bucket,
        "subtype": subtype_artifact,
        "training": {
            "dataset_root": str(dataset_root),
            "counts": dict(counts),
            "num_total": len(samples),
            "num_train": len(train_samples),
            "num_val": len(val_samples),
            "seed": int(args.seed),
            "val_ratio": float(args.val_ratio),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "l2": float(args.l2),
            "validation_metrics": bucket_metrics,
            "binary_calibration": {
                "ece": round(float(binary_ece), 4),
                "platt_a": round(float(calibrator_a), 6),
                "platt_b": round(float(calibrator_b), 6),
            },
            "three_way_metrics": three_way_metrics,
        },
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "saved_model": str(out_path),
                "counts": dict(counts),
                "thresholds": thresholds_by_bucket,
                "three_way_metrics": three_way_metrics,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
