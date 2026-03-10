from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .pipeline import AuthenticityPipeline
from .schemas import AnalysisRequest, Category, MediaType, ModeProfile, Verdict


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".nef",
    ".cr2",
    ".arw",
}
CLASS_NAMES = ("real", "ai_generated", "ai_edited")


def iter_labelled_images(dataset_root: Path) -> Iterable[tuple[str, Path]]:
    for label in CLASS_NAMES:
        label_dir = dataset_root / label
        if not label_dir.exists():
            continue
        for path in sorted(label_dir.iterdir()):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            if path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            yield label, path


def to_binary_label(label: str) -> str:
    return "real" if label == "real" else "ai"


def to_binary_pred(verdict: Verdict, suspicious_as_ai: bool) -> str:
    if verdict == Verdict.LIKELY_AUTHENTIC:
        return "real"
    if verdict == Verdict.SUSPICIOUS and not suspicious_as_ai:
        return "unknown"
    return "ai"


def default_class_from_verdict(verdict: Verdict) -> str:
    if verdict == Verdict.LIKELY_AUTHENTIC:
        return "real"
    if verdict == Verdict.LIKELY_SYNTHETIC_OR_EDITED:
        return "ai_generated"
    return "ai_edited"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_class_probabilities(raw: dict[str, float] | None, fallback: str) -> dict[str, float]:
    if not raw:
        return {
            "real": 1.0 if fallback == "real" else 0.0,
            "ai_generated": 1.0 if fallback == "ai_generated" else 0.0,
            "ai_edited": 1.0 if fallback == "ai_edited" else 0.0,
        }
    real = clamp01(float(raw.get("real", 0.0)))
    generated = clamp01(float(raw.get("ai_generated", 0.0)))
    edited = clamp01(float(raw.get("ai_edited", 0.0)))
    total = real + generated + edited
    if total <= 0:
        return normalize_class_probabilities(None, fallback)
    return {
        "real": real / total,
        "ai_generated": generated / total,
        "ai_edited": edited / total,
    }


def compute_three_way_metrics(
    truth: list[str],
    predicted: list[str],
    probability_rows: list[dict[str, float]],
) -> dict[str, object]:
    confusion: Counter[tuple[str, str]] = Counter()
    for t, p in zip(truth, predicted):
        confusion[(t, p)] += 1

    per_class: dict[str, dict[str, float]] = {}
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
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
            "support": int(sum(confusion[(class_name, pred_name)] for pred_name in CLASS_NAMES)),
        }

    total = len(truth)
    correct = sum(confusion[(name, name)] for name in CLASS_NAMES)
    accuracy = correct / total if total else 0.0

    brier = 0.0
    for true_name, probs in zip(truth, probability_rows):
        row = 0.0
        for name in CLASS_NAMES:
            target = 1.0 if true_name == name else 0.0
            row += (float(probs.get(name, 0.0)) - target) ** 2
        brier += row
    brier /= max(1, total)

    confusion_matrix = {
        truth_name: {
            pred_name: int(confusion[(truth_name, pred_name)])
            for pred_name in CLASS_NAMES
        }
        for truth_name in CLASS_NAMES
    }
    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(precision_sum / len(CLASS_NAMES), 4),
        "macro_recall": round(recall_sum / len(CLASS_NAMES), 4),
        "macro_f1": round(f1_sum / len(CLASS_NAMES), 4),
        "brier_3way": round(brier, 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate detector performance on local labelled dataset.")
    parser.add_argument(
        "--dataset-root",
        default="dataset",
        help="Dataset root containing real/, ai_generated/, ai_edited/ subfolders",
    )
    parser.add_argument(
        "--suspicious-as-ai",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat suspicious verdict as AI-positive for binary metrics (default: true)",
    )
    parser.add_argument(
        "--mode-profile",
        default="",
        help="Optional inference mode profile: balanced or strict",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional path to write JSON summary.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        raise SystemExit(f"dataset root not found: {dataset_root}")

    mode_profile = None
    raw_mode = str(args.mode_profile or "").strip().lower()
    if raw_mode:
        try:
            mode_profile = ModeProfile(raw_mode)
        except ValueError as exc:
            raise SystemExit(f"unsupported --mode-profile: {raw_mode}") from exc

    pipeline = AuthenticityPipeline()

    rows: list[dict[str, object]] = []
    verdict_counts: dict[str, Counter[str]] = defaultdict(Counter)
    binary_cm: Counter[tuple[str, str]] = Counter()
    category_risk_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    category_conf_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    class_truth: list[str] = []
    class_pred: list[str] = []
    class_probs: list[dict[str, float]] = []

    for label, path in iter_labelled_images(dataset_root):
        response = pipeline.analyze(
            AnalysisRequest(
                media_type=MediaType.IMAGE,
                media_uri=str(path),
                provided_scores={},
                mode_profile=mode_profile,
            )
        )
        verdict = response.verdict
        verdict_counts[label][verdict.value] += 1

        binary_truth = to_binary_label(label)
        binary_pred = to_binary_pred(verdict, suspicious_as_ai=args.suspicious_as_ai)
        if binary_pred != "unknown":
            binary_cm[(binary_truth, binary_pred)] += 1

        for category, score in response.category_scores.items():
            category_risk_values[label][category.value].append(float(score.risk))
            category_conf_values[label][category.value].append(float(score.confidence))

        predicted_class = response.predicted_class or default_class_from_verdict(verdict)
        if predicted_class not in CLASS_NAMES:
            predicted_class = default_class_from_verdict(verdict)
        probabilities = normalize_class_probabilities(response.class_probabilities, predicted_class)
        class_truth.append(label)
        class_pred.append(predicted_class)
        class_probs.append(probabilities)

        rows.append(
            {
                "label": label,
                "file": path.name,
                "verdict": verdict.value,
                "predicted_class": predicted_class,
                "class_probabilities": probabilities,
                "overall_risk": response.overall_risk,
                "overall_confidence": response.overall_confidence,
            }
        )

    tp = binary_cm[("ai", "ai")]
    tn = binary_cm[("real", "real")]
    fp = binary_cm[("real", "ai")]
    fn = binary_cm[("ai", "real")]
    total_binary = tp + tn + fp + fn
    accuracy = (tp + tn) / total_binary if total_binary else 0.0
    precision_ai = tp / (tp + fp) if (tp + fp) else 0.0
    recall_ai = tp / (tp + fn) if (tp + fn) else 0.0
    specificity_real = tn / (tn + fp) if (tn + fp) else 0.0

    suspicious_count = sum(1 for row in rows if row["verdict"] == Verdict.SUSPICIOUS.value)
    suspicious_rate = suspicious_count / max(1, len(rows))

    category_means: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    ordered_categories = [category.value for category in Category]
    for label in CLASS_NAMES:
        for category in ordered_categories:
            risks = category_risk_values[label].get(category, [])
            confs = category_conf_values[label].get(category, [])
            if not risks:
                continue
            category_means[label][category] = {
                "risk_mean": round(sum(risks) / len(risks), 4),
                "confidence_mean": round(sum(confs) / len(confs), 4),
            }

    summary = {
        "dataset_root": str(dataset_root),
        "mode_profile": mode_profile.value if mode_profile is not None else "env_or_default",
        "num_samples": len(rows),
        "verdict_counts": {label: dict(counts) for label, counts in verdict_counts.items()},
        "binary_confusion_matrix": {
            "real_as_real": tn,
            "real_as_ai": fp,
            "ai_as_real": fn,
            "ai_as_ai": tp,
        },
        "binary_metrics": {
            "accuracy": round(accuracy, 4),
            "precision_ai": round(precision_ai, 4),
            "recall_ai": round(recall_ai, 4),
            "specificity_real": round(specificity_real, 4),
            "suspicious_as_ai": bool(args.suspicious_as_ai),
        },
        "three_way_metrics": compute_three_way_metrics(class_truth, class_pred, class_probs),
        "suspicious_rate": round(suspicious_rate, 4),
        "category_means": category_means,
    }

    print(json.dumps(summary, indent=2))
    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
