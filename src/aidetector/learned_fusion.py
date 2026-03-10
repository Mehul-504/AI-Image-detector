from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

from .config import DecisionThresholds
from .schemas import Category, CategoryScore, MediaType


MODEL_PATH_ENV = "AIDETECTOR_FUSION_MODEL_PATH"


@dataclass(frozen=True)
class LearnedFusionModel:
    model_id: str
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    thresholds: dict[str, DecisionThresholds]
    calibrator_a: float = 1.0
    calibrator_b: float = 0.0
    subtype_feature_names: tuple[str, ...] = ()
    subtype_means: tuple[float, ...] = ()
    subtype_stds: tuple[float, ...] = ()
    subtype_weights: tuple[float, ...] = ()
    subtype_bias: float = 0.0
    subtype_calibrator_a: float = 1.0
    subtype_calibrator_b: float = 0.0


@dataclass(frozen=True)
class LearnedFusionPrediction:
    risk_01: float
    confidence_01: float
    thresholds: DecisionThresholds
    bucket: str
    reason: str
    class_probabilities: dict[str, float]
    predicted_class: str


_CACHE_PATH: str | None = None
_CACHE_MTIME: float | None = None
_CACHE_MODEL: LearnedFusionModel | None = None


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def compression_bucket(score: float) -> str:
    score = clamp01(score)
    if score < 0.35:
        return "low"
    if score < 0.65:
        return "medium"
    return "high"


def _score_or_default(
    category_scores: Dict[Category, CategoryScore],
    category: Category,
) -> tuple[float, float]:
    score = category_scores.get(category)
    if score is None:
        return 0.5, 0.15
    return clamp01(score.risk / 100.0), clamp01(score.confidence / 100.0)


def _mean(values: Iterable[float], default: float) -> float:
    items = [float(v) for v in values]
    if not items:
        return float(default)
    return sum(items) / float(len(items))


def _metadata_hints(metadata: CategoryScore | None) -> tuple[float, float]:
    if metadata is None or not metadata.rationale:
        return 0.0, 0.0
    rationale = metadata.rationale.lower()
    camera_hint = 1.0 if ("camera exif" in rationale or "camera exif present" in rationale) else 0.0
    transport_hint = (
        1.0
        if ("transport-compressed" in rationale or "recompression" in rationale or "whatsapp" in rationale)
        else 0.0
    )
    return camera_hint, transport_hint


def extract_fusion_features(
    category_scores: Dict[Category, CategoryScore],
    *,
    compression_score: float = 0.0,
) -> dict[str, float]:
    provenance_risk, provenance_conf = _score_or_default(category_scores, Category.PROVENANCE)
    watermark_risk, watermark_conf = _score_or_default(category_scores, Category.WATERMARK)
    metadata_risk, metadata_conf = _score_or_default(category_scores, Category.METADATA)
    prnu_risk, prnu_conf = _score_or_default(category_scores, Category.PRNU)
    frequency_risk, frequency_conf = _score_or_default(category_scores, Category.FREQUENCY)
    spatial_risk, spatial_conf = _score_or_default(category_scores, Category.SPATIAL)
    tamper_risk, tamper_conf = _score_or_default(category_scores, Category.TAMPER_LOCALIZATION)
    clip_risk, clip_conf = _score_or_default(category_scores, Category.CLIP_ANOMALY)
    forensic_risk, forensic_conf = _score_or_default(category_scores, Category.FORENSIC_TRANSFORMER)

    confidence_mean = _mean(
        [
            provenance_conf,
            watermark_conf,
            metadata_conf,
            prnu_conf,
            frequency_conf,
            spatial_conf,
            tamper_conf,
            clip_conf,
            forensic_conf,
        ],
        default=0.15,
    )
    deterministic_risk = _mean([prnu_risk, frequency_risk, spatial_risk, tamper_risk], default=0.5)
    deterministic_conf = _mean([prnu_conf, frequency_conf, spatial_conf, tamper_conf], default=0.15)
    transformer_risk = _mean([clip_risk, forensic_risk], default=0.5)
    transformer_conf = _mean([clip_conf, forensic_conf], default=0.15)
    provenance_stack = _mean([provenance_risk, watermark_risk, metadata_risk], default=0.5)

    metadata_camera_hint, metadata_transport_hint = _metadata_hints(category_scores.get(Category.METADATA))
    compression_score = clamp01(compression_score)
    features = {
        "provenance_risk": provenance_risk,
        "watermark_risk": watermark_risk,
        "metadata_risk": metadata_risk,
        "prnu_risk": prnu_risk,
        "frequency_risk": frequency_risk,
        "spatial_risk": spatial_risk,
        "tamper_risk": tamper_risk,
        "clip_risk": clip_risk,
        "forensic_risk": forensic_risk,
        "provenance_conf": provenance_conf,
        "watermark_conf": watermark_conf,
        "metadata_conf": metadata_conf,
        "prnu_conf": prnu_conf,
        "frequency_conf": frequency_conf,
        "spatial_conf": spatial_conf,
        "tamper_conf": tamper_conf,
        "clip_conf": clip_conf,
        "forensic_conf": forensic_conf,
        "deterministic_risk_mean": deterministic_risk,
        "deterministic_conf_mean": deterministic_conf,
        "transformer_risk_mean": transformer_risk,
        "transformer_conf_mean": transformer_conf,
        "provenance_stack_mean": provenance_stack,
        "clip_forensic_gap": clip_risk - forensic_risk,
        "prnu_frequency_gap": prnu_risk - frequency_risk,
        "confidence_mean": confidence_mean,
        "compression_score": compression_score,
        "compression_x_forensic": compression_score * forensic_risk,
        "compression_x_clip": compression_score * clip_risk,
        "metadata_camera_hint": metadata_camera_hint,
        "metadata_transport_hint": metadata_transport_hint,
    }
    return {name: float(value) for name, value in features.items()}


def _confidence_from_probability(
    probability_01: float,
    confidence_hint_01: float,
    compression_score: float,
) -> float:
    margin = abs(probability_01 - 0.5) * 2.0
    confidence = 0.44 * clamp01(confidence_hint_01) + 0.46 * margin + 0.10
    confidence *= 1.0 - 0.22 * clamp01(compression_score)
    return clamp01(confidence)


def confidence_from_probability(
    probability_01: float,
    confidence_hint_01: float,
    compression_score: float,
) -> float:
    return _confidence_from_probability(
        probability_01=probability_01,
        confidence_hint_01=confidence_hint_01,
        compression_score=compression_score,
    )


def vectorize_features(
    feature_map: dict[str, float],
    feature_names: Iterable[str],
) -> list[float]:
    return [float(feature_map.get(name, 0.0)) for name in feature_names]


def standardize_vector(
    values: Iterable[float],
    means: Iterable[float],
    stds: Iterable[float],
) -> list[float]:
    return [
        (float(value) - float(mean)) / max(1e-6, float(std))
        for value, mean, std in zip(values, means, stds)
    ]


def predict_probability_from_linear(
    normalized_values: Iterable[float],
    weights: Iterable[float],
    bias: float,
) -> float:
    logit = float(bias) + sum(float(weight) * float(value) for weight, value in zip(weights, normalized_values))
    return clamp01(sigmoid(logit))


def apply_platt(logit: float, a: float, b: float) -> float:
    return clamp01(sigmoid(float(a) * float(logit) + float(b)))


def _decision_thresholds_for_bucket(
    model: LearnedFusionModel,
    compression_score: float,
) -> tuple[str, DecisionThresholds]:
    bucket = compression_bucket(compression_score)
    if bucket in model.thresholds:
        return bucket, model.thresholds[bucket]
    return bucket, model.thresholds["default"]


def _coerce_thresholds(raw: object) -> DecisionThresholds:
    if not isinstance(raw, dict):
        return DecisionThresholds()
    risk_low = float(raw.get("risk_low", DecisionThresholds().risk_low))
    risk_high = float(raw.get("risk_high", DecisionThresholds().risk_high))
    min_confidence = float(raw.get("min_confidence", DecisionThresholds().min_confidence))
    return DecisionThresholds(risk_low=risk_low, risk_high=risk_high, min_confidence=min_confidence)


def _parse_model_payload(payload: dict[str, object]) -> LearnedFusionModel | None:
    feature_names = payload.get("feature_names")
    means = payload.get("means")
    stds = payload.get("stds")
    weights = payload.get("weights")
    bias = payload.get("bias")
    thresholds_raw = payload.get("thresholds", {})
    if not isinstance(feature_names, list):
        return None
    if not isinstance(means, list) or not isinstance(stds, list) or not isinstance(weights, list):
        return None
    if len(feature_names) != len(means) or len(feature_names) != len(stds) or len(feature_names) != len(weights):
        return None
    if not isinstance(thresholds_raw, dict):
        thresholds_raw = {}
    thresholds = {"default": _coerce_thresholds(thresholds_raw.get("default", {}))}
    for bucket_name in ("low", "medium", "high"):
        if bucket_name in thresholds_raw:
            thresholds[bucket_name] = _coerce_thresholds(thresholds_raw[bucket_name])
    model_id = str(payload.get("model_id", "logreg_v1"))
    calibrator = payload.get("calibrator", {})
    calibrator_a = float(calibrator.get("a", 1.0)) if isinstance(calibrator, dict) else 1.0
    calibrator_b = float(calibrator.get("b", 0.0)) if isinstance(calibrator, dict) else 0.0

    subtype = payload.get("subtype", {})
    subtype_feature_names: tuple[str, ...] = ()
    subtype_means: tuple[float, ...] = ()
    subtype_stds: tuple[float, ...] = ()
    subtype_weights: tuple[float, ...] = ()
    subtype_bias = 0.0
    subtype_calibrator_a = 1.0
    subtype_calibrator_b = 0.0
    if isinstance(subtype, dict):
        sf = subtype.get("feature_names", [])
        sm = subtype.get("means", [])
        ss = subtype.get("stds", [])
        sw = subtype.get("weights", [])
        sb = subtype.get("bias", 0.0)
        sc = subtype.get("calibrator", {})
        if (
            isinstance(sf, list)
            and isinstance(sm, list)
            and isinstance(ss, list)
            and isinstance(sw, list)
            and len(sf) == len(sm) == len(ss) == len(sw)
            and len(sf) > 0
        ):
            subtype_feature_names = tuple(str(name) for name in sf)
            subtype_means = tuple(float(value) for value in sm)
            subtype_stds = tuple(max(1e-6, float(value)) for value in ss)
            subtype_weights = tuple(float(value) for value in sw)
            subtype_bias = float(sb)
            if isinstance(sc, dict):
                subtype_calibrator_a = float(sc.get("a", 1.0))
                subtype_calibrator_b = float(sc.get("b", 0.0))
    try:
        return LearnedFusionModel(
            model_id=model_id,
            feature_names=tuple(str(name) for name in feature_names),
            means=tuple(float(value) for value in means),
            stds=tuple(max(1e-6, float(value)) for value in stds),
            weights=tuple(float(value) for value in weights),
            bias=float(bias),
            thresholds=thresholds,
            calibrator_a=calibrator_a,
            calibrator_b=calibrator_b,
            subtype_feature_names=subtype_feature_names,
            subtype_means=subtype_means,
            subtype_stds=subtype_stds,
            subtype_weights=subtype_weights,
            subtype_bias=subtype_bias,
            subtype_calibrator_a=subtype_calibrator_a,
            subtype_calibrator_b=subtype_calibrator_b,
        )
    except (TypeError, ValueError):
        return None


def load_learned_fusion_model() -> LearnedFusionModel | None:
    global _CACHE_MODEL, _CACHE_MTIME, _CACHE_PATH

    model_path = os.getenv(MODEL_PATH_ENV, "").strip()
    if not model_path:
        return None

    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    if _CACHE_MODEL is not None and _CACHE_PATH == str(path) and _CACHE_MTIME == mtime:
        return _CACHE_MODEL

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None
    parsed = _parse_model_payload(payload)
    if parsed is None:
        return None
    _CACHE_PATH = str(path)
    _CACHE_MTIME = mtime
    _CACHE_MODEL = parsed
    return parsed


def predict_with_learned_fusion(
    *,
    media_type: MediaType,
    category_scores: Dict[Category, CategoryScore],
    aux_features: dict[str, float] | None,
) -> LearnedFusionPrediction | None:
    if media_type != MediaType.IMAGE:
        return None
    model = load_learned_fusion_model()
    if model is None:
        return None

    compression_score = clamp01(float((aux_features or {}).get("compression_score", 0.0)))
    feature_map = extract_fusion_features(category_scores, compression_score=compression_score)
    vector = vectorize_features(feature_map, model.feature_names)
    normalized = standardize_vector(vector, model.means, model.stds)
    raw_logit = float(model.bias) + sum(
        float(weight) * float(value)
        for weight, value in zip(model.weights, normalized)
    )
    probability_01 = apply_platt(raw_logit, model.calibrator_a, model.calibrator_b)
    confidence_hint = feature_map.get("confidence_mean", 0.2)
    confidence_01 = _confidence_from_probability(
        probability_01=probability_01,
        confidence_hint_01=confidence_hint,
        compression_score=compression_score,
    )

    subtype_generated = 0.5
    if model.subtype_feature_names:
        sub_vector = vectorize_features(feature_map, model.subtype_feature_names)
        sub_norm = standardize_vector(sub_vector, model.subtype_means, model.subtype_stds)
        sub_logit = float(model.subtype_bias) + sum(
            float(weight) * float(value)
            for weight, value in zip(model.subtype_weights, sub_norm)
        )
        subtype_generated = apply_platt(sub_logit, model.subtype_calibrator_a, model.subtype_calibrator_b)
    else:
        generated_signal = 0.42 * feature_map.get("frequency_risk", 0.5) + 0.32 * feature_map.get(
            "forensic_risk", 0.5
        ) + 0.26 * feature_map.get("clip_risk", 0.5)
        edited_signal = 0.38 * feature_map.get("tamper_risk", 0.5) + 0.32 * feature_map.get(
            "compression_score", 0.0
        ) + 0.30 * feature_map.get("metadata_transport_hint", 0.0)
        subtype_generated = clamp01(sigmoid(4.0 * (generated_signal - edited_signal)))

    p_real = clamp01(1.0 - probability_01)
    p_ai_generated = clamp01(probability_01 * subtype_generated)
    p_ai_edited = clamp01(probability_01 * (1.0 - subtype_generated))
    total = p_real + p_ai_generated + p_ai_edited
    if total > 0:
        p_real /= total
        p_ai_generated /= total
        p_ai_edited /= total
    class_probs = {
        "real": round(p_real, 6),
        "ai_generated": round(p_ai_generated, 6),
        "ai_edited": round(p_ai_edited, 6),
    }
    predicted_class = max(class_probs.items(), key=lambda item: item[1])[0]

    bucket, thresholds = _decision_thresholds_for_bucket(model, compression_score)
    return LearnedFusionPrediction(
        risk_01=probability_01,
        confidence_01=confidence_01,
        thresholds=thresholds,
        bucket=bucket,
        reason=f"learned_fusion:{model.model_id}:bucket={bucket}",
        class_probabilities=class_probs,
        predicted_class=predicted_class,
    )
