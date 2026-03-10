from __future__ import annotations

from dataclasses import dataclass
import os

from .schemas import Category, MediaType, ModeProfile


IMAGE_WEIGHTS: dict[Category, float] = {
    Category.PROVENANCE: 0.16,
    Category.WATERMARK: 0.07,
    Category.METADATA: 0.08,
    Category.PRNU: 0.12,
    Category.FREQUENCY: 0.10,
    Category.SPATIAL: 0.10,
    Category.TAMPER_LOCALIZATION: 0.08,
    Category.CLIP_ANOMALY: 0.12,
    Category.FORENSIC_TRANSFORMER: 0.17,
}

VIDEO_WEIGHTS: dict[Category, float] = {
    Category.PROVENANCE: 0.13,
    Category.WATERMARK: 0.05,
    Category.METADATA: 0.05,
    Category.PRNU: 0.07,
    Category.FREQUENCY: 0.08,
    Category.SPATIAL: 0.07,
    Category.CLIP_ANOMALY: 0.10,
    Category.FORENSIC_TRANSFORMER: 0.18,
    Category.AUDIO_VISUAL_TRANSFORMER: 0.15,
    Category.TEMPORAL_CONSISTENCY: 0.07,
    Category.TAMPER_LOCALIZATION: 0.05,
}


@dataclass(frozen=True)
class DecisionThresholds:
    # Calibrated for mixed real / AI-generated / AI-edited image sets.
    # Goal: reduce "always suspicious" collapse while preserving AI recall.
    risk_low: float = 0.40
    risk_high: float = 0.56
    min_confidence: float = 0.38


DEFAULT_THRESHOLDS = DecisionThresholds()


def weights_for_media_type(media_type: MediaType) -> dict[Category, float]:
    if media_type == MediaType.IMAGE:
        return IMAGE_WEIGHTS.copy()
    if media_type == MediaType.VIDEO:
        return VIDEO_WEIGHTS.copy()
    raise ValueError(f"Unsupported media type: {media_type}")


def resolve_mode_profile(mode_profile: ModeProfile | None) -> ModeProfile:
    if mode_profile is not None:
        return mode_profile
    raw = os.getenv("AIDETECTOR_MODE_PROFILE", ModeProfile.BALANCED.value).strip().lower()
    try:
        return ModeProfile(raw)
    except ValueError:
        return ModeProfile.BALANCED


def thresholds_for_mode(base: DecisionThresholds, mode_profile: ModeProfile) -> DecisionThresholds:
    if mode_profile == ModeProfile.BALANCED:
        return base
    # STRICT mode emphasizes AI recall and conservative authenticity claims.
    return DecisionThresholds(
        risk_low=max(0.20, base.risk_low - 0.04),
        risk_high=max(base.risk_low + 0.08, base.risk_high - 0.05),
        min_confidence=min(0.90, base.min_confidence + 0.05),
    )
