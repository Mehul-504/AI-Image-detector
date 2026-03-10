from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class ModeProfile(str, Enum):
    BALANCED = "balanced"
    STRICT = "strict"


class Verdict(str, Enum):
    LIKELY_AUTHENTIC = "likely_authentic"
    SUSPICIOUS = "suspicious"
    LIKELY_SYNTHETIC_OR_EDITED = "likely_synthetic_or_edited"


class Category(str, Enum):
    PROVENANCE = "provenance"
    WATERMARK = "watermark"
    METADATA = "metadata"
    PRNU = "prnu"
    FREQUENCY = "frequency"
    SPATIAL = "spatial"
    TAMPER_LOCALIZATION = "tamper_localization"
    CLIP_ANOMALY = "clip_anomaly"
    FORENSIC_TRANSFORMER = "forensic_transformer"
    AUDIO_VISUAL_TRANSFORMER = "audio_visual_transformer"
    TEMPORAL_CONSISTENCY = "temporal_consistency"


@dataclass(frozen=True)
class CategoryScore:
    risk: float
    confidence: float
    rationale: Optional[str] = None

    def __post_init__(self) -> None:
        if not 0 <= self.risk <= 100:
            raise ValueError(f"risk must be in [0, 100], got {self.risk}")
        if not 0 <= self.confidence <= 100:
            raise ValueError(f"confidence must be in [0, 100], got {self.confidence}")


@dataclass(frozen=True)
class OverrideHints:
    provenance_declares_synthetic_or_edited: bool = False
    watermark_strong_positive: bool = False
    provenance_declares_authentic: bool = False
    forensic_contradiction: bool = False


@dataclass(frozen=True)
class AnalysisRequest:
    media_type: MediaType
    media_uri: Optional[str] = None
    provided_scores: dict[Category, CategoryScore] = field(default_factory=dict)
    override_hints: OverrideHints = field(default_factory=OverrideHints)
    mode_profile: ModeProfile | None = None


@dataclass(frozen=True)
class CategoryContribution:
    category: Category
    weighted_contribution: float
    effective_weight: float
    score: CategoryScore


@dataclass(frozen=True)
class AnalysisResponse:
    media_type: MediaType
    verdict: Verdict
    overall_risk: float
    overall_confidence: float
    applied_override: Optional[str]
    category_scores: dict[Category, CategoryScore]
    category_contributions: list[CategoryContribution]
    predicted_class: Optional[str] = None
    class_probabilities: dict[str, float] = field(default_factory=dict)
