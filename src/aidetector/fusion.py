from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .config import (
    DEFAULT_THRESHOLDS,
    DecisionThresholds,
    resolve_mode_profile,
    thresholds_for_mode,
    weights_for_media_type,
)
from .learned_fusion import predict_with_learned_fusion
from .schemas import (
    AnalysisResponse,
    Category,
    CategoryContribution,
    CategoryScore,
    MediaType,
    ModeProfile,
    OverrideHints,
    Verdict,
)


@dataclass(frozen=True)
class FusionSnapshot:
    overall_risk_01: float
    overall_confidence_01: float
    effective_weights: dict[Category, float]
    contribution_shares: dict[Category, float]


def _redistribute_weights(
    base_weights: Dict[Category, float], available: set[Category]
) -> Dict[Category, float]:
    selected = {category: weight for category, weight in base_weights.items() if category in available}
    total = sum(selected.values())
    if total <= 0:
        return {}
    return {category: weight / total for category, weight in selected.items()}


def _compute_snapshot(
    media_type: MediaType,
    category_scores: Dict[Category, CategoryScore],
) -> FusionSnapshot:
    weights = weights_for_media_type(media_type)
    redistributed = _redistribute_weights(weights, set(category_scores))
    if not redistributed:
        return FusionSnapshot(
            overall_risk_01=0.5,
            overall_confidence_01=0.0,
            effective_weights={},
            contribution_shares={},
        )

    weighted_conf_sum = 0.0
    weighted_conf_risk_sum = 0.0
    raw_contrib: dict[Category, float] = {}
    for category, eff_weight in redistributed.items():
        score = category_scores[category]
        confidence_01 = score.confidence / 100.0
        risk_01 = score.risk / 100.0
        weighted_conf = eff_weight * confidence_01
        contribution = weighted_conf * risk_01
        weighted_conf_sum += weighted_conf
        weighted_conf_risk_sum += contribution
        raw_contrib[category] = contribution

    if weighted_conf_sum <= 0:
        return FusionSnapshot(
            overall_risk_01=0.5,
            overall_confidence_01=0.0,
            effective_weights=redistributed,
            contribution_shares={category: 0.0 for category in redistributed},
        )

    overall_risk = weighted_conf_risk_sum / weighted_conf_sum
    total_contrib = sum(raw_contrib.values())
    if total_contrib <= 0:
        normalized_contrib = {category: 0.0 for category in raw_contrib}
    else:
        normalized_contrib = {category: value / total_contrib for category, value in raw_contrib.items()}

    return FusionSnapshot(
        overall_risk_01=overall_risk,
        overall_confidence_01=weighted_conf_sum,
        effective_weights=redistributed,
        contribution_shares=normalized_contrib,
    )


def _apply_hard_override(hints: OverrideHints) -> tuple[Verdict | None, str | None]:
    if hints.provenance_declares_synthetic_or_edited:
        return Verdict.LIKELY_SYNTHETIC_OR_EDITED, "trusted_provenance_says_synthetic_or_edited"
    if hints.watermark_strong_positive:
        return Verdict.LIKELY_SYNTHETIC_OR_EDITED, "strong_watermark_positive"
    if hints.provenance_declares_authentic and not hints.forensic_contradiction:
        return Verdict.LIKELY_AUTHENTIC, "trusted_provenance_says_authentic"
    return None, None


def _matrix_verdict(
    overall_risk_01: float,
    overall_confidence_01: float,
    thresholds: DecisionThresholds,
) -> Verdict:
    if overall_confidence_01 < thresholds.min_confidence:
        return Verdict.SUSPICIOUS
    if overall_risk_01 < thresholds.risk_low:
        return Verdict.LIKELY_AUTHENTIC
    if overall_risk_01 >= thresholds.risk_high:
        return Verdict.LIKELY_SYNTHETIC_OR_EDITED
    return Verdict.SUSPICIOUS


def fuse_to_verdict(
    media_type: MediaType,
    category_scores: Dict[Category, CategoryScore],
    override_hints: OverrideHints,
    aux_features: dict[str, float] | None = None,
    mode_profile: ModeProfile | None = None,
    thresholds: DecisionThresholds = DEFAULT_THRESHOLDS,
) -> AnalysisResponse:
    snapshot = _compute_snapshot(media_type, category_scores)
    resolved_mode = resolve_mode_profile(mode_profile)
    learned_prediction = predict_with_learned_fusion(
        media_type=media_type,
        category_scores=category_scores,
        aux_features=aux_features,
    )
    effective_risk_01 = (
        learned_prediction.risk_01 if learned_prediction is not None else snapshot.overall_risk_01
    )
    effective_confidence_01 = (
        learned_prediction.confidence_01 if learned_prediction is not None else snapshot.overall_confidence_01
    )
    base_thresholds = learned_prediction.thresholds if learned_prediction is not None else thresholds
    effective_thresholds = thresholds_for_mode(base_thresholds, resolved_mode)
    overridden_verdict, override_reason = _apply_hard_override(override_hints)
    verdict = overridden_verdict or _matrix_verdict(
        overall_risk_01=effective_risk_01,
        overall_confidence_01=effective_confidence_01,
        thresholds=effective_thresholds,
    )

    contributions = [
        CategoryContribution(
            category=category,
            weighted_contribution=round(snapshot.contribution_shares.get(category, 0.0), 6),
            effective_weight=round(snapshot.effective_weights.get(category, 0.0), 6),
            score=score,
        )
        for category, score in category_scores.items()
    ]
    contributions.sort(key=lambda item: item.weighted_contribution, reverse=True)

    predicted_class: str | None = None
    class_probabilities: dict[str, float] = {}
    if learned_prediction is not None:
        class_probabilities = dict(learned_prediction.class_probabilities)
        predicted_class = learned_prediction.predicted_class
        if verdict == Verdict.LIKELY_AUTHENTIC:
            predicted_class = "real"
        elif verdict != Verdict.LIKELY_AUTHENTIC and predicted_class == "real":
            predicted_class = "ai_edited"
    else:
        if verdict == Verdict.LIKELY_AUTHENTIC:
            predicted_class = "real"
            class_probabilities = {"real": 1.0, "ai_generated": 0.0, "ai_edited": 0.0}
        elif verdict == Verdict.LIKELY_SYNTHETIC_OR_EDITED:
            predicted_class = "ai_generated"
            class_probabilities = {"real": 0.0, "ai_generated": 0.6, "ai_edited": 0.4}
        else:
            predicted_class = "ai_edited"
            class_probabilities = {"real": 0.2, "ai_generated": 0.35, "ai_edited": 0.45}

    return AnalysisResponse(
        media_type=media_type,
        verdict=verdict,
        overall_risk=round(effective_risk_01 * 100.0, 2),
        overall_confidence=round(effective_confidence_01 * 100.0, 2),
        applied_override=override_reason or (learned_prediction.reason if learned_prediction else None),
        category_scores=category_scores,
        category_contributions=contributions,
        predicted_class=predicted_class,
        class_probabilities=class_probabilities,
    )
