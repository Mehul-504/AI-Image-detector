from __future__ import annotations

from pathlib import Path

from aidetector.schemas import Category
from aidetector.schemas import CategoryScore, OverrideHints

from .base import DetectionContext, DetectionResult, Detector
from .signal_utils import clamp01, gaussian_blur, load_image_signals, radial_spectrum_features


class WatermarkLayerDetector(Detector):
    name = "watermark_layer"

    _watermark_terms = (
        "watermark",
        "synthid",
        "stegastamp",
        "invisible watermark",
        "content credentials",
    )
    _ai_terms = (
        "stable diffusion",
        "midjourney",
        "dall-e",
        "dalle",
        "generated",
        "synthetic",
    )

    def _text_markers(self, media_uri: str | None) -> tuple[bool, bool]:
        if not media_uri:
            return False, False
        path = Path(media_uri)
        if not path.exists():
            return False, False
        try:
            text = path.read_bytes()[:1_000_000].decode("latin-1", errors="ignore").lower()
        except Exception:
            text = ""
        has_wm = any(term in text for term in self._watermark_terms)
        has_ai = any(term in text for term in self._ai_terms)
        return has_wm, has_ai

    def _score_watermark(self, context: DetectionContext) -> tuple[CategoryScore, OverrideHints]:
        bundle, err = load_image_signals(context.media_uri, max_side=768)
        has_wm_text, has_ai_text = self._text_markers(context.media_uri)
        if bundle is None:
            risk = 70.0 if has_wm_text else (58.0 if has_ai_text else 48.0)
            conf = 62.0 if has_wm_text else 20.0
            hints = OverrideHints(watermark_strong_positive=has_wm_text)
            return (
                CategoryScore(
                    risk=risk,
                    confidence=conf,
                    rationale=f"watermark heuristic limited: {err or 'unknown'}",
                ),
                hints,
            )

        np = __import__("numpy")
        gray = bundle.gray
        residual = gray - gaussian_blur(gray)
        spectrum = radial_spectrum_features(residual)
        peak_ratio = spectrum["peak_ratio"]
        periodic_score = clamp01((peak_ratio - 1.3) / 1.8)

        lsb = ((bundle.rgb * 255.0).astype(np.uint8) & 1).astype(np.float32)
        lsb_bias = abs(float(lsb.mean()) - 0.5) * 2.0
        lsb_score = clamp01((lsb_bias - 0.02) / 0.18)

        meta_score = 0.0
        if has_wm_text:
            meta_score = 1.0
        elif has_ai_text:
            meta_score = 0.45

        risk01 = clamp01(0.52 * periodic_score + 0.18 * lsb_score + 0.30 * meta_score)
        conf01 = clamp01(0.25 + 0.45 * periodic_score + 0.20 * lsb_score + 0.20 * meta_score)
        risk = round(risk01 * 100.0, 2)
        confidence = round(conf01 * 100.0, 2)
        strong = risk >= 82.0 and confidence >= 72.0
        hints = OverrideHints(watermark_strong_positive=strong)
        rationale = (
            "watermark heuristic: "
            f"periodic={periodic_score:.3f}, lsb={lsb_score:.3f}, meta={meta_score:.3f}"
        )
        return CategoryScore(risk=risk, confidence=confidence, rationale=rationale), hints

    def run(self, context: DetectionContext) -> DetectionResult:
        score = context.provided_scores.get(Category.WATERMARK)
        if score is None:
            computed, hints = self._score_watermark(context)
            merged = OverrideHints(
                provenance_declares_synthetic_or_edited=context.override_hints.provenance_declares_synthetic_or_edited,
                watermark_strong_positive=(
                    context.override_hints.watermark_strong_positive or hints.watermark_strong_positive
                ),
                provenance_declares_authentic=context.override_hints.provenance_declares_authentic,
                forensic_contradiction=context.override_hints.forensic_contradiction,
            )
            return DetectionResult(scores={Category.WATERMARK: computed}, override_hints=merged)
        return DetectionResult(
            scores={Category.WATERMARK: score},
            override_hints=context.override_hints,
        )
