from __future__ import annotations

from aidetector.schemas import Category, CategoryScore, MediaType

from .base import DetectionContext, DetectionResult, Detector
from .signal_utils import clamp01, image_ela, load_image_signals, load_video_frames, patchwise_mean


class TamperLocalizationLayerDetector(Detector):
    name = "tamper_localization_layer"

    def _score_rgb(self, rgb) -> CategoryScore:
        np = __import__("numpy")
        ela, err = image_ela(rgb, jpeg_quality=90)
        if ela is None:
            return CategoryScore(risk=50.0, confidence=18.0, rationale=f"tamper unavailable: {err}")

        patch_map = patchwise_mean(ela, patch=24)
        p50 = float(np.percentile(patch_map, 50))
        p90 = float(np.percentile(patch_map, 90))
        p98 = float(np.percentile(patch_map, 98))
        if p90 <= 1e-6:
            anomaly_ratio = 0.0
        else:
            threshold = p90 + 0.35 * max(1e-6, (p98 - p90))
            anomaly_ratio = float((patch_map >= threshold).mean())
        localization_contrast = clamp01((p98 - p50) / (p98 + 1e-6))

        anomaly_score = clamp01((anomaly_ratio - 0.04) / 0.30)
        contrast_score = clamp01((localization_contrast - 0.20) / 0.55)
        risk01 = clamp01(0.62 * anomaly_score + 0.38 * contrast_score)
        conf01 = clamp01(0.32 + 0.38 * anomaly_score + 0.20 * contrast_score)
        rationale = (
            "tamper(ela): "
            f"anomaly={anomaly_ratio:.3f}, contrast={localization_contrast:.3f}, "
            f"p50={p50:.4f}, p98={p98:.4f}"
        )
        return CategoryScore(
            risk=round(risk01 * 100.0, 2),
            confidence=round(conf01 * 100.0, 2),
            rationale=rationale,
        )

    def _score_image(self, context: DetectionContext) -> CategoryScore:
        bundle, err = load_image_signals(context.media_uri, max_side=1024)
        if bundle is None:
            return CategoryScore(risk=50.0, confidence=18.0, rationale=f"tamper unavailable: {err}")
        return self._score_rgb(bundle.rgb)

    def _score_video(self, context: DetectionContext) -> CategoryScore:
        video, err = load_video_frames(context.media_uri, max_frames=8, max_side=320)
        if video is None:
            return CategoryScore(risk=50.0, confidence=18.0, rationale=f"tamper unavailable: {err}")
        np = __import__("numpy")
        results = [self._score_rgb(frame) for frame in video.frames_rgb]
        risks = np.array([r.risk for r in results], dtype=np.float32)
        confs = np.array([r.confidence for r in results], dtype=np.float32)
        rationale = (
            f"tamper(video): frames={len(results)}, risk_med={float(np.median(risks)):.2f}, "
            f"conf_mean={float(np.mean(confs)):.2f}"
        )
        return CategoryScore(
            risk=round(float(np.median(risks)), 2),
            confidence=round(float(min(88.0, np.mean(confs) + len(results))), 2),
            rationale=rationale,
        )

    def run(self, context: DetectionContext) -> DetectionResult:
        score = context.provided_scores.get(Category.TAMPER_LOCALIZATION)
        if score is not None:
            return DetectionResult(
                scores={Category.TAMPER_LOCALIZATION: score},
                override_hints=context.override_hints,
            )
        if context.media_type == MediaType.VIDEO:
            computed = self._score_video(context)
        else:
            computed = self._score_image(context)
        return DetectionResult(
            scores={Category.TAMPER_LOCALIZATION: computed},
            override_hints=context.override_hints,
        )
