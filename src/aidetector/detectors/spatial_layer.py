from __future__ import annotations

from aidetector.schemas import Category, CategoryScore, MediaType

from .base import DetectionContext, DetectionResult, Detector
from .signal_utils import clamp01, gaussian_blur, load_image_signals, load_video_frames, sobel_magnitude


class SpatialLayerDetector(Detector):
    name = "spatial_layer"

    def _score_rgb(self, rgb) -> CategoryScore:
        np = __import__("numpy")
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
        edges = sobel_magnitude(gray)
        edge_density = float((edges > np.percentile(edges, 75)).mean())

        # Oversmooth / over-sharp mix
        blur = gaussian_blur(gray)
        residual = np.abs(gray - blur)
        smooth_ratio = float((residual < np.percentile(residual, 35)).mean())
        sharp_ratio = float((edges > np.percentile(edges, 92)).mean())

        clipped = float(((rgb <= 0.01) | (rgb >= 0.99)).mean())

        # Channel correlation can reveal synthetic color coupling.
        r = rgb[..., 0].ravel()
        g = rgb[..., 1].ravel()
        b = rgb[..., 2].ravel()
        corr_rg = float(np.corrcoef(r, g)[0, 1]) if r.size > 10 else 0.0
        corr_rb = float(np.corrcoef(r, b)[0, 1]) if r.size > 10 else 0.0
        corr_gb = float(np.corrcoef(g, b)[0, 1]) if r.size > 10 else 0.0
        if corr_rg != corr_rg:
            corr_rg = 0.0
        if corr_rb != corr_rb:
            corr_rb = 0.0
        if corr_gb != corr_gb:
            corr_gb = 0.0
        channel_correlation = clamp01((abs(corr_rg) + abs(corr_rb) + abs(corr_gb)) / 3.0)

        smooth_score = clamp01((smooth_ratio - 0.45) / 0.35)
        sharp_score = clamp01((sharp_ratio - 0.05) / 0.25)
        clip_score = clamp01((clipped - 0.02) / 0.18)
        edge_score = clamp01((edge_density - 0.14) / 0.26)
        corr_score = clamp01((channel_correlation - 0.78) / 0.22)

        risk01 = clamp01(
            0.22 * smooth_score
            + 0.20 * sharp_score
            + 0.18 * clip_score
            + 0.18 * edge_score
            + 0.22 * corr_score
        )
        conf01 = clamp01(0.30 + 0.25 * edge_score + 0.25 * corr_score + 0.20 * sharp_score)
        rationale = (
            "spatial: "
            f"smooth={smooth_ratio:.3f}, sharp={sharp_ratio:.3f}, clip={clipped:.3f}, "
            f"edge={edge_density:.3f}, cc={channel_correlation:.3f}"
        )
        return CategoryScore(
            risk=round(risk01 * 100.0, 2),
            confidence=round(conf01 * 100.0, 2),
            rationale=rationale,
        )

    def _score_image(self, context: DetectionContext) -> CategoryScore:
        bundle, err = load_image_signals(context.media_uri, max_side=1024)
        if bundle is None:
            return CategoryScore(risk=50.0, confidence=18.0, rationale=f"spatial unavailable: {err}")
        return self._score_rgb(bundle.rgb)

    def _score_video(self, context: DetectionContext) -> CategoryScore:
        video, err = load_video_frames(context.media_uri, max_frames=10, max_side=320)
        if video is None:
            return CategoryScore(risk=50.0, confidence=18.0, rationale=f"spatial unavailable: {err}")
        np = __import__("numpy")
        results = [self._score_rgb(frame) for frame in video.frames_rgb]
        risks = np.array([r.risk for r in results], dtype=np.float32)
        confs = np.array([r.confidence for r in results], dtype=np.float32)
        rationale = (
            f"spatial(video): frames={len(results)}, risk_med={float(np.median(risks)):.2f}, "
            f"conf_mean={float(np.mean(confs)):.2f}"
        )
        return CategoryScore(
            risk=round(float(np.median(risks)), 2),
            confidence=round(float(min(90.0, np.mean(confs) + len(results))), 2),
            rationale=rationale,
        )

    def run(self, context: DetectionContext) -> DetectionResult:
        score = context.provided_scores.get(Category.SPATIAL)
        if score is not None:
            return DetectionResult(scores={Category.SPATIAL: score}, override_hints=context.override_hints)
        if context.media_type == MediaType.VIDEO:
            computed = self._score_video(context)
        else:
            computed = self._score_image(context)
        return DetectionResult(scores={Category.SPATIAL: computed}, override_hints=context.override_hints)
