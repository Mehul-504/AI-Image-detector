from __future__ import annotations

from aidetector.schemas import Category, CategoryScore, MediaType

from .base import DetectionContext, DetectionResult, Detector
from .signal_utils import clamp01, load_video_frames


class TemporalConsistencyLayerDetector(Detector):
    name = "temporal_consistency_layer"

    def _score_video(self, context: DetectionContext) -> CategoryScore:
        video, err = load_video_frames(context.media_uri, max_frames=24, max_side=320)
        if video is None:
            return CategoryScore(risk=50.0, confidence=18.0, rationale=f"temporal unavailable: {err}")

        np = __import__("numpy")
        grays = [
            (0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]).astype(np.float32)
            for frame in video.frames_rgb
        ]
        if len(grays) < 3:
            return CategoryScore(
                risk=50.0,
                confidence=20.0,
                rationale=f"temporal unavailable: too_few_frames ({len(grays)})",
            )

        diffs = np.array(
            [float(np.mean(np.abs(grays[i] - grays[i - 1]))) for i in range(1, len(grays))],
            dtype=np.float32,
        )
        median = float(np.median(diffs))
        mad = float(np.median(np.abs(diffs - median)) + 1e-6)
        abrupt_threshold = median + 2.8 * mad
        abrupt_ratio = float((diffs > abrupt_threshold).mean())

        centered = diffs - diffs.mean()
        spectrum = np.abs(np.fft.rfft(centered))
        if spectrum.size > 2:
            periodic = float(spectrum[1:].max() / (spectrum[1:].sum() + 1e-6))
        else:
            periodic = 0.0
        smoothness = clamp01(1.0 - float(diffs.std() / (diffs.mean() + 1e-6)))

        abrupt_score = clamp01((abrupt_ratio - 0.05) / 0.45)
        periodic_score = clamp01((periodic - 0.12) / 0.55)
        smooth_break = 1.0 - smoothness

        risk01 = clamp01(0.45 * abrupt_score + 0.35 * periodic_score + 0.20 * smooth_break)
        conf01 = clamp01(0.32 + 0.40 * min(1.0, len(diffs) / 18.0) + 0.18 * abrupt_score)
        rationale = (
            "temporal(dft): "
            f"abrupt={abrupt_ratio:.3f}, periodic={periodic:.3f}, smooth={smoothness:.3f}, "
            f"fps={video.fps:.2f}"
        )
        return CategoryScore(
            risk=round(risk01 * 100.0, 2),
            confidence=round(conf01 * 100.0, 2),
            rationale=rationale,
        )

    def run(self, context: DetectionContext) -> DetectionResult:
        if context.media_type != MediaType.VIDEO:
            return DetectionResult(scores={}, override_hints=context.override_hints)
        score = context.provided_scores.get(Category.TEMPORAL_CONSISTENCY)
        if score is None:
            score = self._score_video(context)
        return DetectionResult(
            scores={Category.TEMPORAL_CONSISTENCY: score},
            override_hints=context.override_hints,
        )
