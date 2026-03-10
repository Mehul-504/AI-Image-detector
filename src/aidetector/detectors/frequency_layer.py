from __future__ import annotations

from aidetector.schemas import Category, CategoryScore, MediaType

from .base import DetectionContext, DetectionResult, Detector
from .signal_utils import (
    block_boundary_score,
    clamp01,
    load_image_signals,
    load_video_frames,
    radial_spectrum_features,
)


class FrequencyLayerDetector(Detector):
    name = "frequency_layer"

    def _score_gray(self, gray) -> CategoryScore:
        spectrum = radial_spectrum_features(gray)
        hf = spectrum["hf_ratio"]
        flatness = spectrum["flatness"]
        peak = spectrum["peak_ratio"]
        block = block_boundary_score(gray)

        hf_score = clamp01((hf - 0.34) / 0.28)
        flat_score = clamp01((0.42 - flatness) / 0.35)
        peak_score = clamp01((peak - 1.2) / 2.0)
        block_score = clamp01((block - 1.0) / 0.9)

        risk01 = clamp01(
            0.18 * hf_score
            + 0.24 * flat_score
            + 0.36 * peak_score
            + 0.22 * block_score
        )
        confidence01 = clamp01(0.30 + 0.20 * hf_score + 0.25 * peak_score + 0.25 * block_score)
        rationale = (
            "frequency(fft/dft): "
            f"hf={hf:.3f}, flat={flatness:.3f}, peak={peak:.3f}, block={block:.3f}"
        )
        return CategoryScore(
            risk=round(risk01 * 100.0, 2),
            confidence=round(confidence01 * 100.0, 2),
            rationale=rationale,
        )

    def _score_image(self, context: DetectionContext) -> CategoryScore:
        bundle, err = load_image_signals(context.media_uri, max_side=1024)
        if bundle is None:
            return CategoryScore(
                risk=50.0,
                confidence=18.0,
                rationale=f"frequency unavailable: {err}",
            )
        return self._score_gray(bundle.gray)

    def _score_video(self, context: DetectionContext) -> CategoryScore:
        video, err = load_video_frames(context.media_uri, max_frames=12, max_side=320)
        if video is None:
            return CategoryScore(
                risk=50.0,
                confidence=18.0,
                rationale=f"frequency unavailable: {err}",
            )

        np = __import__("numpy")
        frame_scores = []
        for frame in video.frames_rgb:
            gray = (0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]).astype(np.float32)
            frame_scores.append(self._score_gray(gray))
        risks = np.array([score.risk for score in frame_scores], dtype=np.float32)
        confs = np.array([score.confidence for score in frame_scores], dtype=np.float32)
        risk = float(np.median(risks))
        conf = float(min(90.0, np.mean(confs) + 0.8 * min(len(frame_scores), 20)))
        rationale = (
            f"frequency(video): frames={len(frame_scores)}, "
            f"risk_med={risk:.2f}, conf_mean={float(np.mean(confs)):.2f}"
        )
        return CategoryScore(risk=round(risk, 2), confidence=round(conf, 2), rationale=rationale)

    def run(self, context: DetectionContext) -> DetectionResult:
        score = context.provided_scores.get(Category.FREQUENCY)
        if score is not None:
            return DetectionResult(scores={Category.FREQUENCY: score}, override_hints=context.override_hints)

        if context.media_type == MediaType.VIDEO:
            computed = self._score_video(context)
        else:
            computed = self._score_image(context)
        return DetectionResult(scores={Category.FREQUENCY: computed}, override_hints=context.override_hints)
