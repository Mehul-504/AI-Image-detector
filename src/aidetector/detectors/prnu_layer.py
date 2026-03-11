from __future__ import annotations

from aidetector.schemas import Category, CategoryScore, MediaType

from .base import DetectionContext, DetectionResult, Detector
from .signal_utils import (
    clamp01,
    gaussian_blur,
    infer_transport_compression_signals,
    load_image_signals,
    load_video_frames,
    sobel_magnitude,
)


class PrnuLayerDetector(Detector):
    name = "prnu_layer"

    def _score_gray(
        self,
        gray,
        metadata_score: CategoryScore | None = None,
        compression_score: float = 0.0,
    ) -> CategoryScore:
        np = __import__("numpy")
        residual = gray - gaussian_blur(gray)
        edges = sobel_magnitude(gray)

        # Patch noise stationarity: camera PRNU tends to be more stationary than synthetic artifacts.
        h, w = residual.shape
        patch = max(8, min(48, h, w))
        ph = h // patch
        pw = w // patch
        cropped_h = ph * patch
        cropped_w = pw * patch
        cropped = residual[:cropped_h, :cropped_w] if cropped_h > 0 and cropped_w > 0 else None
        if (
            cropped is not None
            and ph > 0
            and pw > 0
            and cropped_h >= patch
            and cropped_w >= patch
            and cropped.size == (ph * patch * pw * patch)
        ):
            patches = cropped.reshape(ph, patch, pw, patch)
            stds = patches.std(axis=(1, 3))
            stationarity = 1.0 - float(stds.std() / (stds.mean() + 1e-6))
        else:
            stationarity = 0.5
        stationarity = clamp01(stationarity)

        edge_corr = 0.0
        res_abs = np.abs(residual).ravel()
        edge_flat = edges.ravel()
        if res_abs.size > 10:
            c = float(np.corrcoef(res_abs, edge_flat)[0, 1])
            if c == c:  # NaN guard
                edge_corr = clamp01((c + 1.0) * 0.5)

        fft = np.abs(np.fft.fftshift(np.fft.fft2(residual)))
        p99 = float(np.percentile(fft, 99.0))
        p95 = float(np.percentile(fft, 95.0)) + 1e-6
        periodic = clamp01((p99 / p95 - 1.0) / 1.5)

        risk01 = clamp01(0.62 * (1.0 - stationarity) + 0.38 * periodic)
        patch_count = max(1, int(ph) * int(pw))
        reliability = clamp01(1.0 - max(0.0, edge_corr - 0.62) / 0.38)
        conf01 = clamp01(
            (0.26 + 0.34 * min(1.0, patch_count / 20.0) + 0.22 * periodic + 0.18 * stationarity)
            * reliability
            + 0.08
        )

        calibration_note = None
        if metadata_score is not None and metadata_score.confidence >= 70.0 and metadata_score.risk <= 20.0:
            risk01 = clamp01(risk01 * 0.62)
            conf01 = clamp01(conf01 * 0.74)
            calibration_note = "camera_metadata_support"
        elif metadata_score is not None and metadata_score.confidence >= 70.0 and metadata_score.risk >= 85.0:
            risk01 = clamp01(risk01 * 0.88 + 0.10)
            conf01 = clamp01(conf01 * 1.05)
            calibration_note = "ai_metadata_support"
        elif compression_score >= 0.58:
            severity = clamp01((compression_score - 0.58) / 0.42)
            risk01 = clamp01(0.5 + (risk01 - 0.5) * (1.0 - 0.60 * severity))
            conf01 = clamp01(conf01 * (1.0 - 0.52 * severity))
            calibration_note = "transport_compression"

        rationale = (
            "prnu-like: "
            f"stationarity={stationarity:.3f}, edge_corr={edge_corr:.3f}, periodic={periodic:.3f}, "
            f"reliability={reliability:.3f}"
        )
        if calibration_note is not None:
            rationale = f"{rationale}; calibration={calibration_note}"
        return CategoryScore(
            risk=round(risk01 * 100.0, 2),
            confidence=round(conf01 * 100.0, 2),
            rationale=rationale,
        )

    def _apply_consensus_adjustment(
        self,
        score: CategoryScore,
        context: DetectionContext,
    ) -> CategoryScore:
        if score.risk < 65.0 or score.confidence < 45.0:
            return score

        def is_low(candidate: CategoryScore | None) -> bool:
            return candidate is not None and candidate.confidence >= 40.0 and candidate.risk <= 45.0

        def is_high(candidate: CategoryScore | None) -> bool:
            return candidate is not None and candidate.confidence >= 45.0 and candidate.risk >= 65.0

        frequency = context.provided_scores.get(Category.FREQUENCY)
        spatial = context.provided_scores.get(Category.SPATIAL)
        tamper = context.provided_scores.get(Category.TAMPER_LOCALIZATION)
        clip = context.provided_scores.get(Category.CLIP_ANOMALY)
        metadata = context.provided_scores.get(Category.METADATA)
        watermark = context.provided_scores.get(Category.WATERMARK)
        forensic = context.provided_scores.get(Category.FORENSIC_TRANSFORMER)

        deterministic_low = sum(1 for candidate in (frequency, spatial, tamper) if is_low(candidate))
        deterministic_high = sum(1 for candidate in (frequency, spatial, tamper) if is_high(candidate))
        clip_strong_ai = clip is not None and clip.confidence >= 55.0 and clip.risk >= 65.0
        clip_non_ai = clip is not None and clip.confidence >= 28.0 and clip.risk <= 50.0
        metadata_auth = metadata is not None and metadata.confidence >= 40.0 and metadata.risk <= 45.0
        metadata_ai = metadata is not None and metadata.confidence >= 70.0 and metadata.risk >= 80.0
        watermark_low = is_low(watermark)
        forensic_high = forensic is not None and forensic.confidence >= 55.0 and forensic.risk >= 75.0

        risk = score.risk
        confidence = score.confidence
        consensus_note = None

        if deterministic_low >= 3 and not clip_strong_ai and not metadata_ai:
            strong_real_support = metadata_auth or clip_non_ai or watermark_low
            risk = max(7.0, risk * (0.46 if strong_real_support else 0.58))
            confidence = max(18.0, confidence * (0.52 if strong_real_support else 0.62))
            consensus_note = "cross_signal_conflict_strong"
        elif (
            deterministic_low >= 2
            and not clip_strong_ai
            and not metadata_ai
            and (clip_non_ai or metadata_auth or forensic_high)
        ):
            risk = max(10.0, risk * 0.70)
            confidence = max(22.0, confidence * 0.74)
            consensus_note = "cross_signal_conflict"
        elif deterministic_high >= 2 or clip_strong_ai or metadata_ai:
            risk = min(99.0, risk * 0.95 + 3.0)
            confidence = min(92.0, confidence * 1.03 + 1.0)
            consensus_note = "cross_signal_support"

        if consensus_note is None:
            return score

        return CategoryScore(
            risk=round(risk, 2),
            confidence=round(confidence, 2),
            rationale=f"{score.rationale}; consensus={consensus_note}",
        )

    def _score_image(self, context: DetectionContext) -> CategoryScore:
        bundle, err = load_image_signals(context.media_uri, max_side=1024)
        if bundle is None:
            return CategoryScore(risk=50.0, confidence=18.0, rationale=f"prnu unavailable: {err}")
        compression = infer_transport_compression_signals(context.media_uri, gray=bundle.gray)
        base = self._score_gray(
            bundle.gray,
            metadata_score=context.provided_scores.get(Category.METADATA),
            compression_score=float(compression.get("score", 0.0)),
        )
        return self._apply_consensus_adjustment(base, context)

    def _score_video(self, context: DetectionContext) -> CategoryScore:
        video, err = load_video_frames(context.media_uri, max_frames=10, max_side=320)
        if video is None:
            return CategoryScore(risk=50.0, confidence=18.0, rationale=f"prnu unavailable: {err}")
        np = __import__("numpy")
        results = []
        metadata = context.provided_scores.get(Category.METADATA)
        compression = infer_transport_compression_signals(context.media_uri)
        compression_score = float(compression.get("score", 0.0))
        for frame in video.frames_rgb:
            gray = (0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]).astype(np.float32)
            results.append(
                self._score_gray(
                    gray,
                    metadata_score=metadata,
                    compression_score=compression_score,
                )
            )
        risks = np.array([r.risk for r in results], dtype=np.float32)
        confs = np.array([r.confidence for r in results], dtype=np.float32)
        rationale = (
            f"prnu(video): frames={len(results)}, risk_med={float(np.median(risks)):.2f}, "
            f"conf_mean={float(np.mean(confs)):.2f}"
        )
        base = CategoryScore(
            risk=round(float(np.median(risks)), 2),
            confidence=round(float(min(92.0, np.mean(confs) + len(results))), 2),
            rationale=rationale,
        )
        return self._apply_consensus_adjustment(base, context)

    def run(self, context: DetectionContext) -> DetectionResult:
        score = context.provided_scores.get(Category.PRNU)
        if score is not None:
            return DetectionResult(scores={Category.PRNU: score}, override_hints=context.override_hints)
        if context.media_type == MediaType.VIDEO:
            computed = self._score_video(context)
        else:
            computed = self._score_image(context)
        return DetectionResult(scores={Category.PRNU: computed}, override_hints=context.override_hints)
