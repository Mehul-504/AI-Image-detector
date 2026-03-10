from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

from .detectors import (
    DetectionContext,
    DetectionResult,
    Detector,
    FrequencyLayerDetector,
    ImageClipAnomalyDetector,
    ImageForensicTransformerDetector,
    MetadataLayerDetector,
    ProvenanceLayerDetector,
    PrnuLayerDetector,
    SpatialLayerDetector,
    TamperLocalizationLayerDetector,
    TemporalConsistencyLayerDetector,
    VideoClipLayerDetector,
    VideoForensicTransformerLayerDetector,
    WatermarkLayerDetector,
    merge_results,
)
from .detectors.signal_utils import infer_transport_compression_signals
from .fusion import fuse_to_verdict
from .schemas import AnalysisRequest, AnalysisResponse, MediaType


@dataclass
class AuthenticityPipeline:
    detectors: Iterable[Detector] = field(
        default_factory=lambda: (
            ProvenanceLayerDetector(),
            WatermarkLayerDetector(),
            MetadataLayerDetector(),
            FrequencyLayerDetector(),
            SpatialLayerDetector(),
            TamperLocalizationLayerDetector(),
            PrnuLayerDetector(),
            TemporalConsistencyLayerDetector(),
            ImageClipAnomalyDetector(),
            ImageForensicTransformerDetector(),
            VideoClipLayerDetector(),
            VideoForensicTransformerLayerDetector(),
        )
    )

    def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        base_context = DetectionContext(
            media_type=request.media_type,
            media_uri=request.media_uri,
            provided_scores=request.provided_scores,
            override_hints=request.override_hints,
        )
        merged = DetectionResult(scores={}, override_hints=request.override_hints)
        for detector in self.detectors:
            effective_scores = dict(merged.scores)
            # User-provided category scores always take precedence.
            effective_scores.update(request.provided_scores)
            detector_context = replace(
                base_context,
                provided_scores=effective_scores,
                override_hints=merged.override_hints,
            )
            merged = merge_results(merged, detector.run(detector_context))

        aux_features: dict[str, float] = {}
        if request.media_type == MediaType.IMAGE and request.media_uri:
            compression = infer_transport_compression_signals(request.media_uri)
            aux_features = {
                "compression_score": float(compression.get("score", 0.0)),
                "compression_quant": float(compression.get("quant", 0.0)),
                "compression_block": float(compression.get("block", 0.0)),
                "compression_detail_loss": float(compression.get("detail_loss", 0.0)),
            }

        return fuse_to_verdict(
            media_type=request.media_type,
            category_scores=merged.scores,
            override_hints=merged.override_hints,
            aux_features=aux_features,
            mode_profile=request.mode_profile,
        )
