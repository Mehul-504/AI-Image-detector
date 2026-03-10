from __future__ import annotations

from aidetector.schemas import Category, MediaType

from .base import DetectionContext, DetectionResult, Detector
from .frequency_layer import FrequencyLayerDetector
from .metadata_layer import MetadataLayerDetector
from .prnu_layer import PrnuLayerDetector
from .provenance_layer import ProvenanceLayerDetector
from .spatial_layer import SpatialLayerDetector
from .tamper_localization_layer import TamperLocalizationLayerDetector
from .temporal_consistency_layer import TemporalConsistencyLayerDetector
from .utils import merge_results
from .watermark_layer import WatermarkLayerDetector


class DeterministicForensicsDetector(Detector):
    name = "deterministic_forensics_layer"

    _image_categories = (
        Category.PROVENANCE,
        Category.WATERMARK,
        Category.METADATA,
        Category.FREQUENCY,
        Category.SPATIAL,
        Category.TAMPER_LOCALIZATION,
        Category.PRNU,
    )
    _video_categories = (
        Category.PROVENANCE,
        Category.WATERMARK,
        Category.METADATA,
        Category.FREQUENCY,
        Category.SPATIAL,
        Category.TAMPER_LOCALIZATION,
        Category.PRNU,
        Category.TEMPORAL_CONSISTENCY,
    )
    _branches = (
        ProvenanceLayerDetector(),
        WatermarkLayerDetector(),
        MetadataLayerDetector(),
        FrequencyLayerDetector(),
        SpatialLayerDetector(),
        TamperLocalizationLayerDetector(),
        PrnuLayerDetector(),
        TemporalConsistencyLayerDetector(),
    )

    def run(self, context: DetectionContext) -> DetectionResult:
        allowed = self._video_categories if context.media_type == MediaType.VIDEO else self._image_categories
        merged = DetectionResult(scores={}, override_hints=context.override_hints)
        for branch in self._branches:
            branch_result = branch.run(context)
            filtered = {
                category: score
                for category, score in branch_result.scores.items()
                if category in allowed
            }
            merged = merge_results(
                merged,
                DetectionResult(scores=filtered, override_hints=branch_result.override_hints),
            )
        return merged
