"""Detector modules for layered signal extraction."""

from .base import DetectionContext, DetectionResult, Detector
from .deterministic_forensics_layer import DeterministicForensicsDetector
from .frequency_layer import FrequencyLayerDetector
from .image_clip_layer import ImageClipAnomalyDetector
from .image_forensic_transformer_layer import ImageForensicTransformerDetector
from .metadata_layer import MetadataLayerDetector
from .provenance_layer import ProvenanceLayerDetector
from .prnu_layer import PrnuLayerDetector
from .spatial_layer import SpatialLayerDetector
from .tamper_localization_layer import TamperLocalizationLayerDetector
from .temporal_consistency_layer import TemporalConsistencyLayerDetector
from .video_clip_layer import VideoClipLayerDetector
from .video_forensic_transformer_layer import VideoForensicTransformerLayerDetector
from .watermark_layer import WatermarkLayerDetector
from .utils import merge_results

# Backward-compatible aliases.
ClipLayerDetector = VideoClipLayerDetector
ForensicTransformerLayerDetector = VideoForensicTransformerLayerDetector

__all__ = [
    "DetectionContext",
    "DetectionResult",
    "Detector",
    "DeterministicForensicsDetector",
    "ProvenanceLayerDetector",
    "WatermarkLayerDetector",
    "MetadataLayerDetector",
    "PrnuLayerDetector",
    "FrequencyLayerDetector",
    "SpatialLayerDetector",
    "TamperLocalizationLayerDetector",
    "TemporalConsistencyLayerDetector",
    "ImageClipAnomalyDetector",
    "ImageForensicTransformerDetector",
    "VideoClipLayerDetector",
    "VideoForensicTransformerLayerDetector",
    "ClipLayerDetector",
    "ForensicTransformerLayerDetector",
    "merge_results",
]
