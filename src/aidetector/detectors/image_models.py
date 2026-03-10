"""Compatibility module re-exporting image detector layers."""

from .image_clip_layer import ImageClipAnomalyDetector, calibrate_ai_probability
from .image_forensic_transformer_layer import (
    DEFAULT_IMAGE_FORENSIC_MODEL_ID,
    ImageForensicTransformerDetector,
    adjust_forensic_with_metadata,
)

# Backward-compatible private aliases used by existing tests.
_calibrate_ai_probability = calibrate_ai_probability
_adjust_forensic_with_metadata = adjust_forensic_with_metadata

__all__ = [
    "DEFAULT_IMAGE_FORENSIC_MODEL_ID",
    "ImageClipAnomalyDetector",
    "ImageForensicTransformerDetector",
    "calibrate_ai_probability",
    "adjust_forensic_with_metadata",
    "_calibrate_ai_probability",
    "_adjust_forensic_with_metadata",
]
