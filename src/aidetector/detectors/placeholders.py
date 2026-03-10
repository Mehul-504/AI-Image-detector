"""Compatibility module re-exporting legacy detector aliases."""

from .deterministic_forensics_layer import DeterministicForensicsDetector
from .video_clip_layer import VideoClipLayerDetector
from .video_forensic_transformer_layer import VideoForensicTransformerLayerDetector

# Backward-compatible names used by existing pipeline/imports.
ClipLayerDetector = VideoClipLayerDetector
ForensicTransformerLayerDetector = VideoForensicTransformerLayerDetector

__all__ = [
    "DeterministicForensicsDetector",
    "VideoClipLayerDetector",
    "VideoForensicTransformerLayerDetector",
    "ClipLayerDetector",
    "ForensicTransformerLayerDetector",
]
