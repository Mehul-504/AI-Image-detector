from __future__ import annotations

from pathlib import Path

from aidetector.schemas import Category, CategoryScore, MediaType

from .base import DetectionContext, DetectionResult, Detector
from .signal_utils import infer_transport_compression_signals
from .utils import neutral_score


class MetadataLayerDetector(Detector):
    name = "metadata_layer"

    _ai_keywords = (
        "stable diffusion",
        "midjourney",
        "dall-e",
        "dalle",
        "comfyui",
        "automatic1111",
        "firefly",
        "generated",
        "synthetic",
        "genai",
        "flux",
        "sdxl",
    )
    _non_ai_editor_keywords = (
        "photoshop",
        "lightroom",
        "gimp",
        "snapseed",
        "capture one",
        "pixelmator",
        "affinity photo",
    )

    def _infer_metadata_score(self, media_uri: str | None) -> CategoryScore:
        if not media_uri:
            return neutral_score("metadata heuristic skipped: media_uri missing")

        image_path = Path(media_uri)
        if not image_path.exists():
            return neutral_score(f"metadata heuristic skipped: media path not found ({media_uri})")

        try:
            from PIL import ExifTags, Image
        except ModuleNotFoundError:
            return neutral_score("metadata heuristic skipped: missing_dependency:PIL")

        try:
            with Image.open(image_path) as image:
                info = dict(getattr(image, "info", {}) or {})
                exif_raw = image.getexif() if hasattr(image, "getexif") else {}
                tag_map = ExifTags.TAGS
                exif = {str(tag_map.get(k, k)): str(v) for k, v in dict(exif_raw or {}).items()}
        except Exception as exc:  # pragma: no cover - depends on image format/runtime.
            return neutral_score(f"metadata heuristic failed:{exc.__class__.__name__}")

        make = exif.get("Make", "").strip()
        model = exif.get("Model", "").strip()
        software_text = " ".join(
            [
                exif.get("Software", ""),
                exif.get("ImageDescription", ""),
                exif.get("UserComment", ""),
                " ".join(f"{k}:{v}" for k, v in info.items()),
            ]
        ).lower()
        has_camera_meta = bool(make or model)
        has_any_meta = bool(exif or info)
        compression = infer_transport_compression_signals(media_uri)
        compression_score = float(compression.get("score", 0.0))
        whatsapp_hint = float(compression.get("whatsapp_hint", 0.0)) >= 0.5
        compression_note = (
            "transport score="
            f"{compression_score:.3f} "
            f"(quant={float(compression.get('quant', 0.0)):.3f}, "
            f"block={float(compression.get('block', 0.0)):.3f}, "
            f"detail={float(compression.get('detail_loss', 0.0)):.3f}, "
            f"small={float(compression.get('small_frame', 0.0)):.3f}, "
            f"whatsapp={int(whatsapp_hint)})"
        )

        if any(keyword in software_text for keyword in self._ai_keywords):
            return CategoryScore(
                risk=95.0,
                confidence=86.0,
                rationale="metadata heuristic: generator/edit metadata indicates AI workflow",
            )

        if whatsapp_hint and not has_camera_meta:
            return CategoryScore(
                risk=50.0,
                confidence=68.0,
                rationale=(
                    "metadata heuristic: transport-compressed messaging copy likely (WhatsApp); "
                    f"{compression_note}"
                ),
            )

        if has_camera_meta:
            if any(keyword in software_text for keyword in self._non_ai_editor_keywords):
                return CategoryScore(
                    risk=28.0,
                    confidence=72.0,
                    rationale=(
                        "metadata heuristic: camera EXIF exists, non-AI editor detected; "
                        f"{compression_note}"
                    ),
                )
            camera_desc = " ".join(part for part in [make, model] if part).strip()
            if compression_score >= 0.60:
                return CategoryScore(
                    risk=20.0,
                    confidence=74.0,
                    rationale=(
                        f"metadata heuristic: camera EXIF present ({camera_desc}) but strong transport "
                        f"recompression likely; {compression_note}"
                    ),
                )
            return CategoryScore(
                risk=12.0,
                confidence=85.0,
                rationale=f"metadata heuristic: camera EXIF present ({camera_desc}); {compression_note}",
            )

        if has_any_meta:
            return CategoryScore(
                risk=43.0,
                confidence=45.0,
                rationale=f"metadata heuristic: metadata present but weak provenance cues; {compression_note}",
            )
        if compression_score >= 0.62:
            return CategoryScore(
                risk=50.0,
                confidence=64.0,
                rationale=f"metadata heuristic: metadata stripped and heavy recompression likely; {compression_note}",
            )
        return neutral_score("metadata heuristic: no metadata found")

    def run(self, context: DetectionContext) -> DetectionResult:
        provided = context.provided_scores.get(Category.METADATA)
        if provided is not None:
            return DetectionResult(scores={Category.METADATA: provided}, override_hints=context.override_hints)
        if context.media_type == MediaType.IMAGE:
            return DetectionResult(
                scores={Category.METADATA: self._infer_metadata_score(context.media_uri)},
                override_hints=context.override_hints,
            )
        return DetectionResult(scores={}, override_hints=context.override_hints)
