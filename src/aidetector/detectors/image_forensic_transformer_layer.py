from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from aidetector.schemas import Category, CategoryScore, MediaType, OverrideHints

from .base import DetectionContext, DetectionResult, Detector
from .signal_utils import clamp01, infer_transport_compression_signals
from .utils import detect_device, neutral_score, transformers_source_candidates

DEFAULT_IMAGE_FORENSIC_MODEL_ID = "dima806/ai_vs_real_image_detection"


def _is_transport_compressed_metadata(metadata_score: CategoryScore | None) -> bool:
    if metadata_score is None or not metadata_score.rationale:
        return False
    rationale = metadata_score.rationale.lower()
    return (
        "transport-compressed" in rationale
        or "recompression likely" in rationale
        or "whatsapp" in rationale
    )


def _build_multicrop_views(image: Any) -> list[Any]:
    views = [image]
    width, height = image.size
    min_side = min(width, height)
    if min_side < 320:
        return views
    crop_w = int(round(width * 0.72))
    crop_h = int(round(height * 0.72))
    if crop_w < 224 or crop_h < 224:
        return views
    x_max = max(0, width - crop_w)
    y_max = max(0, height - crop_h)
    points = [
        (0, 0),
        (x_max, 0),
        (0, y_max),
        (x_max, y_max),
        (x_max // 2, y_max // 2),
    ]
    seen: set[tuple[int, int]] = set()
    for x, y in points:
        key = (int(x), int(y))
        if key in seen:
            continue
        seen.add(key)
        views.append(image.crop((key[0], key[1], key[0] + crop_w, key[1] + crop_h)))
    return views


def adjust_forensic_with_metadata(
    risk: float,
    confidence: float,
    metadata_score: CategoryScore | None,
) -> tuple[float, float, str | None]:
    """Apply cross-signal correction to forensic outputs using metadata evidence."""
    if metadata_score is None:
        return risk, confidence, None

    # Strong camera metadata should temper false-positive forensic spikes.
    if metadata_score.confidence >= 70.0 and metadata_score.risk <= 20.0:
        adjusted_risk = max(5.0, risk * 0.52)
        adjusted_conf = max(22.0, confidence * 0.72)
        return round(adjusted_risk, 2), round(adjusted_conf, 2), "camera_metadata_support"

    # Moderately authentic metadata still reduces forensic aggressiveness.
    if metadata_score.confidence >= 60.0 and metadata_score.risk <= 35.0:
        adjusted_risk = max(8.0, risk * 0.68)
        adjusted_conf = max(24.0, confidence * 0.80)
        return round(adjusted_risk, 2), round(adjusted_conf, 2), "auth_metadata_support"

    # Strong AI-generation metadata can strengthen forensic signal.
    if metadata_score.confidence >= 70.0 and metadata_score.risk >= 85.0:
        adjusted_risk = min(99.0, risk * 0.86 + 14.0)
        adjusted_conf = min(90.0, confidence * 1.05 + 2.0)
        return round(adjusted_risk, 2), round(adjusted_conf, 2), "ai_metadata_support"

    return risk, confidence, None


def adjust_forensic_with_consensus(
    risk: float,
    confidence: float,
    *,
    clip_score: CategoryScore | None,
    frequency_score: CategoryScore | None,
    spatial_score: CategoryScore | None,
    tamper_score: CategoryScore | None,
    watermark_score: CategoryScore | None,
    metadata_score: CategoryScore | None,
) -> tuple[float, float, str | None]:
    def is_high(score: CategoryScore | None) -> bool:
        return score is not None and score.confidence >= 45.0 and score.risk >= 65.0

    def is_low(score: CategoryScore | None) -> bool:
        return score is not None and score.confidence >= 40.0 and score.risk <= 45.0

    deterministic_scores = [frequency_score, spatial_score, tamper_score]
    deterministic_low = sum(1 for score in deterministic_scores if is_low(score))
    deterministic_high = sum(1 for score in deterministic_scores if is_high(score))
    high_support = deterministic_high + sum(
        1 for score in (clip_score, watermark_score, metadata_score) if is_high(score)
    )
    low_support = deterministic_low + sum(
        1 for score in (clip_score, watermark_score, metadata_score) if is_low(score)
    )

    clip_strong_ai = (
        clip_score is not None and clip_score.confidence >= 55.0 and clip_score.risk >= 65.0
    )
    clip_non_ai = (
        clip_score is not None and clip_score.confidence >= 28.0 and clip_score.risk <= 50.0
    )
    metadata_auth = (
        metadata_score is not None and metadata_score.confidence >= 40.0 and metadata_score.risk <= 45.0
    )
    metadata_ai = (
        metadata_score is not None and metadata_score.confidence >= 70.0 and metadata_score.risk >= 80.0
    )

    if risk >= 78.0 and confidence >= 55.0:
        if deterministic_low >= 3 and not clip_strong_ai and not metadata_ai:
            if clip_non_ai or metadata_auth:
                return (
                    round(max(6.0, risk * 0.45), 2),
                    round(max(18.0, confidence * 0.50), 2),
                    "cross_signal_conflict_strong",
                )
            return (
                round(max(8.0, risk * 0.58), 2),
                round(max(20.0, confidence * 0.62), 2),
                "cross_signal_conflict",
            )

        if high_support <= 1 and low_support >= 3 and (clip_non_ai or metadata_auth):
            return (
                round(max(10.0, risk * 0.68), 2),
                round(max(24.0, confidence * 0.72), 2),
                "cross_signal_conflict",
            )

        if high_support >= 3 or clip_strong_ai or metadata_ai:
            return (
                round(min(99.0, risk * 0.97 + 2.0), 2),
                round(min(94.0, confidence * 1.04 + 1.0), 2),
                "cross_signal_support",
            )

    return risk, confidence, None


def adjust_forensic_with_transport_compression(
    risk: float,
    confidence: float,
    *,
    compression_score: float,
    clip_score: CategoryScore | None,
    metadata_score: CategoryScore | None,
) -> tuple[float, float, str | None]:
    if compression_score < 0.58:
        return risk, confidence, None

    severity = clamp01((compression_score - 0.58) / 0.42)
    clip_strong_ai = clip_score is not None and clip_score.confidence >= 55.0 and clip_score.risk >= 65.0
    metadata_ai = metadata_score is not None and metadata_score.confidence >= 70.0 and metadata_score.risk >= 80.0
    transport_metadata = _is_transport_compressed_metadata(metadata_score)

    if clip_strong_ai or metadata_ai:
        adjusted_risk = 50.0 + (risk - 50.0) * (1.0 - 0.20 * severity)
        adjusted_conf = max(24.0, confidence * (1.0 - 0.18 * severity))
        return round(adjusted_risk, 2), round(adjusted_conf, 2), "transport_compression_soft"

    adjusted_risk = 50.0 + (risk - 50.0) * (1.0 - 0.52 * severity)
    adjusted_conf = max(18.0, confidence * (1.0 - (0.45 + (0.10 if transport_metadata else 0.0)) * severity))
    return round(adjusted_risk, 2), round(adjusted_conf, 2), "transport_compression"


class ImageForensicTransformerDetector(Detector):
    name = "image_forensic_transformer_layer"

    _fake_keywords = ("fake", "synthetic", "generated", "manipulated", "deepfake", "ai")
    _real_keywords = ("real", "authentic", "pristine", "camera")

    def __init__(
        self,
        model_id_or_path: str | None = None,
        preferred_device: str | None = None,
    ) -> None:
        self.enabled = os.getenv("AIDETECTOR_ENABLE_FORENSIC_TRANSFORMER", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.model_id_or_path = (
            model_id_or_path
            or os.getenv("AIDETECTOR_IMAGE_FORENSIC_MODEL")
            or DEFAULT_IMAGE_FORENSIC_MODEL_ID
        )
        self.preferred_device = preferred_device or os.getenv("AIDETECTOR_DEVICE", "auto")
        self._loaded = False
        self._load_error: str | None = None
        self._torch: Any | None = None
        self._image_cls: Any | None = None
        self._processor: Any | None = None
        self._model: Any | None = None
        self._device: str = "cpu"
        self._source_note: str = "unresolved"

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return self._load_error is None

        try:
            import torch
            from PIL import Image
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ModuleNotFoundError as exc:
            self._load_error = f"missing_dependency:{exc.name}"
            self._loaded = True
            return False

        try:
            self._device = detect_device(torch, self.preferred_device)
            self._torch = torch
            self._image_cls = Image
            errors: list[str] = []
            for source, load_kwargs, source_note in transformers_source_candidates(self.model_id_or_path):
                try:
                    processor = AutoImageProcessor.from_pretrained(source, **load_kwargs)
                    model = AutoModelForImageClassification.from_pretrained(source, **load_kwargs)
                    model.to(self._device)
                    model.eval()
                    self._processor = processor
                    self._model = model
                    self._source_note = source_note
                    break
                except Exception as exc:  # pragma: no cover - runtime dependent.
                    errors.append(f"{source_note}:{exc.__class__.__name__}")
            if self._processor is None or self._model is None:
                raise RuntimeError(",".join(errors) or "no_model_source")
        except Exception as exc:  # pragma: no cover - depends on local model/runtime.
            self._load_error = f"model_load_failed:{exc}"
            self._loaded = True
            return False

        self._loaded = True
        self._load_error = None
        return True

    def _resolve_fake_indices(self) -> list[int]:
        assert self._model is not None
        id2label = getattr(self._model.config, "id2label", {}) or {}
        if not id2label:
            num_labels = int(getattr(self._model.config, "num_labels", 0) or 0)
            if num_labels == 2:
                return [1]
            return []

        normalized = {int(index): str(label).lower() for index, label in id2label.items()}
        fake_hits = [
            index
            for index, label in normalized.items()
            if any(keyword in label for keyword in self._fake_keywords)
        ]
        if fake_hits:
            return sorted(set(fake_hits))

        if len(normalized) == 2:
            real_like = {
                index
                for index, label in normalized.items()
                if any(keyword in label for keyword in self._real_keywords)
            }
            if len(real_like) == 1:
                return [index for index in normalized if index not in real_like]
        return []

    def run(self, context: DetectionContext) -> DetectionResult:
        if context.media_type != MediaType.IMAGE:
            return DetectionResult(scores={}, override_hints=OverrideHints())
        if not self.enabled:
            return DetectionResult(
                scores={
                    Category.FORENSIC_TRANSFORMER: neutral_score(
                        "forensic transformer disabled by env:AIDETECTOR_ENABLE_FORENSIC_TRANSFORMER"
                    )
                },
                override_hints=OverrideHints(),
            )

        precomputed = context.provided_scores.get(Category.FORENSIC_TRANSFORMER)
        if precomputed is not None:
            return DetectionResult(
                scores={Category.FORENSIC_TRANSFORMER: precomputed},
                override_hints=OverrideHints(),
            )

        if not context.media_uri:
            return DetectionResult(
                scores={
                    Category.FORENSIC_TRANSFORMER: neutral_score(
                        "forensic transformer skipped: media_uri missing"
                    )
                },
                override_hints=OverrideHints(),
            )

        image_path = Path(context.media_uri)
        if not image_path.exists():
            return DetectionResult(
                scores={
                    Category.FORENSIC_TRANSFORMER: neutral_score(
                        f"forensic transformer skipped: media path not found ({context.media_uri})"
                    )
                },
                override_hints=OverrideHints(),
            )

        if not self._ensure_loaded():
            return DetectionResult(
                scores={
                    Category.FORENSIC_TRANSFORMER: neutral_score(
                        f"forensic transformer skipped: {self._load_error}"
                    )
                },
                override_hints=OverrideHints(),
            )

        assert self._torch is not None
        assert self._image_cls is not None
        assert self._model is not None
        assert self._processor is not None

        fake_indices = self._resolve_fake_indices()
        if not fake_indices:
            return DetectionResult(
                scores={
                    Category.FORENSIC_TRANSFORMER: neutral_score(
                        "forensic transformer skipped: could not resolve fake label index"
                    )
                },
                override_hints=OverrideHints(),
            )

        try:
            with self._image_cls.open(image_path) as image:
                image = image.convert("RGB")
                views = _build_multicrop_views(image)
                inputs = self._processor(images=views, return_tensors="pt")
                tensor_inputs = {key: value.to(self._device) for key, value in inputs.items()}
                with self._torch.no_grad():
                    logits = self._model(**tensor_inputs).logits
                probs_batch = self._torch.softmax(logits, dim=1).detach().cpu().tolist()
        except Exception as exc:  # pragma: no cover - depends on local image/runtime.
            return DetectionResult(
                scores={
                    Category.FORENSIC_TRANSFORMER: neutral_score(
                        f"forensic transformer inference_failed:{exc.__class__.__name__}"
                    )
                },
                override_hints=OverrideHints(),
            )

        if not probs_batch:
            return DetectionResult(
                scores={
                    Category.FORENSIC_TRANSFORMER: neutral_score(
                        "forensic transformer skipped: empty model output"
                    )
                },
                override_hints=OverrideHints(),
            )
        valid_fake_indices = [index for index in fake_indices if 0 <= index < len(probs_batch[0])]
        if not valid_fake_indices:
            return DetectionResult(
                scores={
                    Category.FORENSIC_TRANSFORMER: neutral_score(
                        "forensic transformer skipped: fake label index out of range"
                    )
                },
                override_hints=OverrideHints(),
            )

        fake_probs = [float(sum(prob[index] for index in valid_fake_indices)) for prob in probs_batch]
        fake_prob = float(sum(fake_probs) / max(1, len(fake_probs)))
        if len(fake_probs) >= 3:
            sorted_probs = sorted(fake_probs, reverse=True)
            top_k = max(1, len(sorted_probs) // 3)
            fake_prob = 0.7 * fake_prob + 0.3 * float(sum(sorted_probs[:top_k]) / top_k)
        non_fake_indices = [index for index in range(len(probs_batch[0])) if index not in valid_fake_indices]
        max_non_fake = max(
            (float(sum(prob[index] for prob in probs_batch) / len(probs_batch)) for index in non_fake_indices),
            default=max(0.0, 1.0 - fake_prob),
        )
        margin = abs(fake_prob - max_non_fake)
        # Conservative calibration to reduce overconfident false positives.
        risk = round((1.0 / (1.0 + math.exp(-4.0 * (fake_prob - 0.5)))) * 100.0, 2)
        confidence = round(min(88.0, max(30.0, 30.0 + margin * 58.0)), 2)
        if len(fake_probs) > 1:
            spread = float(max(fake_probs) - min(fake_probs))
            confidence = round(max(24.0, confidence * (1.0 - 0.30 * clamp01((spread - 0.18) / 0.52))), 2)

        metadata_score = context.provided_scores.get(Category.METADATA)
        risk, confidence, metadata_note = adjust_forensic_with_metadata(
            risk=risk,
            confidence=confidence,
            metadata_score=metadata_score,
        )
        risk, confidence, consensus_note = adjust_forensic_with_consensus(
            risk=risk,
            confidence=confidence,
            clip_score=context.provided_scores.get(Category.CLIP_ANOMALY),
            frequency_score=context.provided_scores.get(Category.FREQUENCY),
            spatial_score=context.provided_scores.get(Category.SPATIAL),
            tamper_score=context.provided_scores.get(Category.TAMPER_LOCALIZATION),
            watermark_score=context.provided_scores.get(Category.WATERMARK),
            metadata_score=metadata_score,
        )
        compression = infer_transport_compression_signals(context.media_uri)
        compression_score = float(compression.get("score", 0.0))
        risk, confidence, compression_note = adjust_forensic_with_transport_compression(
            risk=risk,
            confidence=confidence,
            compression_score=compression_score,
            clip_score=context.provided_scores.get(Category.CLIP_ANOMALY),
            metadata_score=metadata_score,
        )
        rationale = (
            "forensic_transformer("
            f"{self.model_id_or_path}) on {self._device} [{self._source_note}]: "
            f"fake_prob={fake_prob:.3f}, views={len(fake_probs)}, "
            f"fake_range=({min(fake_probs):.3f},{max(fake_probs):.3f}), "
            f"fake_idx={valid_fake_indices}"
        )
        if metadata_note is not None:
            rationale = f"{rationale}; calibration={metadata_note}"
        if consensus_note is not None:
            rationale = f"{rationale}; consensus={consensus_note}"
        if compression_note is not None:
            rationale = (
                f"{rationale}; compression={compression_note}:"
                f"{compression_score:.3f}"
            )

        return DetectionResult(
            scores={
                Category.FORENSIC_TRANSFORMER: CategoryScore(
                    risk=risk,
                    confidence=confidence,
                    rationale=rationale,
                )
            },
            override_hints=OverrideHints(),
        )
