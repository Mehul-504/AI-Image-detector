from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from aidetector.schemas import Category, CategoryScore, MediaType, OverrideHints

from .base import DetectionContext, DetectionResult, Detector
from .utils import detect_device, neutral_score, transformers_source_candidates


def calibrate_ai_probability(ai_prob: float, real_prob: float) -> tuple[float, float]:
    """Calibrate CLIP binary probabilities into conservative risk/confidence."""
    pair_total = ai_prob + real_prob
    if pair_total > 0:
        ai_prob = ai_prob / pair_total
        real_prob = real_prob / pair_total

    margin = abs(ai_prob - real_prob)
    compressed_risk = 1.0 / (1.0 + math.exp(-4.0 * (ai_prob - 0.5)))
    confidence = min(80.0, max(20.0, 20.0 + margin * 60.0))
    if confidence < 35.0:
        # Pull uncertain predictions toward neutral.
        compressed_risk = 0.5 + (compressed_risk - 0.5) * 0.5

    return round(compressed_risk * 100.0, 2), round(confidence, 2)


class ImageClipAnomalyDetector(Detector):
    name = "image_clip_transformer_layer"

    def __init__(
        self,
        model_id: str | None = None,
        preferred_device: str | None = None,
    ) -> None:
        self.enabled = os.getenv("AIDETECTOR_ENABLE_CLIP", "1").strip().lower() in {"1", "true", "yes", "on"}
        self.model_id = model_id or os.getenv("AIDETECTOR_CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
        self.preferred_device = preferred_device or os.getenv("AIDETECTOR_DEVICE", "auto")
        self._loaded = False
        self._load_error: str | None = None
        self._torch: Any | None = None
        self._image_cls: Any | None = None
        self._processor: Any | None = None
        self._model: Any | None = None
        self._device: str = "cpu"
        self._source_note: str = "unresolved"
        self._real_prompts = (
            "a real camera photograph",
            "an authentic natural photo",
            "a non-generated smartphone camera image",
            "a genuine documentary style photo",
            "an unedited original camera photo",
        )
        self._ai_prompts = (
            "an AI-generated image",
            "an AI-edited image",
            "a synthetic diffusion image",
            "an artificial generative artwork",
            "a deepfake-like generated visual",
            "a real photo edited with AI inpainting",
            "a camera image with generative fill artifacts",
            "a photoreal image with AI object replacement",
        )
        self._all_prompts = [*self._real_prompts, *self._ai_prompts]

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return self._load_error is None

        try:
            import torch
            from PIL import Image
            from transformers import CLIPModel, CLIPProcessor
        except ModuleNotFoundError as exc:
            self._load_error = f"missing_dependency:{exc.name}"
            self._loaded = True
            return False

        try:
            self._device = detect_device(torch, self.preferred_device)
            self._torch = torch
            self._image_cls = Image
            errors: list[str] = []
            for source, load_kwargs, source_note in transformers_source_candidates(self.model_id):
                try:
                    processor = CLIPProcessor.from_pretrained(source, **load_kwargs)
                    model = CLIPModel.from_pretrained(source, **load_kwargs)
                    model.to(self._device)
                    model.eval()
                    self._processor = processor
                    self._model = model
                    self._source_note = source_note
                    break
                except Exception as exc:  # pragma: no cover - runtime dependent.
                    detail = str(exc).splitlines()[0][:140] if str(exc) else ""
                    errors.append(f"{source_note}:{exc.__class__.__name__}:{detail}")
            if self._processor is None or self._model is None:
                raise RuntimeError(",".join(errors) or "no_model_source")
        except Exception as exc:  # pragma: no cover - depends on local model/runtime.
            self._load_error = f"model_load_failed:{exc}"
            self._loaded = True
            return False

        self._loaded = True
        self._load_error = None
        return True

    def run(self, context: DetectionContext) -> DetectionResult:
        if context.media_type != MediaType.IMAGE:
            return DetectionResult(scores={}, override_hints=OverrideHints())
        if not self.enabled:
            return DetectionResult(
                scores={Category.CLIP_ANOMALY: neutral_score("clip disabled by env:AIDETECTOR_ENABLE_CLIP")},
                override_hints=OverrideHints(),
            )

        precomputed = context.provided_scores.get(Category.CLIP_ANOMALY)
        if precomputed is not None:
            return DetectionResult(scores={Category.CLIP_ANOMALY: precomputed}, override_hints=OverrideHints())

        if not context.media_uri:
            return DetectionResult(
                scores={Category.CLIP_ANOMALY: neutral_score("clip skipped: media_uri missing")},
                override_hints=OverrideHints(),
            )

        image_path = Path(context.media_uri)
        if not image_path.exists():
            return DetectionResult(
                scores={
                    Category.CLIP_ANOMALY: neutral_score(
                        f"clip skipped: media path not found ({context.media_uri})"
                    )
                },
                override_hints=OverrideHints(),
            )

        if not self._ensure_loaded():
            return DetectionResult(
                scores={Category.CLIP_ANOMALY: neutral_score(f"clip skipped: {self._load_error}")},
                override_hints=OverrideHints(),
            )

        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._image_cls is not None

        try:
            with self._image_cls.open(image_path) as image:
                image = image.convert("RGB")
                inputs = self._processor(
                    text=self._all_prompts,
                    images=image,
                    return_tensors="pt",
                    padding=True,
                )
                tensor_inputs = {key: value.to(self._device) for key, value in inputs.items()}
                with self._torch.no_grad():
                    logits = self._model(**tensor_inputs).logits_per_image[0]
                probs = self._torch.softmax(logits, dim=0).detach().cpu().tolist()
        except Exception as exc:  # pragma: no cover - depends on local image/runtime.
            return DetectionResult(
                scores={Category.CLIP_ANOMALY: neutral_score(f"clip inference_failed:{exc.__class__.__name__}")},
                override_hints=OverrideHints(),
            )

        real_prob = float(sum(probs[: len(self._real_prompts)]))
        ai_prob = float(sum(probs[len(self._real_prompts) :]))
        risk, confidence = calibrate_ai_probability(ai_prob, real_prob)
        rationale = (
            f"clip({self.model_id}) on {self._device} [{self._source_note}]: "
            f"ai_prob={ai_prob:.3f}, real_prob={real_prob:.3f}"
        )

        return DetectionResult(
            scores={
                Category.CLIP_ANOMALY: CategoryScore(
                    risk=risk,
                    confidence=confidence,
                    rationale=rationale,
                )
            },
            override_hints=OverrideHints(),
        )
