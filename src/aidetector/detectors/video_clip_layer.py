from __future__ import annotations

from typing import Any

from aidetector.schemas import Category, CategoryScore, MediaType, OverrideHints

from .base import DetectionContext, DetectionResult, Detector
from .image_clip_layer import calibrate_ai_probability
from .signal_utils import load_video_frames
from .utils import detect_device, neutral_score, transformers_source_candidates


class VideoClipLayerDetector(Detector):
    name = "video_clip_transformer_layer"

    def __init__(self, model_id: str | None = None, preferred_device: str | None = None) -> None:
        import os

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
        )
        self._ai_prompts = (
            "an AI-generated image",
            "an AI-edited image",
            "a synthetic diffusion image",
            "an artificial generative artwork",
            "a deepfake-like generated visual",
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
                    errors.append(f"{source_note}:{exc.__class__.__name__}")
            if self._processor is None or self._model is None:
                raise RuntimeError(",".join(errors) or "no_model_source")
        except Exception as exc:  # pragma: no cover - runtime dependent.
            self._load_error = f"model_load_failed:{exc}"
            self._loaded = True
            return False

        self._loaded = True
        self._load_error = None
        return True

    def _score_video(self, context: DetectionContext) -> CategoryScore:
        video, err = load_video_frames(context.media_uri, max_frames=10, max_side=288)
        if video is None:
            return neutral_score(f"video clip unavailable: {err}")
        if not self._ensure_loaded():
            return neutral_score(f"video clip unavailable: {self._load_error}")

        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._image_cls is not None
        np = __import__("numpy")

        pil_frames = [
            self._image_cls.fromarray((np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8))
            for frame in video.frames_rgb
        ]
        try:
            inputs = self._processor(
                text=self._all_prompts,
                images=pil_frames,
                return_tensors="pt",
                padding=True,
            )
            tensor_inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with self._torch.no_grad():
                logits = self._model(**tensor_inputs).logits_per_image
            probs = self._torch.softmax(logits, dim=1).detach().cpu().numpy()
        except Exception as exc:  # pragma: no cover
            return neutral_score(f"video clip inference_failed:{exc.__class__.__name__}")

        frame_risks = []
        frame_confs = []
        ai_probs = []
        for row in probs:
            real_prob = float(row[: len(self._real_prompts)].sum())
            ai_prob = float(row[len(self._real_prompts) :].sum())
            risk, conf = calibrate_ai_probability(ai_prob, real_prob)
            frame_risks.append(risk)
            frame_confs.append(conf)
            ai_probs.append(ai_prob)
        risk = float(np.median(np.array(frame_risks, dtype=np.float32)))
        base_conf = float(np.mean(np.array(frame_confs, dtype=np.float32)))
        temporal_stability = float(1.0 - np.std(np.array(ai_probs, dtype=np.float32)))
        confidence = min(90.0, max(25.0, base_conf + 12.0 * temporal_stability))
        return CategoryScore(
            risk=round(risk, 2),
            confidence=round(confidence, 2),
            rationale=(
                f"video clip [{self._source_note}]: frames={len(frame_risks)}, "
                f"risk_med={risk:.2f}, stability={temporal_stability:.3f}"
            ),
        )

    def run(self, context: DetectionContext) -> DetectionResult:
        if context.media_type != MediaType.VIDEO:
            return DetectionResult(scores={}, override_hints=OverrideHints())
        score = context.provided_scores.get(Category.CLIP_ANOMALY)
        if score is None:
            score = self._score_video(context)
        return DetectionResult(scores={Category.CLIP_ANOMALY: score}, override_hints=OverrideHints())
