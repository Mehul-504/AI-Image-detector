from __future__ import annotations

import json
import math
import os
import subprocess
from typing import Any

from aidetector.schemas import Category, CategoryScore, MediaType, OverrideHints

from .base import DetectionContext, DetectionResult, Detector
from .image_forensic_transformer_layer import DEFAULT_IMAGE_FORENSIC_MODEL_ID
from .signal_utils import clamp01, load_video_frames
from .utils import detect_device, neutral_score, transformers_source_candidates


class VideoForensicTransformerLayerDetector(Detector):
    name = "video_forensic_transformer_layer"

    _fake_keywords = ("fake", "synthetic", "generated", "manipulated", "deepfake", "ai")
    _real_keywords = ("real", "authentic", "pristine", "camera")

    def __init__(self, model_id_or_path: str | None = None, preferred_device: str | None = None) -> None:
        self.model_id_or_path = (
            model_id_or_path
            or os.getenv("AIDETECTOR_VIDEO_FORENSIC_MODEL")
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
        except Exception as exc:  # pragma: no cover
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
            return [1] if num_labels == 2 else []
        normalized = {int(index): str(label).lower() for index, label in id2label.items()}
        fake_hits = [
            idx
            for idx, label in normalized.items()
            if any(keyword in label for keyword in self._fake_keywords)
        ]
        if fake_hits:
            return sorted(set(fake_hits))
        if len(normalized) == 2:
            real_like = {
                idx
                for idx, label in normalized.items()
                if any(keyword in label for keyword in self._real_keywords)
            }
            if len(real_like) == 1:
                return [idx for idx in normalized if idx not in real_like]
        return []

    def _probe_has_audio(self, media_uri: str | None) -> bool | None:
        if not media_uri:
            return None
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-print_format",
                    "json",
                    "-show_streams",
                    media_uri,
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        try:
            payload = json.loads(result.stdout or "{}")
            streams = payload.get("streams", [])
        except Exception:
            return None
        for stream in streams:
            if str(stream.get("codec_type", "")).lower() == "audio":
                return True
        return False

    def _score_video(self, context: DetectionContext) -> tuple[CategoryScore, CategoryScore]:
        video, err = load_video_frames(context.media_uri, max_frames=14, max_side=288)
        if video is None:
            unavailable = neutral_score(f"video forensic unavailable: {err}")
            return unavailable, unavailable
        if not self._ensure_loaded():
            unavailable = neutral_score(f"video forensic unavailable: {self._load_error}")
            return unavailable, unavailable

        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._image_cls is not None
        np = __import__("numpy")

        fake_indices = self._resolve_fake_indices()
        if not fake_indices:
            unavailable = neutral_score("video forensic unavailable: unresolved fake label")
            return unavailable, unavailable

        pil_frames = [
            self._image_cls.fromarray((np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8))
            for frame in video.frames_rgb
        ]
        try:
            inputs = self._processor(images=pil_frames, return_tensors="pt")
            tensor_inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with self._torch.no_grad():
                logits = self._model(**tensor_inputs).logits
            probs = self._torch.softmax(logits, dim=1).detach().cpu().numpy()
        except Exception as exc:  # pragma: no cover
            unavailable = neutral_score(f"video forensic inference_failed:{exc.__class__.__name__}")
            return unavailable, unavailable

        fake_probs = []
        margins = []
        for row in probs:
            fake_prob = float(sum(row[idx] for idx in fake_indices if 0 <= idx < len(row)))
            non_fake = [row[i] for i in range(len(row)) if i not in fake_indices]
            max_non_fake = float(max(non_fake) if non_fake else max(0.0, 1.0 - fake_prob))
            fake_probs.append(fake_prob)
            margins.append(abs(fake_prob - max_non_fake))

        fake_probs_np = np.array(fake_probs, dtype=np.float32)
        margins_np = np.array(margins, dtype=np.float32)
        fake_median = float(np.median(fake_probs_np))
        risk = (1.0 / (1.0 + math.exp(-4.0 * (fake_median - 0.5)))) * 100.0
        confidence = min(92.0, max(30.0, 32.0 + float(margins_np.mean()) * 62.0))
        forensic_score = CategoryScore(
            risk=round(risk, 2),
            confidence=round(confidence, 2),
            rationale=(
                f"video forensic [{self._source_note}]: frames={len(fake_probs)}, "
                f"fake_med={fake_median:.3f}, margin_mean={float(margins_np.mean()):.3f}"
            ),
        )

        # Audio-visual consistency proxy: combine motion profile with audio-stream presence.
        grays = [
            (0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]).astype(np.float32)
            for frame in video.frames_rgb
        ]
        if len(grays) >= 2:
            diffs = np.array(
                [float(np.mean(np.abs(grays[i] - grays[i - 1]))) for i in range(1, len(grays))],
                dtype=np.float32,
            )
            motion = float(diffs.mean())
            motion_var = float(diffs.std() / (motion + 1e-6))
        else:
            motion = 0.0
            motion_var = 0.0
        has_audio = self._probe_has_audio(context.media_uri)
        motion_score = clamp01((motion - 0.015) / 0.12)
        inconsistency = clamp01((motion_var - 0.35) / 1.0)
        if has_audio is True:
            av_risk = 28.0 + 45.0 * inconsistency
            av_conf = 48.0 + 22.0 * motion_score
            audio_note = "audio_present"
        elif has_audio is False:
            av_risk = 40.0 + 35.0 * motion_score + 15.0 * inconsistency
            av_conf = 34.0 + 18.0 * motion_score
            audio_note = "audio_absent"
        else:
            av_risk = 35.0 + 30.0 * inconsistency
            av_conf = 28.0 + 16.0 * motion_score
            audio_note = "audio_unknown"
        av_score = CategoryScore(
            risk=round(max(0.0, min(100.0, av_risk)), 2),
            confidence=round(max(20.0, min(80.0, av_conf)), 2),
            rationale=(
                f"audio-visual proxy: {audio_note}, motion={motion:.4f}, "
                f"motion_var={motion_var:.3f}"
            ),
        )
        return forensic_score, av_score

    def run(self, context: DetectionContext) -> DetectionResult:
        if context.media_type != MediaType.VIDEO:
            return DetectionResult(scores={}, override_hints=OverrideHints())
        forensic = context.provided_scores.get(Category.FORENSIC_TRANSFORMER)
        av = context.provided_scores.get(Category.AUDIO_VISUAL_TRANSFORMER)
        if forensic is None or av is None:
            computed_forensic, computed_av = self._score_video(context)
            if forensic is None:
                forensic = computed_forensic
            if av is None:
                av = computed_av
        scores = {
            Category.FORENSIC_TRANSFORMER: forensic,
            Category.AUDIO_VISUAL_TRANSFORMER: av,
        }
        return DetectionResult(scores=scores, override_hints=OverrideHints())
