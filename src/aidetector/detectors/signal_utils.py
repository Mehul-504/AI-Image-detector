from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageSignalBundle:
    rgb: Any  # numpy.ndarray float32 [H, W, 3] in [0, 1]
    gray: Any  # numpy.ndarray float32 [H, W] in [0, 1]
    path: Path


@dataclass(frozen=True)
class VideoFrameBundle:
    frames_rgb: list[Any]  # list[numpy.ndarray float32 [H, W, 3] in [0, 1]]
    fps: float
    duration: float


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def safe_import_numpy() -> tuple[Any | None, str | None]:
    try:
        import numpy as np
    except ModuleNotFoundError:
        return None, "missing_dependency:numpy"
    return np, None


def load_image_signals(media_uri: str | None, max_side: int = 1024) -> tuple[ImageSignalBundle | None, str | None]:
    if not media_uri:
        return None, "media_uri missing"
    path = Path(media_uri)
    if not path.exists():
        return None, f"media path not found ({media_uri})"

    np, err = safe_import_numpy()
    if np is None:
        return None, err

    try:
        from PIL import Image
    except ModuleNotFoundError:
        return None, "missing_dependency:PIL"

    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            if max(width, height) > max_side:
                scale = max_side / float(max(width, height))
                resized = (
                    max(16, int(round(width * scale))),
                    max(16, int(round(height * scale))),
                )
                image = image.resize(resized)
            rgb = np.asarray(image, dtype=np.float32) / 255.0
    except Exception as exc:  # pragma: no cover - runtime dependent.
        return None, f"image_load_failed:{exc.__class__.__name__}"

    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    return ImageSignalBundle(rgb=rgb, gray=gray, path=path), None


def gaussian_blur(gray: Any) -> Any:
    np, _ = safe_import_numpy()
    if np is None:
        return gray
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    kernel = kernel / float(kernel.sum())
    # separable blur
    padded = np.pad(gray, ((0, 0), (2, 2)), mode="reflect")
    row_blur = (
        kernel[0] * padded[:, 0:-4]
        + kernel[1] * padded[:, 1:-3]
        + kernel[2] * padded[:, 2:-2]
        + kernel[3] * padded[:, 3:-1]
        + kernel[4] * padded[:, 4:]
    )
    padded2 = np.pad(row_blur, ((2, 2), (0, 0)), mode="reflect")
    col_blur = (
        kernel[0] * padded2[0:-4, :]
        + kernel[1] * padded2[1:-3, :]
        + kernel[2] * padded2[2:-2, :]
        + kernel[3] * padded2[3:-1, :]
        + kernel[4] * padded2[4:, :]
    )
    return col_blur.astype(np.float32)


def sobel_magnitude(gray: Any) -> Any:
    np, _ = safe_import_numpy()
    if np is None:
        return gray
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="reflect")
    gx = (
        padded[0:-2, 2:]
        + 2.0 * padded[1:-1, 2:]
        + padded[2:, 2:]
        - padded[0:-2, 0:-2]
        - 2.0 * padded[1:-1, 0:-2]
        - padded[2:, 0:-2]
    )
    gy = (
        padded[2:, 0:-2]
        + 2.0 * padded[2:, 1:-1]
        + padded[2:, 2:]
        - padded[0:-2, 0:-2]
        - 2.0 * padded[0:-2, 1:-1]
        - padded[0:-2, 2:]
    )
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def patchwise_mean(values: Any, patch: int = 32) -> Any:
    np, _ = safe_import_numpy()
    if np is None:
        return values
    h, w = values.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((1, 1), dtype=np.float32)
    if h < patch or w < patch:
        return np.array([[float(np.mean(values))]], dtype=np.float32)

    ph = h // patch
    pw = w // patch
    small_h = ph * patch
    small_w = pw * patch
    if small_h <= 0 or small_w <= 0:
        return np.array([[float(np.mean(values))]], dtype=np.float32)
    cropped = values[:small_h, :small_w]
    reshaped = cropped.reshape(ph, patch, pw, patch)
    return reshaped.mean(axis=(1, 3))


def radial_spectrum_features(gray: Any) -> dict[str, float]:
    np, _ = safe_import_numpy()
    if np is None:
        return {"hf_ratio": 0.5, "flatness": 0.5, "peak_ratio": 1.0}

    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(f) + 1e-8
    mag = np.log1p(mag)
    h, w = gray.shape
    yy, xx = np.indices((h, w))
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rr = rr / float(rr.max() + 1e-8)

    total = float(mag.sum() + 1e-8)
    hf_ratio = float(mag[rr >= 0.35].sum() / total)

    flatness = float(math.exp(float(np.mean(np.log(mag + 1e-8)))) / float(np.mean(mag + 1e-8)))
    p99 = float(np.percentile(mag, 99.5))
    p90 = float(np.percentile(mag, 90.0)) + 1e-8
    peak_ratio = p99 / p90
    return {"hf_ratio": hf_ratio, "flatness": flatness, "peak_ratio": peak_ratio}


def block_boundary_score(gray: Any, block: int = 8) -> float:
    np, _ = safe_import_numpy()
    if np is None:
        return 0.5
    h, w = gray.shape
    if h < block * 2 or w < block * 2:
        return 0.5
    diff_x = np.abs(np.diff(gray, axis=1))
    diff_y = np.abs(np.diff(gray, axis=0))

    x_idx = list(range(block - 1, diff_x.shape[1], block))
    y_idx = list(range(block - 1, diff_y.shape[0], block))
    if not x_idx or not y_idx:
        return 0.5

    boundary_x = float(diff_x[:, x_idx].mean())
    boundary_y = float(diff_y[y_idx, :].mean())
    interior_x = float(diff_x.mean()) + 1e-8
    interior_y = float(diff_y.mean()) + 1e-8
    ratio = 0.5 * ((boundary_x / interior_x) + (boundary_y / interior_y))
    return float(ratio)


def image_ela(gray_rgb: Any, jpeg_quality: int = 90) -> tuple[Any | None, str | None]:
    np, err = safe_import_numpy()
    if np is None:
        return None, err
    try:
        from PIL import Image
    except ModuleNotFoundError:
        return None, "missing_dependency:PIL"

    try:
        image = Image.fromarray((np.clip(gray_rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality)
        buffer.seek(0)
        recompressed = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float32) / 255.0
        diff = np.mean(np.abs(gray_rgb - recompressed), axis=2).astype(np.float32)
        return diff, None
    except Exception as exc:  # pragma: no cover - runtime dependent.
        return None, f"ela_failed:{exc.__class__.__name__}"


def infer_transport_compression_signals(
    media_uri: str | None,
    *,
    gray: Any | None = None,
) -> dict[str, float | str]:
    """Estimate whether an input likely passed through strong transport recompression.

    Returns normalized [0, 1] sub-signals and aggregate `score`.
    """
    base = {
        "score": 0.0,
        "quant": 0.0,
        "block": 0.0,
        "detail_loss": 0.0,
        "small_frame": 0.0,
        "whatsapp_hint": 0.0,
        "format": "",
    }
    if not media_uri:
        return base
    path = Path(media_uri)
    if not path.exists():
        return base

    np, err = safe_import_numpy()
    if np is None:
        return base | {"format": f"error:{err}"}

    try:
        from PIL import Image
    except ModuleNotFoundError:
        return base | {"format": "error:missing_dependency:PIL"}

    fmt = ""
    quant_strength = 0.0
    whatsapp_hint = 1.0 if "whatsapp" in path.name.lower() else 0.0
    min_side = 0
    try:
        with Image.open(path) as image:
            fmt = str(image.format or "").lower()
            width, height = image.size
            min_side = int(min(width, height))
            info_text = " ".join(f"{k}:{v}" for k, v in dict(getattr(image, "info", {}) or {}).items()).lower()
            if "whatsapp" in info_text:
                whatsapp_hint = 1.0

            quant_tables = getattr(image, "quantization", None) or {}
            if quant_tables:
                values: list[float] = []
                for table in quant_tables.values():
                    values.extend(float(v) for v in table)
                if values:
                    arr = np.asarray(values, dtype=np.float32)
                    q_mean = float(arr.mean())
                    q_p90 = float(np.percentile(arr, 90.0))
                    quant_strength = clamp01((q_mean - 16.0) / 45.0) * 0.7 + clamp01(
                        (q_p90 - 55.0) / 110.0
                    ) * 0.3
            elif fmt in {"jpeg", "jpg", "jfif"}:
                # JPEG-like stream without exposed table metadata still implies lossy path.
                quant_strength = 0.22
    except Exception:
        return base | {"format": "error:image_open_failed"}

    if gray is None:
        image_bundle, load_err = load_image_signals(media_uri, max_side=1024)
        if image_bundle is not None:
            gray = image_bundle.gray
        elif load_err is not None:
            fmt = f"{fmt}|{load_err}"

    block_strength = 0.0
    detail_loss = 0.0
    if gray is not None:
        block = block_boundary_score(gray)
        block_strength = clamp01((block - 1.0) / 0.65)
        edge = sobel_magnitude(gray)
        edge_mean = float(edge.mean())
        detail_loss = clamp01((0.085 - edge_mean) / 0.07)
        min_side = int(min(min_side or gray.shape[0], gray.shape[0], gray.shape[1]))

    small_frame = clamp01((780.0 - float(max(1, min_side))) / 780.0)
    score = clamp01(
        0.42 * quant_strength
        + 0.28 * block_strength
        + 0.16 * detail_loss
        + 0.08 * small_frame
        + 0.20 * whatsapp_hint
    )
    return {
        "score": round(score, 4),
        "quant": round(quant_strength, 4),
        "block": round(block_strength, 4),
        "detail_loss": round(detail_loss, 4),
        "small_frame": round(small_frame, 4),
        "whatsapp_hint": round(whatsapp_hint, 4),
        "format": fmt,
    }


def load_video_frames(
    media_uri: str | None,
    *,
    max_frames: int = 16,
    max_side: int = 320,
) -> tuple[VideoFrameBundle | None, str | None]:
    if not media_uri:
        return None, "media_uri missing"
    path = Path(media_uri)
    if not path.exists():
        return None, f"media path not found ({media_uri})"

    np, err = safe_import_numpy()
    if np is None:
        return None, err

    try:
        import cv2
    except ModuleNotFoundError:
        return None, "missing_dependency:cv2"

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, "video_open_failed"

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total > 0:
        target_idx = np.linspace(0, total - 1, num=min(max_frames, total), dtype=int)
        target_set = set(int(i) for i in target_idx.tolist())
    else:
        target_set = set()

    frames: list[Any] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        take = False
        if total > 0:
            take = idx in target_set
        else:
            take = (idx % max(1, int((fps or 24.0) // 2))) == 0 and len(frames) < max_frames
        if take:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            if max(h, w) > max_side:
                scale = max_side / float(max(h, w))
                new_size = (max(16, int(round(w * scale))), max(16, int(round(h * scale))))
                rgb = cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)
            frames.append(rgb.astype(np.float32) / 255.0)
            if len(frames) >= max_frames:
                break
        idx += 1
    cap.release()

    if not frames:
        return None, "no_frames_decoded"

    duration = float(total / fps) if fps > 0 and total > 0 else 0.0
    return VideoFrameBundle(frames_rgb=frames, fps=fps, duration=duration), None
