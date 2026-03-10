from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .eval_dataset import CLASS_NAMES, iter_labelled_images

try:
    from PIL import Image, ImageOps
except ModuleNotFoundError:  # pragma: no cover - depends on optional image extras
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]


VALID_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".nef",
    ".cr2",
    ".arw",
}


@dataclass(frozen=True)
class ImageRecord:
    label: str
    path: Path
    sha1: str
    width: int
    height: int
    mode: str
    format_name: str


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(path: Path, digest: str) -> str:
    stem = "".join(char if char.isalnum() else "_" for char in path.stem)
    stem = stem.strip("_") or "image"
    return f"{stem}_{digest[:10]}"


def _open_image(path: Path):
    if Image is None or ImageOps is None:
        raise ModuleNotFoundError("Pillow is required. Install with: pip install -e '.[image]'")
    with Image.open(path) as source:
        oriented = ImageOps.exif_transpose(source)
        try:
            width, height = oriented.size
            mode = str(oriented.mode)
            fmt = str(oriented.format or source.format or path.suffix.lstrip(".") or "unknown")
            rgb = oriented.convert("RGB")
            return width, height, mode, fmt, rgb
        finally:
            oriented.close()


def _copy_record(record: ImageRecord, output_root: Path) -> Path:
    dest_dir = output_root / record.label
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = record.path.suffix.lower() if record.path.suffix.lower() in VALID_SUFFIXES else ".jpg"
    dest = dest_dir / f"{_safe_name(record.path, record.sha1)}{suffix}"
    shutil.copy2(record.path, dest)
    return dest


def _augment_transport_compression(
    record: ImageRecord,
    output_root: Path,
    *,
    max_side: int,
) -> list[Path]:
    _, _, _, _, image = _open_image(record.path)
    generated: list[Path] = []
    try:
        out_dir = output_root / record.label
        out_dir.mkdir(parents=True, exist_ok=True)
        base = _safe_name(record.path, record.sha1)

        # Preserve aspect ratio while forcing network-style downscaling.
        resized = image.copy()
        resized.thumbnail((max_side, max_side))

        jpeg_qualities = (92, 82, 68)
        for quality in jpeg_qualities:
            out_path = out_dir / f"{base}__aug_jpeg_q{quality}.jpg"
            resized.save(
                out_path,
                format="JPEG",
                quality=quality,
                optimize=True,
                progressive=True,
                subsampling="4:2:0",
            )
            generated.append(out_path)

        webp_path = out_dir / f"{base}__aug_webp_q60.webp"
        resized.save(webp_path, format="WEBP", quality=60, method=4)
        generated.append(webp_path)
        return generated
    finally:
        image.close()


def curate_dataset(
    dataset_root: Path,
    output_root: Path,
    *,
    copy_valid: bool,
    augment_compression: bool,
    max_side: int,
) -> dict[str, object]:
    total = 0
    valid = 0
    unreadable = 0
    duplicate_same_label = 0
    duplicate_cross_label = 0
    copied = 0
    augmented = 0

    counts_by_label: Counter[str] = Counter()
    kept_by_label: Counter[str] = Counter()
    hashes: dict[str, ImageRecord] = {}
    duplicate_examples: list[dict[str, str]] = []
    unreadable_files: list[str] = []
    copy_outputs: dict[str, list[str]] = defaultdict(list)

    for label, path in iter_labelled_images(dataset_root):
        total += 1
        counts_by_label[label] += 1
        try:
            width, height, mode, fmt, _ = _open_image(path)
        except ModuleNotFoundError:
            raise
        except Exception:
            unreadable += 1
            unreadable_files.append(str(path))
            continue

        valid += 1
        digest = _sha1_file(path)
        current = ImageRecord(
            label=label,
            path=path,
            sha1=digest,
            width=width,
            height=height,
            mode=mode,
            format_name=fmt,
        )

        previous = hashes.get(digest)
        if previous is not None:
            if previous.label == label:
                duplicate_same_label += 1
            else:
                duplicate_cross_label += 1
                duplicate_examples.append(
                    {
                        "sha1": digest,
                        "first_label": previous.label,
                        "first_file": str(previous.path),
                        "second_label": label,
                        "second_file": str(path),
                    }
                )
            continue

        hashes[digest] = current
        kept_by_label[label] += 1
        if not copy_valid:
            continue

        copied_path = _copy_record(current, output_root)
        copied += 1
        copy_outputs[label].append(str(copied_path))
        if augment_compression:
            generated = _augment_transport_compression(
                current,
                output_root,
                max_side=max_side,
            )
            augmented += len(generated)
            copy_outputs[label].extend(str(item) for item in generated)

    return {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "counts": {
            "total_files_seen": total,
            "valid_files": valid,
            "unreadable_files": unreadable,
            "unique_kept": len(hashes),
            "duplicate_same_label": duplicate_same_label,
            "duplicate_cross_label": duplicate_cross_label,
            "copied": copied,
            "augmented": augmented,
            "output_total": copied + augmented,
        },
        "input_by_label": dict(counts_by_label),
        "kept_by_label": dict(kept_by_label),
        "duplicate_examples": duplicate_examples,
        "unreadable_examples": unreadable_files[:50],
        "output_samples": {label: paths[:20] for label, paths in copy_outputs.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Curate + augment local dataset for training/evaluation.")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--output-root", default="dataset_curated")
    parser.add_argument(
        "--copy-valid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy unique valid files into output-root (default: true)",
    )
    parser.add_argument(
        "--augment-compression",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate transport-compressed variants for robustness (default: true)",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=1600,
        help="Max edge for generated compressed variants (default: 1600)",
    )
    parser.add_argument(
        "--report",
        default="logs/dataset/curation_report.json",
        help="Path for JSON report.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not dataset_root.exists():
        raise SystemExit(f"dataset root not found: {dataset_root}")

    if args.copy_valid:
        output_root.mkdir(parents=True, exist_ok=True)

    summary = curate_dataset(
        dataset_root,
        output_root,
        copy_valid=bool(args.copy_valid),
        augment_compression=bool(args.augment_compression),
        max_side=max(320, int(args.max_side)),
    )

    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
