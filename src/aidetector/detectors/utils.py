from __future__ import annotations

import os
from pathlib import Path
from dataclasses import replace

from aidetector.schemas import CategoryScore, OverrideHints

from .base import DetectionResult


def neutral_score(reason: str) -> CategoryScore:
    return CategoryScore(risk=50.0, confidence=15.0, rationale=reason)


def detect_device(torch_module: object, preferred: str) -> str:
    if preferred and preferred != "auto":
        return preferred
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    if mps is not None and mps.is_available():
        return "mps"
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        return "cuda"
    return "cpu"


def _truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _hf_cache_roots() -> list[Path]:
    roots: list[Path] = []
    hub_cache = os.getenv("HUGGINGFACE_HUB_CACHE")
    hf_home = os.getenv("HF_HOME")
    if hub_cache:
        roots.append(Path(hub_cache).expanduser())
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")
    if not roots:
        roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def _find_hf_snapshot(model_id: str) -> Path | None:
    normalized = model_id.strip()
    if not normalized:
        return None
    if "/" not in normalized:
        return None
    repo_dir_name = f"models--{normalized.replace('/', '--')}"
    for root in _hf_cache_roots():
        snapshots_dir = root / repo_dir_name / "snapshots"
        if not snapshots_dir.exists():
            continue
        candidates = [path for path in snapshots_dir.iterdir() if path.is_dir()]
        if not candidates:
            continue
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[0]
    return None


def resolve_transformers_source(model_id_or_path: str) -> tuple[str, dict[str, object], str]:
    raw = (model_id_or_path or "").strip()
    local_only = _truthy_env(os.getenv("AIDETECTOR_HF_LOCAL_ONLY"))
    if not raw:
        return raw, ({"local_files_only": True} if local_only else {}), "empty"

    explicit_path = Path(raw).expanduser()
    if explicit_path.exists():
        return str(explicit_path.resolve()), {"local_files_only": True}, "local_path"

    snapshot = _find_hf_snapshot(raw)
    if snapshot is not None:
        return str(snapshot), {"local_files_only": True}, "hf_cache_snapshot"

    if local_only:
        return raw, {"local_files_only": True}, "remote_id_local_only"
    return raw, {}, "remote_id"


def transformers_source_candidates(model_id_or_path: str) -> list[tuple[str, dict[str, object], str]]:
    raw = (model_id_or_path or "").strip()
    local_only = _truthy_env(os.getenv("AIDETECTOR_HF_LOCAL_ONLY"))
    if not raw:
        return [(raw, ({"local_files_only": True} if local_only else {}), "empty")]

    explicit_path = Path(raw).expanduser()
    if explicit_path.exists():
        return [(str(explicit_path.resolve()), {"local_files_only": True}, "local_path")]

    candidates: list[tuple[str, dict[str, object], str]] = []
    snapshot = _find_hf_snapshot(raw)
    if snapshot is not None:
        candidates.append((str(snapshot), {"local_files_only": True}, "hf_cache_snapshot"))

    if local_only:
        candidates.append((raw, {"local_files_only": True}, "remote_id_local_only"))
    else:
        candidates.append((raw, {}, "remote_id"))

    deduped: list[tuple[str, dict[str, object], str]] = []
    seen: set[tuple[str, tuple[tuple[str, object], ...]]] = set()
    for source, kwargs, note in candidates:
        key = (source, tuple(sorted(kwargs.items())))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((source, kwargs, note))
    return deduped


def merge_override(base: OverrideHints, incoming: OverrideHints) -> OverrideHints:
    return OverrideHints(
        provenance_declares_synthetic_or_edited=(
            base.provenance_declares_synthetic_or_edited
            or incoming.provenance_declares_synthetic_or_edited
        ),
        watermark_strong_positive=base.watermark_strong_positive or incoming.watermark_strong_positive,
        provenance_declares_authentic=base.provenance_declares_authentic or incoming.provenance_declares_authentic,
        forensic_contradiction=base.forensic_contradiction or incoming.forensic_contradiction,
    )


def merge_results(base: DetectionResult, incoming: DetectionResult) -> DetectionResult:
    merged_scores = dict(base.scores)
    merged_scores.update(incoming.scores)
    merged_hints = merge_override(base.override_hints, incoming.override_hints)
    return replace(base, scores=merged_scores, override_hints=merged_hints)
