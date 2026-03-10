from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{uuid4().hex[:8]}"


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")
    return cleaned or "upload.bin"


class RunLogger:
    def __init__(self, root: str | Path | None = None) -> None:
        root_path = Path(root or os.getenv("AIDETECTOR_LOG_DIR", "logs"))
        self.root = root_path.resolve()
        self.runs_dir = self.root / "runs"
        self.uploads_dir = self.root / "uploads"
        self.index_path = self.root / "runs_index.jsonl"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def run_path(self, run_id: str) -> Path:
        return self.runs_dir / f"{run_id}.json"

    def store_upload(self, filename: str, content: bytes, *, run_id: str | None = None) -> Path:
        run_id = run_id or new_run_id()
        safe_name = _safe_name(filename)
        path = self.uploads_dir / f"{run_id}_{safe_name}"
        path.write_bytes(content)
        return path

    def log_run(
        self,
        *,
        run_id: str,
        source: str,
        payload: dict[str, Any],
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> Path:
        timestamp = utc_now_iso()
        record = {
            "run_id": run_id,
            "timestamp_utc": timestamp,
            "source": source,
            "payload": payload,
            "result": result,
            "error": error,
        }
        run_file = self.run_path(run_id)
        run_file.write_text(json.dumps(record, indent=2, ensure_ascii=True), encoding="utf-8")

        summary = {
            "run_id": run_id,
            "timestamp_utc": timestamp,
            "source": source,
            "media_type": payload.get("media_type"),
            "media_uri": payload.get("media_uri"),
            "verdict": (result or {}).get("verdict"),
            "overall_risk": (result or {}).get("overall_risk"),
            "overall_confidence": (result or {}).get("overall_confidence"),
            "error": error,
            "run_file": str(run_file),
        }
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=True) + "\n")

        return run_file

    def recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        lines = self.index_path.read_text(encoding="utf-8").splitlines()
        selected = lines[-max(1, limit) :]
        results: list[dict[str, Any]] = []
        for line in reversed(selected):
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return results
