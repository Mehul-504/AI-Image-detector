from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StepResult:
    name: str
    command: list[str]
    return_code: int
    duration_sec: float
    stdout: str
    stderr: str


def _run_step(name: str, command: list[str], env: dict[str, str], cwd: Path) -> StepResult:
    start = time.time()
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    duration = time.time() - start
    return StepResult(
        name=name,
        command=command,
        return_code=int(proc.returncode),
        duration_sec=float(duration),
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def _summarize(step: StepResult) -> dict[str, object]:
    return {
        "name": step.name,
        "return_code": step.return_code,
        "duration_sec": round(step.duration_sec, 3),
        "command": " ".join(step.command),
        "stdout_tail": "\n".join(step.stdout.splitlines()[-40:]),
        "stderr_tail": "\n".join(step.stderr.splitlines()[-40:]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full local cycle: curate dataset -> train fusion -> evaluate balanced/strict."
    )
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--curated-root", default="dataset_curated")
    parser.add_argument("--model-out", default="models/fusion_model.json")
    parser.add_argument("--eval-dir", default="logs/evals")
    parser.add_argument("--report", default="logs/evals/cycle_summary.json")
    parser.add_argument(
        "--curate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run dataset curation before training (default: true)",
    )
    parser.add_argument(
        "--train-on-curated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train on curated-root instead of dataset-root (default: true)",
    )
    parser.add_argument(
        "--hf-local-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force transformers local_files_only mode (default: false)",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Optional AIDETECTOR_DEVICE override (e.g. mps/cuda/cpu)",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    dataset_root = Path(args.dataset_root).resolve()
    curated_root = Path(args.curated_root).resolve()
    model_out = Path(args.model_out).resolve()
    eval_dir = Path(args.eval_dir).resolve()
    report_path = Path(args.report).resolve()

    eval_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    train_dataset_root = curated_root if args.train_on_curated else dataset_root

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "src")
    env["AIDETECTOR_FUSION_MODEL_PATH"] = str(model_out)
    if args.hf_local_only:
        env["AIDETECTOR_HF_LOCAL_ONLY"] = "1"
    if args.device:
        env["AIDETECTOR_DEVICE"] = args.device

    steps: list[StepResult] = []

    if args.curate:
        curate_cmd = [
            sys.executable,
            "-m",
            "aidetector.curate_dataset",
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(curated_root),
            "--report",
            str((eval_dir / "curation_report.json").resolve()),
        ]
        steps.append(_run_step("curate_dataset", curate_cmd, env=env, cwd=cwd))
        if steps[-1].return_code != 0:
            summary = {
                "status": "failed",
                "failed_step": "curate_dataset",
                "steps": [_summarize(step) for step in steps],
            }
            report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary, indent=2))
            return 1

    train_cmd = [
        sys.executable,
        "-m",
        "aidetector.train_fusion",
        "--dataset-root",
        str(train_dataset_root),
        "--out",
        str(model_out),
    ]
    steps.append(_run_step("train_fusion", train_cmd, env=env, cwd=cwd))
    if steps[-1].return_code != 0:
        summary = {
            "status": "failed",
            "failed_step": "train_fusion",
            "steps": [_summarize(step) for step in steps],
        }
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 1

    eval_balanced = eval_dir / "latest_balanced.json"
    eval_strict = eval_dir / "latest_strict.json"
    balanced_cmd = [
        sys.executable,
        "-m",
        "aidetector.eval_dataset",
        "--dataset-root",
        str(dataset_root),
        "--mode-profile",
        "balanced",
        "--out",
        str(eval_balanced.resolve()),
    ]
    strict_cmd = [
        sys.executable,
        "-m",
        "aidetector.eval_dataset",
        "--dataset-root",
        str(dataset_root),
        "--mode-profile",
        "strict",
        "--out",
        str(eval_strict.resolve()),
    ]
    steps.append(_run_step("eval_balanced", balanced_cmd, env=env, cwd=cwd))
    steps.append(_run_step("eval_strict", strict_cmd, env=env, cwd=cwd))

    status = "ok" if all(step.return_code == 0 for step in steps) else "failed"
    failed_step = next((step.name for step in steps if step.return_code != 0), None)
    summary = {
        "status": status,
        "failed_step": failed_step,
        "model_out": str(model_out),
        "balanced_eval": str(eval_balanced),
        "strict_eval": str(eval_strict),
        "train_dataset_root": str(train_dataset_root),
        "steps": [_summarize(step) for step in steps],
    }
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
