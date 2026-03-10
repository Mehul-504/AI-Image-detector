from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import analyze_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI detector analysis on JSON payload.")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to analysis payload JSON.",
    )
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze_payload(payload, source="cli")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
