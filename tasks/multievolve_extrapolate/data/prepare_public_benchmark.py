#!/usr/bin/env python
"""Prepare the public MULTI-evolve extrapolation benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pacevolve.tasks.multievolve_extrapolate.eval import reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the public MULTI-evolve extrapolation benchmark.")
    parser.add_argument(
        "--benchmark-level",
        default="lite",
        choices=["lite", "full"],
        help="Benchmark subset to prepare.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parent),
        help="Task data directory containing raw/ and prepared/ folders.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild prepared artifacts even if they already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    summary = reference.prepare_public_benchmark(
        data_dir=data_dir,
        benchmark_level=args.benchmark_level,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
