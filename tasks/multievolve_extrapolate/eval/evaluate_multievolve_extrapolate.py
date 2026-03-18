"""CLI evaluator for the MULTI-evolve extrapolation benchmark."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import reference


def _load_candidate(candidate_path: str):
    spec = importlib.util.spec_from_file_location("multievolve_extrapolate_submission", candidate_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {candidate_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["multievolve_extrapolate_submission"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "fit_and_predict"):
        raise AttributeError("Missing `fit_and_predict` function")
    return module


def _syntax_check(candidate_path: str) -> None:
    source = Path(candidate_path).read_text(encoding="utf-8")
    compile(source, candidate_path, "exec")
    module_ast = ast.parse(source, filename=candidate_path)
    has_entrypoint = any(
        isinstance(node, ast.FunctionDef) and node.name == "fit_and_predict"
        for node in module_ast.body
    )
    if not has_entrypoint:
        raise AttributeError("Missing `fit_and_predict` function")


def evaluate(candidate_path: str, data_dir: str, benchmark_level: str) -> dict:
    candidate_module = _load_candidate(candidate_path)
    benchmark_payloads = reference.load_prepared_benchmark(Path(data_dir), benchmark_level=benchmark_level)
    return reference.evaluate_predictions(benchmark_payloads, candidate_module)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a MULTI-evolve extrapolation candidate.")
    parser.add_argument("--candidate_path", required=True, type=str)
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--benchmark_level", default="lite", choices=["lite", "full"], type=str)
    parser.add_argument(
        "--syntax_only",
        action="store_true",
        help="Only validate syntax/imports without running the benchmark.",
    )
    args = parser.parse_args()

    try:
        if args.syntax_only:
            _syntax_check(args.candidate_path)
            print("Syntax check passed.")
            return 0

        metrics = evaluate(args.candidate_path, args.data_dir, args.benchmark_level)
        print("Candidate: " + json.dumps(metrics, sort_keys=True))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
