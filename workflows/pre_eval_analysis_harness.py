# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pre-eval analysis harness.

This script is rewritten during evolution. The generated code must define:
`analyze_candidate(candidate_source: str) -> dict[str, float]`.
"""

import argparse
import ast
import json
import re
import sys


# RegexTagPreEvalAnalysisStart
def analyze_candidate(candidate_source: str) -> dict[str, float]:
    """Fallback analysis if LLM-generated analyzer is unavailable."""
    lines = candidate_source.splitlines()
    nonempty = [line for line in lines if line.strip()]
    comment_lines = [
        line for line in lines
        if line.strip().startswith("#") or line.strip().startswith("//")
    ]

    func_count = 0
    class_count = 0
    try:
        tree = ast.parse(candidate_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_count += 1
            elif isinstance(node, ast.ClassDef):
                class_count += 1
    except Exception:
        pass

    return {
        "analysis_lines_total": float(len(lines)),
        "analysis_lines_nonempty": float(len(nonempty)),
        "analysis_comment_ratio": float(len(comment_lines)) / max(1.0, float(len(nonempty))),
        "analysis_char_count": float(len(candidate_source)),
        "analysis_function_count": float(func_count),
        "analysis_class_count": float(class_count),
        "analysis_loop_token_count": float(len(re.findall(r"\\b(for|while)\\b", candidate_source))),
        "analysis_conditional_token_count": float(len(re.findall(r"\\b(if|elif|else)\\b", candidate_source))),
    }
# RegexTagPreEvalAnalysisEnd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_path", type=str, required=True)
    args = parser.parse_args()

    try:
        with open(args.candidate_path, "r", encoding="utf-8") as f:
            candidate_source = f.read()
        metrics = analyze_candidate(candidate_source)
        if not isinstance(metrics, dict):
            print("AnalysisMetrics: {}", flush=True)
            print("Analyzer did not return a dictionary.", file=sys.stderr)
            return 1
        # Keep only numeric values.
        clean_metrics = {}
        for key, value in metrics.items():
            try:
                clean_metrics[str(key)] = float(value)
            except Exception:
                continue
        print("AnalysisMetrics: " + json.dumps(clean_metrics, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(f"Pre-eval analysis failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
