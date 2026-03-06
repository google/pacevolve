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
    """Fallback analysis using task-facing keyword and syntax proxies."""
    lowered = candidate_source.lower()

    def count_any(patterns: list[str]) -> float:
        total = 0
        for pattern in patterns:
            total += len(re.findall(pattern, lowered))
        return float(total)

    call_count = 0
    assign_count = 0
    return_count = 0
    try:
        tree = ast.parse(candidate_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_count += 1
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                assign_count += 1
            elif isinstance(node, ast.Return):
                return_count += 1
    except Exception:
        pass

    return {
        "analysis_loss_keyword_count": count_any([r"\bloss\b", r"\bobjective\b"]),
        "analysis_reward_keyword_count": count_any([r"\breward\b", r"\breturns?\b"]),
        "analysis_entropy_keyword_count": count_any([r"\bentropy\b"]),
        "analysis_kl_keyword_count": count_any([r"\bkl\b", r"kl_div", r"kldiv"]),
        "analysis_grad_norm_keyword_count": count_any([r"grad[_\- ]?norm", r"clip[_\- ]?grad"]),
        "analysis_clip_keyword_count": count_any([r"\bclip\b", r"clamp"]),
        "analysis_correction_keyword_count": count_any([r"\bcorrection\b", r"\bimportance\b", r"\bweight(s|ing)?\b", r"\btis\b"]),
        "analysis_normalization_keyword_count": count_any([r"\bnorm(?:aliz\w*)?\b", r"\bstandardiz\w*\b", r"\bscale\w*\b"]),
        "analysis_logprob_keyword_count": count_any([r"log[_\- ]?prob", r"logprob"]),
        "analysis_advantage_keyword_count": count_any([r"\badvantage\b", r"\bgae\b"]),
        "analysis_balance_keyword_count": count_any([r"\bbalance\w*\b", r"\bload\w*\b", r"\brebalance\w*\b"]),
        "analysis_throughput_keyword_count": count_any([r"\bthroughput\b", r"\blatency\b", r"\bspeed\b", r"\bbandwidth\b"]),
        "analysis_tensor_keyword_count": count_any([r"\btensor\b", r"\btorch\b", r"\bdevice\b", r"\bgpu\b"]),
        "analysis_call_count": float(call_count),
        "analysis_assignment_count": float(assign_count),
        "analysis_return_count": float(return_count),
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
