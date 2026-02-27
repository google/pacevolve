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

"""Utilities for iteration-level analysis and post-mortem memory."""

import dataclasses
import json
import logging
import os
import re
from collections import Counter, defaultdict, deque
from datetime import datetime
from typing import Any

logger = logging.getLogger("controller")


@dataclasses.dataclass
class IterationAnalysisRecord:
  """Structured analytics payload for one iteration."""

  iteration: int
  island_id: int
  success: bool
  compile_success: bool
  eval_success: bool

  compile_attempts: int = 0
  eval_attempts: int = 0
  analysis_attempts: int = 0
  idea_id: int = -1

  eval_score: float | None = None
  eval_metrics: dict[str, float] = dataclasses.field(default_factory=dict)
  analysis_success: bool = False
  analysis_metrics: dict[str, float] = dataclasses.field(default_factory=dict)
  summary_bullets: list[str] = dataclasses.field(default_factory=list)
  compile_errors: list[str] = dataclasses.field(default_factory=list)
  eval_failures: list[str] = dataclasses.field(default_factory=list)
  analysis_errors: list[str] = dataclasses.field(default_factory=list)
  eval_results: list[str] = dataclasses.field(default_factory=list)

  failure_reason: str | None = None
  elapsed_seconds: float | None = None


def _safe_float(value: Any) -> float | None:
  try:
    return float(value)
  except Exception:
    return None


def _truncate(text: str, max_chars: int = 220) -> str:
  if text is None:
    return ""
  text = " ".join(str(text).split())
  if len(text) <= max_chars:
    return text
  return text[: max_chars - 3] + "..."


def _metric_key(raw_key: str) -> str:
  key = raw_key.strip().lower()
  key = key.replace("%", "pct")
  key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
  return key or "metric"


def _extract_key_value_metrics(text: str) -> dict[str, float]:
  """Best-effort extraction of numeric key-value metrics from logs."""
  if not text:
    return {}
  metrics: dict[str, float] = {}
  pattern = re.compile(
      r"([A-Za-z][A-Za-z0-9_ /\-\(\)%]{0,60})\s*:\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
      re.IGNORECASE
  )
  for match in pattern.finditer(text):
    key = _metric_key(match.group(1))
    value = _safe_float(match.group(2))
    if value is None:
      continue
    metrics[key] = value
  return metrics


def _classify_failure(message: str) -> str:
  if not message:
    return "unknown"
  msg = message.lower()
  checks = [
      (r"syntaxerror|invalid syntax|parse error", "syntax"),
      (r"nameerror|undefined reference|is not defined|undeclared", "undefined_symbol"),
      (r"importerror|module not found|no module named", "import"),
      (r"timeout|timed out", "timeout"),
      (r"out of memory|oom|cuda error: out of memory", "oom"),
      (r"shape mismatch|dimension mismatch|size mismatch|broadcast", "shape_mismatch"),
      (r"assert|assertionerror", "assertion"),
      (r"segmentation fault|sigsegv", "segfault"),
      (r"compile|compiler|error:", "compile_error"),
      (r"runtimeerror|traceback|exception", "runtime_error"),
  ]
  for pattern, label in checks:
    if re.search(pattern, msg):
      return label
  return "other"


class AnalysisManager:
  """Stores per-iteration analytics and generates post-mortem memory."""

  def __init__(
      self,
      metric_direction: str,
      jsonl_path: str,
      report_path: str,
      task_eval_utils: Any = None,
      history_window: int = 60,
      max_context_chars: int = 2400,
      recent_analysis_window: int = 3,
  ):
    self.metric_direction = metric_direction
    self.jsonl_path = jsonl_path
    self.report_path = report_path
    self.task_eval_utils = task_eval_utils
    self.history_window = max(10, history_window)
    self.max_context_chars = max(500, max_context_chars)
    self.recent_analysis_window = max(1, recent_analysis_window)

    self._records: list[dict[str, Any]] = []
    self._best_score_by_island: dict[int, float] = {}
    self._last_improve_iteration_by_island: dict[int, int] = {}
    self._compile_failure_counts: Counter[str] = Counter()
    self._eval_failure_counts: Counter[str] = Counter()
    self._recent_analysis_notes_by_island: dict[int, deque[str]] = defaultdict(
      lambda: deque(maxlen=self.recent_analysis_window)
    )

    os.makedirs(os.path.dirname(self.jsonl_path), exist_ok=True)
    os.makedirs(os.path.dirname(self.report_path), exist_ok=True)

  def _is_better(self, new_score: float, old_score: float) -> bool:
    if self.metric_direction == "max":
      return new_score > old_score
    return new_score < old_score

  def _extract_eval_metrics(self, record: IterationAnalysisRecord) -> dict[str, float]:
    metrics: dict[str, float] = dict(record.eval_metrics or {})
    metrics.update(dict(record.analysis_metrics or {}))
    if metrics:
      return metrics

    parser = getattr(self.task_eval_utils, "parse_eval_metrics", None)
    if callable(parser):
      try:
        parsed = parser(record.eval_results if len(record.eval_results) > 1 else (record.eval_results[0] if record.eval_results else ""))
        if isinstance(parsed, dict):
          for k, v in parsed.items():
            v_float = _safe_float(v)
            if v_float is not None:
              metrics[_metric_key(k)] = v_float
      except Exception as e:
        logger.debug(f"parse_eval_metrics failed, falling back to regex extraction: {e}")

    if not metrics and record.eval_results:
      for idx, eval_text in enumerate(record.eval_results):
        extracted = _extract_key_value_metrics(eval_text)
        if len(record.eval_results) == 1:
          metrics.update(extracted)
        else:
          for k, v in extracted.items():
            metrics[f"{k}_ds{idx}"] = v

    return metrics

  def _score_delta_vs_best(self, island_id: int, score: float) -> tuple[float | None, bool]:
    old_best = self._best_score_by_island.get(island_id)
    if old_best is None:
      self._best_score_by_island[island_id] = score
      return None, True

    if self.metric_direction == "max":
      delta = score - old_best
    else:
      delta = old_best - score

    improved = delta > 0
    if improved:
      self._best_score_by_island[island_id] = score
    return delta, improved

  def record_iteration(self, record: IterationAnalysisRecord) -> None:
    metrics = self._extract_eval_metrics(record)
    if record.eval_score is not None:
      metrics.setdefault("primary_score", float(record.eval_score))

    compile_labels = [_classify_failure(msg) for msg in record.compile_errors if msg]
    eval_labels = [_classify_failure(msg) for msg in record.eval_failures if msg]
    analysis_labels = [_classify_failure(msg) for msg in record.analysis_errors if msg]
    self._compile_failure_counts.update(compile_labels)
    self._eval_failure_counts.update(eval_labels)
    self._eval_failure_counts.update(analysis_labels)

    score_delta = None
    score_improved = False
    if record.eval_score is not None:
      score_delta, score_improved = self._score_delta_vs_best(record.island_id, record.eval_score)
      if score_improved:
        self._last_improve_iteration_by_island[record.island_id] = record.iteration

    payload = {
      "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
      "iteration": record.iteration,
      "island_id": record.island_id,
      "success": record.success,
      "compile_success": record.compile_success,
      "eval_success": record.eval_success,
      "compile_attempts": record.compile_attempts,
      "eval_attempts": record.eval_attempts,
      "analysis_attempts": record.analysis_attempts,
      "idea_id": record.idea_id,
      "eval_score": record.eval_score,
      "score_delta_vs_island_best": score_delta,
      "score_improved": score_improved,
      "analysis_success": record.analysis_success,
      "failure_reason": record.failure_reason,
      "elapsed_seconds": record.elapsed_seconds,
      "eval_metrics": metrics,
      "summary_bullets": record.summary_bullets[:3],
      "compile_error_labels": compile_labels[:5],
      "eval_failure_labels": eval_labels[:5],
      "analysis_error_labels": analysis_labels[:5],
      "compile_error_examples": [_truncate(x, 240) for x in record.compile_errors[:3]],
      "eval_failure_examples": [_truncate(x, 240) for x in record.eval_failures[:3]],
      "analysis_error_examples": [_truncate(x, 240) for x in record.analysis_errors[:3]],
    }

    self._records.append(payload)
    self._update_recent_analysis_window(payload)
    with open(self.jsonl_path, "a", encoding="utf-8") as f:
      f.write(json.dumps(payload) + "\n")

    self._write_postmortem_report()

  def _update_recent_analysis_window(self, payload: dict[str, Any]) -> None:
    island_id = int(payload["island_id"])
    iter_idx = int(payload["iteration"])
    analysis_state = "ok" if payload.get("analysis_success") else "fail"

    analysis_metrics = payload.get("eval_metrics", {}) or {}
    analysis_only_metrics = sorted(
      [(k, v) for k, v in analysis_metrics.items() if str(k).startswith("analysis_")],
      key=lambda kv: kv[0],
    )
    metrics_preview = ", ".join(
      [f"{k}={v:.4g}" for k, v in analysis_only_metrics[:3]]
    )
    if not metrics_preview:
      metrics_preview = "no_analysis_metrics"

    failure_reason = payload.get("failure_reason") or "none"
    note = (
      f"iter={iter_idx} analysis={analysis_state} "
      f"score={payload.get('eval_score')} failure={failure_reason} "
      f"metrics[{metrics_preview}]"
    )
    self._recent_analysis_notes_by_island[island_id].append(_truncate(note, 220))

  def _recent_records_for_island(self, island_id: int | None) -> list[dict[str, Any]]:
    if not self._records:
      return []
    if island_id is None:
      return self._records[-self.history_window:]
    filtered = [r for r in self._records if r["island_id"] == island_id]
    if not filtered:
      return self._records[-self.history_window:]
    return filtered[-self.history_window:]

  def build_reasoning_context(self, island_id: int | None = None) -> str:
    recent = self._recent_records_for_island(island_id)
    if not recent:
      return ""

    compile_counts = Counter()
    eval_counts = Counter()
    recent_wins: list[tuple[int, str, float | None]] = []

    for rec in recent:
      compile_counts.update(rec.get("compile_error_labels", []))
      eval_counts.update(rec.get("eval_failure_labels", []))
      eval_counts.update(rec.get("analysis_error_labels", []))
      if rec.get("score_improved") and rec.get("summary_bullets"):
        recent_wins.append((
          rec["iteration"],
          rec["summary_bullets"][0],
          rec.get("score_delta_vs_island_best")
        ))

    lines = [
      "### Post-mortem Memory (Auto-generated)",
      f"- Recent window: {len(recent)} iterations.",
    ]

    if island_id is not None:
      best_score = self._best_score_by_island.get(island_id)
      if best_score is not None:
        lines.append(f"- Best score on island {island_id}: {best_score:.6g}.")
      if island_id in self._last_improve_iteration_by_island:
        stagnation = recent[-1]["iteration"] - self._last_improve_iteration_by_island[island_id]
        lines.append(f"- Stagnation on island {island_id}: {stagnation} iteration(s) since last best-score improvement.")

    if compile_counts:
      major = ", ".join([f"{k} ({v})" for k, v in compile_counts.most_common(3)])
      lines.append(f"- Frequent compile failures: {major}.")
    if eval_counts:
      major = ", ".join([f"{k} ({v})" for k, v in eval_counts.most_common(3)])
      lines.append(f"- Frequent runtime/eval failures: {major}.")

    if recent_wins:
      for iter_idx, bullet, delta in recent_wins[-2:]:
        delta_text = "" if delta is None else f" (delta vs prior best: {delta:.4g})"
        lines.append(f"- Winning pattern at iter {iter_idx}{delta_text}: {_truncate(bullet, 280)}")

    recent_notes = list(self._recent_analysis_notes_by_island.get(island_id, []))
    if not recent_notes and island_id is None:
      all_notes = []
      for notes in self._recent_analysis_notes_by_island.values():
        all_notes.extend(list(notes))
      recent_notes = all_notes[-self.recent_analysis_window:]
    if recent_notes:
      lines.append(f"- Recent post-mortem window (last {len(recent_notes)}):")
      for note in recent_notes[-self.recent_analysis_window:]:
        lines.append(f"  - {note}")

    lines.append(
      "- Decision rule: avoid repeating failure signatures above; prioritize experiments similar to recent winning patterns."
    )

    context = "\n".join(lines)
    if len(context) > self.max_context_chars:
      context = context[: self.max_context_chars - 3] + "..."
    return context

  def _write_postmortem_report(self) -> None:
    total = len(self._records)
    if total == 0:
      return

    success_count = sum(1 for r in self._records if r["success"])
    compile_fail_count = sum(1 for r in self._records if not r["compile_success"])
    eval_fail_count = sum(1 for r in self._records if r["compile_success"] and not r["eval_success"])

    lines = [
      "# Evolution Post-mortem",
      "",
      f"- Total iterations analyzed: {total}",
      f"- Success rate: {success_count / total:.2%} ({success_count}/{total})",
      f"- Compile failure rate: {compile_fail_count / total:.2%} ({compile_fail_count}/{total})",
      f"- Eval/runtime failure rate: {eval_fail_count / total:.2%} ({eval_fail_count}/{total})",
      "",
      "## Best Scores By Island",
    ]

    if self._best_score_by_island:
      for island_id in sorted(self._best_score_by_island):
        lines.append(f"- Island {island_id}: {self._best_score_by_island[island_id]:.6g}")
    else:
      lines.append("- No successful scored iterations yet.")

    lines.extend(["", "## Frequent Failure Signatures"])
    if self._compile_failure_counts:
      compile_str = ", ".join([f"{k} ({v})" for k, v in self._compile_failure_counts.most_common(5)])
      lines.append(f"- Compile: {compile_str}")
    else:
      lines.append("- Compile: none")
    if self._eval_failure_counts:
      eval_str = ", ".join([f"{k} ({v})" for k, v in self._eval_failure_counts.most_common(5)])
      lines.append(f"- Eval/runtime: {eval_str}")
    else:
      lines.append("- Eval/runtime: none")

    lines.extend(["", "## Recent Winning Iterations"])
    winning = [r for r in self._records if r.get("score_improved")]
    if winning:
      for rec in winning[-5:]:
        summary = rec["summary_bullets"][0] if rec["summary_bullets"] else "No summary bullet"
        delta = rec.get("score_delta_vs_island_best")
        delta_text = "n/a" if delta is None else f"{delta:.4g}"
        lines.append(
          f"- Iter {rec['iteration']} island {rec['island_id']} "
          f"(delta={delta_text}): {_truncate(summary, 260)}"
        )
    else:
      lines.append("- No score-improving iterations yet.")

    with open(self.report_path, "w", encoding="utf-8") as f:
      f.write("\n".join(lines) + "\n")
