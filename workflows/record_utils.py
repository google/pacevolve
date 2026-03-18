"""Helpers for writing per-island iteration records."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any


def get_records_run_dir(config: dict, run_name: str) -> str:
  paths = config.setdefault("paths", {})
  base_dir = paths.get("records_dir")
  if base_dir is None:
    log_dir = os.path.expanduser(paths["log_dir"])
    base_dir = os.path.join(os.path.dirname(log_dir), "records")
  base_dir = os.path.expanduser(base_dir)
  run_dir = os.path.join(base_dir, run_name)
  os.makedirs(run_dir, exist_ok=True)
  paths["records_dir"] = base_dir
  paths["records_run_dir"] = run_dir
  return run_dir


def _safe_float(value: Any) -> float | None:
  try:
    return float(value)
  except Exception:
    return None


def _metric_key(raw_key: str) -> str:
  key = raw_key.strip().lower()
  key = key.replace("%", "pct")
  key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
  return key or "metric"


def _extract_key_value_metrics(text: str) -> dict[str, float]:
  if not text:
    return {}
  metrics: dict[str, float] = {}
  pattern = re.compile(
    r"([A-Za-z][A-Za-z0-9_ /\-\(\)%]{0,60})\s*:\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
    re.IGNORECASE,
  )
  for match in pattern.finditer(text):
    value = _safe_float(match.group(2))
    if value is None:
      continue
    metrics[_metric_key(match.group(1))] = value
  return metrics


def parse_eval_metrics(task_eval_utils: Any, eval_results: list[str] | str) -> dict[str, float]:
  parser = getattr(task_eval_utils, "parse_eval_metrics", None)
  if callable(parser):
    try:
      parsed = parser(eval_results)
      if isinstance(parsed, dict):
        metrics = {}
        for key, value in parsed.items():
          value_float = _safe_float(value)
          if value_float is not None:
            metrics[_metric_key(str(key))] = value_float
        if metrics:
          return metrics
    except Exception:
      pass

  if isinstance(eval_results, str):
    return _extract_key_value_metrics(eval_results)

  metrics: dict[str, float] = {}
  for idx, eval_text in enumerate(eval_results):
    extracted = _extract_key_value_metrics(eval_text)
    if len(eval_results) == 1:
      metrics.update(extracted)
    else:
      for key, value in extracted.items():
        metrics[f"{key}_ds{idx}"] = value
  return metrics


def _format_model_responses(transcript) -> str:
  if transcript is None:
    return "No transcript available.\n"

  sections: list[str] = []
  response_count = 0
  for chunk in transcript:
    if getattr(chunk, "role", None) != "model":
      continue
    response_count += 1
    tags = ", ".join(getattr(chunk, "tags", []) or [])
    hidden = getattr(chunk, "hidden", False)
    sections.extend([
      f"## Model Response {response_count}",
      f"tags: {tags or '(none)'}",
      f"hidden: {hidden}",
      "",
      getattr(chunk, "content", "") or "",
      "",
    ])

  if response_count == 0:
    return "No model responses captured.\n"
  return "\n".join(sections).rstrip() + "\n"


def write_iteration_records(
  records_run_dir: str,
  iteration: int,
  island_id: int,
  transcript,
  candidate_code: str,
  eval_results: list[str],
  task_eval_utils: Any,
  *,
  success: bool,
  compile_success: bool,
  eval_success: bool,
  compile_attempts: int,
  eval_attempts: int,
  analysis_attempts: int,
  idea_id: int,
  eval_score: float | None,
  summary_bullets: list[str] | None,
  compile_errors: list[str] | None,
  eval_failures: list[str] | None,
  analysis_errors: list[str] | None,
  analysis_success: bool,
  analysis_metrics: dict[str, float] | None,
  failure_reason: str | None,
  elapsed_seconds: float | None,
  cuda_visible_devices: str | None = None,
) -> tuple[str, str]:
  island_dir = os.path.join(records_run_dir, f"island_{island_id:02d}")
  os.makedirs(island_dir, exist_ok=True)

  record_stem = f"iter_{iteration:04d}"
  txt_path = os.path.join(island_dir, f"{record_stem}.txt")
  json_path = os.path.join(island_dir, f"{record_stem}.json")

  summary_bullets = summary_bullets or []
  compile_errors = compile_errors or []
  eval_failures = eval_failures or []
  analysis_errors = analysis_errors or []
  analysis_metrics = analysis_metrics or {}

  eval_metrics = parse_eval_metrics(task_eval_utils, eval_results)
  if eval_score is not None:
    eval_metrics.setdefault("primary_score", float(eval_score))

  txt_sections = [
    f"iteration: {iteration}",
    f"island_id: {island_id}",
    f"timestamp_utc: {datetime.utcnow().isoformat(timespec='seconds')}Z",
    f"success: {success}",
    f"compile_success: {compile_success}",
    f"eval_success: {eval_success}",
    f"idea_id: {idea_id}",
    f"compile_attempts: {compile_attempts}",
    f"eval_attempts: {eval_attempts}",
    f"analysis_attempts: {analysis_attempts}",
    f"eval_score: {eval_score}",
    f"failure_reason: {failure_reason}",
    f"cuda_visible_devices: {cuda_visible_devices}",
    "",
    "# Model Responses",
    "",
    _format_model_responses(transcript).rstrip(),
    "",
    "# Generated Candidate",
    "",
    candidate_code or "(no candidate code generated)",
    "",
  ]

  if summary_bullets:
    txt_sections.extend([
      "# Summary Bullets",
      "",
      "\n".join(summary_bullets),
      "",
    ])

  with open(txt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(txt_sections).rstrip() + "\n")

  payload = {
    "iteration": iteration,
    "island_id": island_id,
    "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "success": success,
    "compile_success": compile_success,
    "eval_success": eval_success,
    "idea_id": idea_id,
    "compile_attempts": compile_attempts,
    "eval_attempts": eval_attempts,
    "analysis_attempts": analysis_attempts,
    "eval_score": eval_score,
    "eval_metrics": eval_metrics,
    "summary_bullets": summary_bullets,
    "compile_errors": compile_errors,
    "eval_failures": eval_failures,
    "analysis_success": analysis_success,
    "analysis_metrics": analysis_metrics,
    "analysis_errors": analysis_errors,
    "failure_reason": failure_reason,
    "elapsed_seconds": elapsed_seconds,
    "cuda_visible_devices": cuda_visible_devices,
    "eval_results": eval_results,
    "candidate_code": candidate_code,
  }
  with open(json_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")

  return txt_path, json_path
