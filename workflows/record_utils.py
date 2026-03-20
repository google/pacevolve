"""Helpers for writing per-island step records."""

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


def _serialize_transcript(transcript) -> list[dict[str, Any]]:
  if transcript is None:
    return []

  serialized = []
  for idx, chunk in enumerate(transcript):
    serialized.append({
      "index": idx,
      "role": getattr(chunk, "role", None),
      "tags": list(getattr(chunk, "tags", []) or []),
      "hidden": bool(getattr(chunk, "hidden", False)),
      "content": getattr(chunk, "content", "") or "",
    })
  return serialized


def _format_transcript(transcript, include_hidden: bool) -> str:
  if transcript is None:
    return "No transcript available.\n"

  sections: list[str] = []
  visible_count = 0
  for idx, chunk in enumerate(transcript, start=1):
    hidden = bool(getattr(chunk, "hidden", False))
    if hidden and not include_hidden:
      continue
    visible_count += 1
    tags = ", ".join(getattr(chunk, "tags", []) or [])
    sections.extend([
      f"## Chunk {idx}",
      f"role: {getattr(chunk, 'role', None) or '(unknown)'}",
      f"tags: {tags or '(none)'}",
      f"hidden: {hidden}",
      "",
      getattr(chunk, "content", "") or "",
      "",
    ])

  if visible_count == 0:
    return "No transcript chunks available.\n"
  return "\n".join(sections).rstrip() + "\n"


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


def _format_eval_outputs(eval_results: list[str]) -> str:
  if not eval_results:
    return "No evaluation output captured.\n"
  if len(eval_results) == 1:
    return (eval_results[0] or "").rstrip() + "\n"

  sections: list[str] = []
  for idx, eval_text in enumerate(eval_results):
    sections.extend([
      f"## Eval Output {idx}",
      "",
      (eval_text or "").rstrip(),
      "",
    ])
  return "\n".join(sections).rstrip() + "\n"


def _write_text(path: str, content: str) -> None:
  with open(path, "w", encoding="utf-8") as handle:
    handle.write((content or "").rstrip() + "\n")


def _write_json(path: str, payload: dict[str, Any]) -> None:
  with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")


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
  analysis_mode: str | None = None,
  analysis_results: str | None = None,
  analysis_script: str | None = None,
) -> tuple[str, str]:
  island_dir = os.path.join(records_run_dir, f"island_{island_id:02d}")
  step_dir = os.path.join(island_dir, f"step_{iteration:05d}")
  os.makedirs(step_dir, exist_ok=True)

  summary_bullets = summary_bullets or []
  compile_errors = compile_errors or []
  eval_failures = eval_failures or []
  analysis_errors = analysis_errors or []
  analysis_metrics = analysis_metrics or {}
  analysis_mode = analysis_mode or "disabled"

  timestamp_utc = datetime.utcnow().isoformat(timespec="seconds") + "Z"
  eval_metrics = parse_eval_metrics(task_eval_utils, eval_results)
  if eval_score is not None:
    eval_metrics.setdefault("primary_score", float(eval_score))

  metrics_payload = {
    "iteration": iteration,
    "step_id": iteration,
    "island_id": island_id,
    "timestamp_utc": timestamp_utc,
    "success": success,
    "compile_success": compile_success,
    "eval_success": eval_success,
    "analysis_success": analysis_success,
    "analysis_mode": analysis_mode,
    "idea_id": idea_id,
    "compile_attempts": compile_attempts,
    "eval_attempts": eval_attempts,
    "analysis_attempts": analysis_attempts,
    "eval_score": eval_score,
    "failure_reason": failure_reason,
    "cuda_visible_devices": cuda_visible_devices,
    "elapsed_seconds": elapsed_seconds,
    "summary_bullets": summary_bullets,
    "compile_errors": compile_errors,
    "eval_failures": eval_failures,
    "analysis_errors": analysis_errors,
    "eval_metrics": eval_metrics,
    "analysis_metrics": analysis_metrics,
    "files": {
      "candidate": "candidate.py",
      "eval_output": "eval_output.txt",
      "model_responses": "model_responses.txt",
      "transcript_all": "transcript_all.txt",
      "transcript_visible": "transcript_visible.txt",
      "transcript_jsonl": "transcript.jsonl",
      "summary": "summary.txt",
      "analysis_output": "analysis_output.txt" if analysis_results else None,
      "analysis_script": "analysis_script.py" if analysis_script else None,
    },
  }

  summary_sections = [
    f"iteration: {iteration}",
    f"island_id: {island_id}",
    f"timestamp_utc: {timestamp_utc}",
    f"success: {success}",
    f"compile_success: {compile_success}",
    f"eval_success: {eval_success}",
    f"analysis_success: {analysis_success}",
    f"analysis_mode: {analysis_mode}",
    f"idea_id: {idea_id}",
    f"compile_attempts: {compile_attempts}",
    f"eval_attempts: {eval_attempts}",
    f"analysis_attempts: {analysis_attempts}",
    f"eval_score: {eval_score}",
    f"failure_reason: {failure_reason}",
    f"elapsed_seconds: {elapsed_seconds}",
    f"cuda_visible_devices: {cuda_visible_devices}",
    "",
    "# Eval Metrics",
    "",
    json.dumps(eval_metrics, indent=2, sort_keys=True),
    "",
    "# Analysis Metrics",
    "",
    json.dumps(analysis_metrics, indent=2, sort_keys=True),
    "",
  ]

  if summary_bullets:
    summary_sections.extend([
      "# Summary Bullets",
      "",
      "\n".join(summary_bullets),
      "",
    ])

  candidate_path = os.path.join(step_dir, "candidate.py")
  eval_output_path = os.path.join(step_dir, "eval_output.txt")
  transcript_all_path = os.path.join(step_dir, "transcript_all.txt")
  transcript_visible_path = os.path.join(step_dir, "transcript_visible.txt")
  transcript_jsonl_path = os.path.join(step_dir, "transcript.jsonl")
  model_responses_path = os.path.join(step_dir, "model_responses.txt")
  summary_path = os.path.join(step_dir, "summary.txt")
  metrics_path = os.path.join(step_dir, "metrics.json")
  eval_metrics_path = os.path.join(step_dir, "eval_metrics.json")
  analysis_metrics_path = os.path.join(step_dir, "analysis_metrics.json")

  _write_text(candidate_path, candidate_code or "")
  _write_text(eval_output_path, _format_eval_outputs(eval_results))
  _write_text(transcript_all_path, _format_transcript(transcript, include_hidden=True))
  _write_text(transcript_visible_path, _format_transcript(transcript, include_hidden=False))
  _write_text(model_responses_path, _format_model_responses(transcript))
  _write_text(summary_path, "\n".join(summary_sections))
  _write_json(metrics_path, metrics_payload)
  _write_json(eval_metrics_path, eval_metrics)
  _write_json(analysis_metrics_path, analysis_metrics)

  with open(transcript_jsonl_path, "w", encoding="utf-8") as handle:
    for item in _serialize_transcript(transcript):
      handle.write(json.dumps(item, sort_keys=True) + "\n")

  if analysis_results:
    _write_text(os.path.join(step_dir, "analysis_output.txt"), analysis_results)
  if analysis_script:
    _write_text(os.path.join(step_dir, "analysis_script.py"), analysis_script)

  return summary_path, metrics_path
