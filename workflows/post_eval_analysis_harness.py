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

"""Post-eval analysis harness.

This script is copied to a temporary file and rewritten during evolution.
The generated code must define:
`analyze_candidate(candidate_source: str, eval_output: str, artifact_info: dict[str, object]) -> dict[str, float]`.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import zlib
from pathlib import Path

_TORCH_STATE = {"loaded": False, "module": None}


def _get_torch():
    if _TORCH_STATE["loaded"]:
        return _TORCH_STATE["module"]
    try:
        import torch as torch_module
    except Exception:  # pragma: no cover - optional dependency in harness fallback
        torch_module = None
    _TORCH_STATE["loaded"] = True
    _TORCH_STATE["module"] = torch_module
    return torch_module


def _safe_float(value) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _tensor_summary(tensors: dict[str, object]) -> dict[str, float]:
    summary = {
        "num_tensors": 0.0,
        "param_count": 0.0,
        "matrix_tensors": 0.0,
        "vector_tensors": 0.0,
        "max_abs": 0.0,
        "abs_mean": 0.0,
        "zero_fraction": 0.0,
    }
    torch = _get_torch()
    if torch is None:
        return summary

    abs_sum = 0.0
    zero_count = 0.0
    for tensor in tensors.values():
        if not isinstance(tensor, torch.Tensor):
            continue
        t = tensor.detach().to("cpu")
        summary["num_tensors"] += 1.0
        summary["param_count"] += float(t.numel())
        if t.ndim >= 2:
            summary["matrix_tensors"] += 1.0
        elif t.ndim == 1:
            summary["vector_tensors"] += 1.0
        if t.numel() == 0:
            continue
        tf = t.float()
        abs_sum += float(tf.abs().sum().item())
        zero_count += float((t == 0).sum().item())
        summary["max_abs"] = max(summary["max_abs"], float(tf.abs().max().item()))

    if summary["param_count"] > 0:
        summary["abs_mean"] = abs_sum / summary["param_count"]
        summary["zero_fraction"] = zero_count / summary["param_count"]
    return summary


def _find_latest(
    search_dirs: list[str],
    exact_names: list[str],
    suffixes: tuple[str, ...] = (),
    excluded_suffixes: tuple[str, ...] = (),
) -> str | None:
    candidates: list[Path] = []
    for directory in search_dirs:
        base = Path(directory)
        if not base.exists():
            continue
        for filename in exact_names:
            direct = base / filename
            if direct.exists():
                candidates.append(direct)
            candidates.extend(base.glob(f"**/{filename}"))
        for suffix in suffixes:
            candidates.extend(base.glob(f"**/*{suffix}"))

    deduped = []
    seen = set()
    for path in candidates:
        path_str = str(path)
        if path_str in seen:
            continue
        seen.add(path_str)
        if excluded_suffixes and path_str.endswith(excluded_suffixes):
            continue
        deduped.append(path)

    if not deduped:
        return None
    deduped.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(deduped[0])


def _parse_eval_output(eval_output: str) -> dict[str, object]:
    train_logs = []
    val_logs = []

    train_pattern = re.compile(
        r"step:(\d+)/(\d+)\s+train_loss:(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s+"
        r"train_time:(-?\d+(?:\.\d+)?)ms\s+step_avg:(-?\d+(?:\.\d+)?)ms"
    )
    val_pattern = re.compile(
        r"step:(\d+)/(\d+)\s+val_loss:(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s+"
        r"val_bpb:(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s+train_time:(-?\d+(?:\.\d+)?)ms\s+"
        r"step_avg:(-?\d+(?:\.\d+)?)ms"
    )

    for match in train_pattern.finditer(eval_output):
        train_logs.append(
            {
                "step": int(match.group(1)),
                "total_steps": int(match.group(2)),
                "train_loss": float(match.group(3)),
                "train_time_ms": float(match.group(4)),
                "step_avg_ms": float(match.group(5)),
            }
        )
    for match in val_pattern.finditer(eval_output):
        val_logs.append(
            {
                "step": int(match.group(1)),
                "total_steps": int(match.group(2)),
                "val_loss": float(match.group(3)),
                "val_bpb": float(match.group(4)),
                "train_time_ms": float(match.group(5)),
                "step_avg_ms": float(match.group(6)),
            }
        )

    lowered = eval_output.lower()
    summary: dict[str, float] = {
        "train_log_count": float(len(train_logs)),
        "val_log_count": float(len(val_logs)),
        "contains_traceback": 1.0 if "traceback" in lowered else 0.0,
        "contains_oom": 1.0 if "out of memory" in lowered else 0.0,
        "contains_nan": 1.0 if re.search(r"\bnan\b", lowered) else 0.0,
    }

    if train_logs:
        summary["first_train_loss"] = train_logs[0]["train_loss"]
        summary["last_train_loss"] = train_logs[-1]["train_loss"]
        summary["min_train_loss"] = min(item["train_loss"] for item in train_logs)
        summary["last_train_step"] = float(train_logs[-1]["step"])
    if val_logs:
        summary["first_val_loss"] = val_logs[0]["val_loss"]
        summary["last_val_loss"] = val_logs[-1]["val_loss"]
        summary["first_val_bpb"] = val_logs[0]["val_bpb"]
        summary["last_val_bpb"] = val_logs[-1]["val_bpb"]
        summary["best_val_bpb"] = min(item["val_bpb"] for item in val_logs)
        summary["last_val_step"] = float(val_logs[-1]["step"])

    candidate_match = re.search(
        r"Candidate val_bpb:(-?\d+(?:\.\d+)?)\s+artifact_bytes:(\d+)\s+size_limit:(\d+)\s+"
        r"size_ok:(True|False)\s+train_time_ms:(-?\d+(?:\.\d+)?)",
        eval_output,
    )
    if candidate_match:
        summary["candidate_val_bpb"] = float(candidate_match.group(1))
        summary["candidate_artifact_bytes"] = float(candidate_match.group(2))
        summary["candidate_size_limit"] = float(candidate_match.group(3))
        summary["candidate_size_ok"] = 1.0 if candidate_match.group(4) == "True" else 0.0
        summary["candidate_train_time_ms"] = float(candidate_match.group(5))

    roundtrip_match = re.search(
        r"final_int8_zlib_roundtrip_exact\s+val_loss:(-?\d+(?:\.\d+)?)\s+val_bpb:(-?\d+(?:\.\d+)?)",
        eval_output,
    )
    if roundtrip_match:
        summary["roundtrip_val_loss"] = float(roundtrip_match.group(1))
        summary["roundtrip_val_bpb"] = float(roundtrip_match.group(2))

    peak_mem_match = re.search(
        r"peak memory allocated:\s*(\d+)\s*MiB\s*reserved:\s*(\d+)\s*MiB",
        eval_output,
    )
    if peak_mem_match:
        summary["peak_memory_allocated_mib"] = float(peak_mem_match.group(1))
        summary["peak_memory_reserved_mib"] = float(peak_mem_match.group(2))

    size_match = re.search(r"Serialized model:\s*(\d+)\s*bytes", eval_output)
    if size_match:
        summary["serialized_model_bytes"] = float(size_match.group(1))
    code_match = re.search(r"Code size:\s*(\d+)\s*bytes", eval_output)
    if code_match:
        summary["code_size_bytes"] = float(code_match.group(1))
    int8_match = re.search(
        r"Serialized model int8\+zlib:\s*(\d+)\s*bytes\s*\(payload:(\d+)\s+raw_torch:(\d+)\s+payload_ratio:(-?\d+(?:\.\d+)?)x\)",
        eval_output,
    )
    if int8_match:
        summary["int8_zlib_bytes"] = float(int8_match.group(1))
        summary["int8_payload_bytes"] = float(int8_match.group(2))
        summary["int8_raw_torch_bytes"] = float(int8_match.group(3))
        summary["int8_payload_ratio_x"] = float(int8_match.group(4))

    if re.search(r"stopping_early:\s+wallclock_cap", eval_output):
        summary["stopped_early_wallclock"] = 1.0

    return {
        "train_logs": train_logs,
        "val_logs": val_logs,
        "summary": summary,
    }


def _extract_state_dict(obj: object) -> dict[str, object]:
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if all(isinstance(key, str) for key in obj.keys()):
            return obj
    return {}


def _summarize_float_checkpoint(path: str | None) -> dict[str, object]:
    summary: dict[str, object] = {
        "exists": 0.0,
        "path": path or "",
        "size_bytes": 0.0,
    }
    if not path:
        return summary
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return summary
    summary["exists"] = 1.0
    summary["size_bytes"] = float(checkpoint_path.stat().st_size)
    try:
        torch = _get_torch()
        if torch is None:
            return summary
        state_dict = _extract_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        summary.update(_tensor_summary(state_dict))
    except Exception as exc:
        summary["load_error"] = str(exc)
    return summary


def _summarize_quantized_artifact(path: str | None) -> dict[str, object]:
    summary: dict[str, object] = {
        "exists": 0.0,
        "path": path or "",
        "size_bytes": 0.0,
    }
    if not path:
        return summary
    artifact_path = Path(path)
    if not artifact_path.exists():
        return summary
    summary["exists"] = 1.0
    summary["size_bytes"] = float(artifact_path.stat().st_size)

    try:
        compressed = artifact_path.read_bytes()
        decompressed = zlib.decompress(compressed)
        summary["decompressed_bytes"] = float(len(decompressed))
        if len(compressed) > 0:
            summary["compression_ratio"] = float(len(decompressed)) / float(len(compressed))
        torch = _get_torch()
        if torch is None:
            return summary
        obj = torch.load(io.BytesIO(decompressed), map_location="cpu")
        quantized = obj.get("quantized", {}) if isinstance(obj, dict) else {}
        scales = obj.get("scales", {}) if isinstance(obj, dict) else {}
        passthrough = obj.get("passthrough", {}) if isinstance(obj, dict) else {}
        qmeta = obj.get("qmeta", {}) if isinstance(obj, dict) else {}
        summary["quantized_tensor_count"] = float(len(quantized))
        summary["scale_tensor_count"] = float(len(scales))
        summary["passthrough_tensor_count"] = float(len(passthrough))
        summary["per_row_quant_tensors"] = float(
            sum(
                1
                for name, scale in scales.items()
                if qmeta.get(name, {}).get("scheme") == "per_row" or getattr(scale, "ndim", 0) > 0
            )
        )
        quantized_stats = _tensor_summary(quantized)
        summary["quantized_param_count"] = quantized_stats["param_count"]
        summary["quantized_int8_abs_mean"] = quantized_stats["abs_mean"]
        summary["quantized_int8_zero_fraction"] = quantized_stats["zero_fraction"]
        passthrough_stats = _tensor_summary(passthrough)
        summary["passthrough_param_count"] = passthrough_stats["param_count"]
    except Exception as exc:
        summary["load_error"] = str(exc)
    return summary


def _flatten_numeric_payload(
    value: object,
    prefix: str,
    output: dict[str, float],
    max_items: int = 64,
) -> None:
    if len(output) >= max_items:
        return
    if isinstance(value, bool):
        output[prefix] = 1.0 if value else 0.0
        return
    if isinstance(value, (int, float)):
        output[prefix] = float(value)
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if len(output) >= max_items:
                break
            child_prefix = f"{prefix}_{key}" if prefix else str(key)
            _flatten_numeric_payload(child, child_prefix, output, max_items=max_items)
        return
    if isinstance(value, list):
        for idx, child in enumerate(value[:8]):
            if len(output) >= max_items:
                break
            child_prefix = f"{prefix}_{idx}" if prefix else str(idx)
            _flatten_numeric_payload(child, child_prefix, output, max_items=max_items)


def _summarize_json_artifact(path: str | None) -> dict[str, object]:
    summary: dict[str, object] = {
        "exists": 0.0,
        "path": path or "",
        "size_bytes": 0.0,
        "numeric_summary": {},
        "payload": {},
    }
    if not path:
        return summary
    artifact_path = Path(path)
    if not artifact_path.exists():
        return summary
    summary["exists"] = 1.0
    summary["size_bytes"] = float(artifact_path.stat().st_size)
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        summary["payload"] = payload
        if isinstance(payload, dict):
            summary["top_level_keys"] = sorted(payload.keys())
        numeric_summary: dict[str, float] = {}
        _flatten_numeric_payload(payload, "", numeric_summary)
        summary["numeric_summary"] = numeric_summary
    except Exception as exc:
        summary["load_error"] = str(exc)
    return summary


def build_artifact_info(src_path: str, eval_output: str, results_path: str | None = None) -> dict[str, object]:
    search_dirs = []
    for candidate_dir in [src_path, results_path]:
        if not candidate_dir:
            continue
        candidate_dir = os.path.abspath(candidate_dir)
        if candidate_dir not in search_dirs:
            search_dirs.append(candidate_dir)

    float_ckpt_path = _find_latest(
        search_dirs,
        exact_names=[
            "final_model.pt",
            "final_model.pth",
            "best_model.pt",
            "best_model.pth",
            "model_best.pt",
            "model_best.pth",
            "model.pt",
            "model.pth",
            "checkpoint.pt",
            "checkpoint.pth",
            "last.ckpt",
            "checkpoint_last.pt",
            "checkpoint_last.pth",
        ],
    )
    quant_artifact_path = _find_latest(
        search_dirs,
        exact_names=[
            "final_model.int8.ptz",
            "model.int8.ptz",
            "checkpoint.int8.ptz",
        ],
    )
    structured_artifact_path = _find_latest(
        search_dirs,
        exact_names=[
            "analysis_artifact.json",
        ],
    )

    structured_artifact = _summarize_json_artifact(structured_artifact_path)
    task_artifact = structured_artifact.get("payload", {}) if isinstance(structured_artifact, dict) else {}

    return {
        "search_dirs": search_dirs,
        "parsed_eval": _parse_eval_output(eval_output),
        "float_checkpoint": _summarize_float_checkpoint(float_ckpt_path),
        "quantized_artifact": _summarize_quantized_artifact(quant_artifact_path),
        "structured_artifact": structured_artifact,
        "task_artifact": task_artifact,
    }


# RegexTagPostEvalAnalysisStart
def analyze_candidate(candidate_source: str, eval_output: str, artifact_info: dict[str, object]) -> dict[str, float]:
    """Fallback post-eval analysis using eval/output and artifact summaries."""
    parsed_eval = artifact_info.get("parsed_eval", {}) if isinstance(artifact_info, dict) else {}
    summary = parsed_eval.get("summary", {}) if isinstance(parsed_eval, dict) else {}
    float_ckpt = artifact_info.get("float_checkpoint", {}) if isinstance(artifact_info, dict) else {}
    quantized = artifact_info.get("quantized_artifact", {}) if isinstance(artifact_info, dict) else {}
    structured = artifact_info.get("structured_artifact", {}) if isinstance(artifact_info, dict) else {}

    def metric(source: dict[str, object], key: str, default: float = 0.0) -> float:
        value = source.get(key, default)
        parsed = _safe_float(value)
        return default if parsed is None else parsed

    candidate_bpb = metric(summary, "candidate_val_bpb", metric(summary, "last_val_bpb"))
    best_val_bpb = metric(summary, "best_val_bpb", candidate_bpb)
    artifact_bytes = metric(summary, "candidate_artifact_bytes", metric(quantized, "size_bytes"))
    size_limit = metric(summary, "candidate_size_limit")
    train_time_ms = metric(summary, "candidate_train_time_ms")
    roundtrip_bpb = metric(summary, "roundtrip_val_bpb", candidate_bpb)
    first_train_loss = metric(summary, "first_train_loss")
    last_train_loss = metric(summary, "last_train_loss")
    first_val_bpb = metric(summary, "first_val_bpb", best_val_bpb)

    return {
        "analysis_candidate_val_bpb": candidate_bpb,
        "analysis_best_val_bpb_seen": best_val_bpb,
        "analysis_train_loss_reduction": max(first_train_loss - last_train_loss, 0.0),
        "analysis_val_bpb_reduction": max(first_val_bpb - best_val_bpb, 0.0),
        "analysis_artifact_utilization": artifact_bytes / size_limit if size_limit > 0 else 0.0,
        "analysis_wallclock_utilization": train_time_ms / 600000.0 if train_time_ms > 0 else 0.0,
        "analysis_quantization_gap_bpb": max(roundtrip_bpb - best_val_bpb, 0.0),
        "analysis_train_log_count": metric(summary, "train_log_count"),
        "analysis_val_log_count": metric(summary, "val_log_count"),
        "analysis_peak_memory_allocated_gib": metric(summary, "peak_memory_allocated_mib") / 1024.0,
        "analysis_float_ckpt_present": metric(float_ckpt, "exists"),
        "analysis_quantized_artifact_present": metric(quantized, "exists"),
        "analysis_float_param_count": metric(float_ckpt, "param_count"),
        "analysis_float_zero_fraction": metric(float_ckpt, "zero_fraction"),
        "analysis_quantized_zero_fraction": metric(quantized, "quantized_int8_zero_fraction"),
        "analysis_checkpoint_size_mib": metric(float_ckpt, "size_bytes") / (1024.0 * 1024.0),
        "analysis_structured_artifact_present": metric(structured, "exists"),
        "analysis_structured_artifact_size_mib": metric(structured, "size_bytes") / (1024.0 * 1024.0),
        "analysis_eval_traceback_flag": metric(summary, "contains_traceback"),
        "analysis_eval_oom_flag": metric(summary, "contains_oom"),
        "analysis_eval_nan_flag": metric(summary, "contains_nan"),
    }
# RegexTagPostEvalAnalysisEnd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_path", type=str, required=True)
    parser.add_argument("--eval_output_path", type=str, required=True)
    parser.add_argument("--src_path", type=str, required=True)
    parser.add_argument("--results_path", type=str, default=None)
    args = parser.parse_args()

    try:
        with open(args.candidate_path, "r", encoding="utf-8") as f:
            candidate_source = f.read()
        with open(args.eval_output_path, "r", encoding="utf-8") as f:
            eval_output = f.read()

        artifact_info = build_artifact_info(args.src_path, eval_output, args.results_path)
        metrics = analyze_candidate(candidate_source, eval_output, artifact_info)
        if not isinstance(metrics, dict):
            print("AnalysisMetrics: {}", flush=True)
            print("Analyzer did not return a dictionary.", file=sys.stderr)
            return 1

        clean_metrics = {}
        for key, value in metrics.items():
            try:
                clean_metrics[str(key)] = float(value)
            except Exception:
                continue
        print("AnalysisMetrics: " + json.dumps(clean_metrics, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(f"Post-eval analysis failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
