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

"""PACEvolve evaluation helpers for the Parameter Golf task.

Metric: val_bpb (bits per byte) on FineWeb validation set.
Constraints: total artifact (code + int8+zlib model) <= 16 MB, training <= 10 min.
Lower val_bpb is better (metric_direction: min).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import subprocess
from typing import Optional

logger = logging.getLogger("controller")

ARTIFACT_SIZE_LIMIT = 16_000_000


@dataclasses.dataclass
class CompletedProcess:
    args: str
    returncode: int
    stdout: str
    stderr: str


def _call_shell_command(
    command: str,
    timeout: int,
    max_retries: int,
    cwd: str | None = None,
) -> Optional[CompletedProcess]:
    for _ in range(max_retries):
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            return CompletedProcess(
                args=command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired:
            continue
    return None


@dataclasses.dataclass
class EvalConfig:
    """Evaluation configuration for the Parameter Golf task."""
    dataset: str


def _build_run_command(config: dict, mode: str = "train") -> str:
    """Build the shell command to invoke run.sh."""
    eval_path = os.path.expanduser(config["paths"]["eval_path"])
    run_script = os.path.join(eval_path, config["evaluation"]["eval_script_name"])
    data_path = os.path.expanduser(config["paths"]["data_path"])
    tokenizer_path = os.path.expanduser(config["paths"]["tokenizer_path"])
    # conda_env = config["compilation"]["conda_env"]
    nproc = config["evaluation"].get("nproc_per_node", 8)
    return f"{run_script} '{data_path}' '{tokenizer_path}' {nproc} {mode}"


def recompile_library(config: dict) -> CompletedProcess:
    """Run a Python syntax check on the training script."""
    command = _build_run_command(config, mode="syntax")
    src_path = os.path.expanduser(config["paths"]["src_path"])
    logger.info(f"recompile_library: Running command: {command}")
    process_result = _call_shell_command(
        command,
        timeout=config["compilation"]["recompile_timeout"],
        max_retries=config["compilation"]["recompile_max_retries"],
        cwd=src_path,
    )
    if not process_result:
        return CompletedProcess(
            args=command,
            returncode=-1,
            stdout="",
            stderr="Syntax check failed to complete.",
        )
    success = process_result.returncode == 0
    logger.info(f"recompile_library: Success: {success}.")
    for line in process_result.stdout.splitlines():
        logger.debug(f"recompile_library: STDOUT: {line}")
    for line in process_result.stderr.splitlines():
        logger.debug(f"recompile_library: STDERR: {line}")
    return CompletedProcess(
        args=command,
        returncode=process_result.returncode,
        stdout=process_result.stdout.strip(),
        stderr=process_result.stderr.strip(),
    )


def evaluate_dataset(
    candidate_id: int,
    baseline_id: int,
    eval_config: EvalConfig,
    config: dict,
) -> CompletedProcess:
    """Run the full training + evaluation pipeline."""
    del candidate_id, baseline_id
    eval_command = _build_run_command(config, mode="train")
    src_path = os.path.expanduser(config["paths"]["src_path"])

    logger.info(f"evaluate_dataset: Running {eval_command}")
    process_result = _call_shell_command(
        eval_command,
        timeout=config["evaluation"]["eval_timeout"],
        max_retries=config["evaluation"]["eval_max_retries"],
        cwd=src_path,
    )
    if not process_result:
        logger.error(
            f"evaluate_dataset: Training for {eval_config.dataset} failed."
        )
        return CompletedProcess(
            args=eval_command,
            returncode=-1,
            stdout="",
            stderr=f"evaluate_dataset for {eval_config.dataset} failed to complete.",
        )
    return process_result


def parse_eval_results(
    eval_results: list[str] | str,
) -> list[float | None] | float | None:
    """Extract val_bpb from training output.

    Parses the summary line:
        Candidate val_bpb:<float> artifact_bytes:<int> ... size_ok:<True|False> ...

    Returns the val_bpb as a float if the artifact fits within the size limit,
    or None if the result is invalid or the artifact exceeds the limit.
    """
    if isinstance(eval_results, str):
        # Primary pattern: the structured Candidate summary line
        pattern = r"Candidate val_bpb:(\d+\.\d+)\s+artifact_bytes:(\d+)\s+.*?size_ok:(True|False)"
        match = re.search(pattern, eval_results)
        if match:
            val_bpb_str = match.group(1)
            artifact_bytes_str = match.group(2)
            size_ok_str = match.group(3)
            try:
                val_bpb = float(val_bpb_str)
                artifact_bytes = int(artifact_bytes_str)
            except ValueError:
                logger.error(f"Could not parse val_bpb='{val_bpb_str}' or artifact_bytes='{artifact_bytes_str}'.")
                return None

            if size_ok_str != "True" or artifact_bytes > ARTIFACT_SIZE_LIMIT:
                logger.error(
                    f"Artifact size {artifact_bytes} exceeds {ARTIFACT_SIZE_LIMIT} byte limit."
                )
                return None
            return val_bpb

        # Fallback: parse the final_int8_zlib_roundtrip line directly
        fallback_bpb = re.search(
            r"final_int8_zlib_roundtrip_exact\s+val_loss:\S+\s+val_bpb:(\d+\.\d+)",
            eval_results,
        )
        fallback_size = re.search(
            r"Total submission size int8\+zlib:\s*(\d+)\s*bytes",
            eval_results,
        )
        if fallback_bpb and fallback_size:
            try:
                val_bpb = float(fallback_bpb.group(1))
                artifact_bytes = int(fallback_size.group(1))
            except ValueError:
                logger.error("Could not parse fallback val_bpb or artifact_bytes.")
                return None
            if artifact_bytes > ARTIFACT_SIZE_LIMIT:
                logger.error(
                    f"Artifact size {artifact_bytes} exceeds {ARTIFACT_SIZE_LIMIT} byte limit."
                )
                return None
            return val_bpb

        logger.error(f"Pattern not found in the string: '{eval_results[-500:]}'")
        return None

    elif isinstance(eval_results, list):
        parsed = [parse_eval_results(r) for r in eval_results]
        valid = [p for p in parsed if p is not None]
        if len(valid) == 1:
            return valid[0]
        if not valid:
            return None
        return valid

    raise ValueError("Input must be a string or a list of strings.")


def parse_eval_metrics(eval_results: list[str] | str) -> dict[str, float]:
    """Extract richer task metrics from Parameter Golf training logs."""
    text = "\n".join(eval_results) if isinstance(eval_results, list) else eval_results
    metrics: dict[str, float] = {}
    if not text:
        return metrics

    candidate_match = re.search(
        r"Candidate val_bpb:(-?\d+(?:\.\d+)?)\s+artifact_bytes:(\d+)\s+size_limit:(\d+)\s+size_ok:(True|False)\s+train_time_ms:(-?\d+(?:\.\d+)?)",
        text,
    )
    if candidate_match:
        metrics["candidate_val_bpb"] = float(candidate_match.group(1))
        metrics["artifact_bytes"] = float(candidate_match.group(2))
        metrics["size_limit"] = float(candidate_match.group(3))
        metrics["size_ok"] = 1.0 if candidate_match.group(4) == "True" else 0.0
        metrics["train_time_ms"] = float(candidate_match.group(5))

    roundtrip_match = re.search(
        r"final_int8_zlib_roundtrip_exact\s+val_loss:(-?\d+(?:\.\d+)?)\s+val_bpb:(-?\d+(?:\.\d+)?)",
        text,
    )
    if roundtrip_match:
        metrics["roundtrip_val_loss"] = float(roundtrip_match.group(1))
        metrics["roundtrip_val_bpb"] = float(roundtrip_match.group(2))

    peak_mem_match = re.search(
        r"peak memory allocated:\s*(\d+)\s*MiB\s*reserved:\s*(\d+)\s*MiB",
        text,
    )
    if peak_mem_match:
        metrics["peak_memory_allocated_mib"] = float(peak_mem_match.group(1))
        metrics["peak_memory_reserved_mib"] = float(peak_mem_match.group(2))

    serialized_model_match = re.search(r"Serialized model:\s*(\d+)\s*bytes", text)
    if serialized_model_match:
        metrics["serialized_model_bytes"] = float(serialized_model_match.group(1))
    code_size_match = re.search(r"Code size:\s*(\d+)\s*bytes", text)
    if code_size_match:
        metrics["code_size_bytes"] = float(code_size_match.group(1))
    int8_match = re.search(
        r"Serialized model int8\+zlib:\s*(\d+)\s*bytes\s*\(payload:(\d+)\s+raw_torch:(\d+)\s+payload_ratio:(-?\d+(?:\.\d+)?)x\)",
        text,
    )
    if int8_match:
        metrics["int8_zlib_bytes"] = float(int8_match.group(1))
        metrics["int8_payload_bytes"] = float(int8_match.group(2))
        metrics["int8_raw_torch_bytes"] = float(int8_match.group(3))
        metrics["int8_payload_ratio_x"] = float(int8_match.group(4))

    train_logs = re.findall(
        r"step:(\d+)/(\d+)\s+train_loss:(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s+train_time:(-?\d+(?:\.\d+)?)ms\s+step_avg:(-?\d+(?:\.\d+)?)ms",
        text,
    )
    if train_logs:
        metrics["train_log_count"] = float(len(train_logs))
        metrics["first_train_loss"] = float(train_logs[0][2])
        metrics["last_train_loss"] = float(train_logs[-1][2])
        metrics["last_train_step"] = float(train_logs[-1][0])

    val_logs = re.findall(
        r"step:(\d+)/(\d+)\s+val_loss:(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s+val_bpb:(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s+train_time:(-?\d+(?:\.\d+)?)ms\s+step_avg:(-?\d+(?:\.\d+)?)ms",
        text,
    )
    if val_logs:
        metrics["val_log_count"] = float(len(val_logs))
        metrics["first_val_loss"] = float(val_logs[0][2])
        metrics["last_val_loss"] = float(val_logs[-1][2])
        metrics["first_val_bpb"] = float(val_logs[0][3])
        metrics["last_val_bpb"] = float(val_logs[-1][3])
        metrics["best_val_bpb"] = min(float(item[3]) for item in val_logs)
        metrics["last_val_step"] = float(val_logs[-1][0])

    if "stopping_early: wallclock_cap" in text:
        metrics["stopped_early_wallclock"] = 1.0

    return metrics
