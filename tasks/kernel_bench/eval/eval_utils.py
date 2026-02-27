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

import dataclasses
import os
from task_utils import CompletedProcess, _call_shell_command
import logging
import re

logger = logging.getLogger("controller")
@dataclasses.dataclass
class EvalConfig:
  """Evaluation configuration for a single consistent hashing scenario."""
  dataset: str

def recompile_library(config: dict) -> CompletedProcess:
    comp_config = config['compilation']
    EVAL_PATH = os.path.expanduser(config['paths']['eval_path'])
    EVAL_SCRIPT = os.path.join(
      EVAL_PATH, config['evaluation']['eval_script_name']
    )
    baseline_unversioned = config['evaluation']['kernel_name'].split("-")[0]
    BASELINE_PATH = os.path.join(EVAL_PATH, "baseline", baseline_unversioned+".py")
    KERNEL_BASE_PATH = os.path.join(config['paths']['src_path'], "kernels", config['evaluation']['kernel_name'])
    KERNEL_PATH = os.path.join(KERNEL_BASE_PATH, "kernel.py")
    CONDA_PREFIX = f"/opt/conda/bin/conda run -n {config['compilation']['conda_env']} "
    command = (
      f"{CONDA_PREFIX} python {EVAL_SCRIPT} --baseline_path {BASELINE_PATH} --kernel_path {KERNEL_PATH} --baseline_time {config['evaluation']['baseline_time']} --build_dir {KERNEL_BASE_PATH}"
    )
    logger.info(f"recompile_library: Running command: {command}")
    process_result = _call_shell_command(
      command, timeout=comp_config['recompile_timeout'], max_retries=comp_config['recompile_max_retries']
    )

    if not process_result:
      return CompletedProcess(
        args=command,
        returncode=-1,
        stdout="",
        stderr="Compilation command failed to complete.",
      )

    success = (process_result.returncode == 0)
    logger.info(f"recompile_library: Success: {success}.")
    for line in process_result.stdout.splitlines():
      logger.debug(f"recompile_library: STDOUT: {line}")
    for line in process_result.stderr.splitlines():
      logger.debug(f"recompile_library: STDERR: {line}")
    return CompletedProcess(
      args=command,
      returncode=process_result.returncode,
      stdout=process_result.stdout.strip(),
      stderr=process_result.stderr.strip()
    )


def evaluate_dataset(
  candidate_id: int,
  baseline_id: int,
  eval_config: EvalConfig,
  config: dict,
) -> CompletedProcess:
  EVAL_PATH = os.path.expanduser(config['paths']['eval_path'])
  EVAL_SCRIPT = os.path.join(
    EVAL_PATH, config['evaluation']['eval_script_name']
  )
  baseline_unversioned = config['evaluation']['kernel_name'].split("-")[0]
  BASELINE_PATH = os.path.join(EVAL_PATH, "baseline", baseline_unversioned+".py")
  KERNEL_BASE_PATH = os.path.join(config['paths']['src_path'], "kernels", config['evaluation']['kernel_name'])
  KERNEL_PATH = os.path.join(KERNEL_BASE_PATH, "kernel.py")
  CONDA_PREFIX = f"/opt/conda/bin/conda run -n {config['compilation']['conda_env']} "
  eval_command = (
    f"{CONDA_PREFIX} python {EVAL_SCRIPT} --baseline_path {BASELINE_PATH} --kernel_path {KERNEL_PATH} --baseline_time {config['evaluation']['baseline_time']} --build_dir {KERNEL_BASE_PATH}"
  )

  logger.info(f"evaluate_dataset: Running {eval_command}")
  process_result_eval = _call_shell_command(
    eval_command, timeout=config['evaluation']['eval_timeout'], max_retries=config['evaluation']['eval_max_retries']
  )
  if not process_result_eval:
    logger.error(
      f"evaluate_dataset: evaluate_dataset for {eval_config.dataset} failed."
    )
    return CompletedProcess(
      args=eval_command,
      returncode=-1,
      stdout="",
      stderr=f"evaluate_dataset for {eval_config.dataset} failed to complete."
    )

  return process_result_eval


def parse_eval_results(
  eval_results: list[str] | str,
) -> list[float | None] | float | None:
  """
  Parses a string or list of strings to extract kernel speedup.
  """
  if isinstance(eval_results, str):
    metrics = parse_eval_metrics(eval_results)
    if not metrics:
      logger.error(f"Pattern not found in the string: '{eval_results}'")
      return None
    return metrics.get("kernel_speedup")
  
  elif isinstance(eval_results, list):
    parsed_results = []
    for result in eval_results:
      parsed_val = parse_eval_results(result)
      if parsed_val is not None:
          parsed_results.append(parsed_val)
          
    if len(parsed_results) == 1:
        return parsed_results[0]
    elif not parsed_results:
        return None
    else:
        return parsed_results
  
  else:
    raise ValueError("Input must be a string or a list of strings.")


def _extract_single_metric(pattern: str, text: str) -> float | None:
  match = re.search(pattern, text, re.IGNORECASE)
  if not match:
    return None
  try:
    return float(match.group(1))
  except ValueError:
    return None


def parse_eval_metrics(eval_results: list[str] | str) -> dict[str, float]:
  if isinstance(eval_results, str):
    metrics: dict[str, float] = {}
    fields = {
      "kernel_speedup": r"Kernel speedup:\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
      "baseline_time": r"Baseline time:\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
      "kernel_time": r"Kernel time:\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
    }
    for key, pattern in fields.items():
      value = _extract_single_metric(pattern, eval_results)
      if value is not None:
        metrics[key] = value
    return metrics

  if isinstance(eval_results, list):
    if len(eval_results) == 1:
      return parse_eval_metrics(eval_results[0])
    merged: dict[str, float] = {}
    for idx, eval_result in enumerate(eval_results):
      sub_metrics = parse_eval_metrics(eval_result)
      for key, value in sub_metrics.items():
        merged[f"{key}_ds{idx}"] = value
    return merged

  return {}
