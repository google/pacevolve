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
import re
import os
import ast
from task_utils import CompletedProcess, _call_shell_command
import logging


logger = logging.getLogger("controller")
@dataclasses.dataclass
class EvalConfig:
  """Evaluation configuration for a single consistent hashing scenario."""
  dataset: str
  dim: int

def recompile_library(config: dict) -> CompletedProcess:
    comp_config = config['compilation']
    EVAL_PATH = os.path.expanduser(config['paths']['eval_path'])
    TARGET_PATH = os.path.expanduser(config['paths']['src_path'])
    EVAL_SCRIPT = os.path.join(
      EVAL_PATH, config['evaluation']['eval_script_name']
    )
    CANDIDATE_SCRIPT = os.path.join(
      TARGET_PATH, config['paths']['target_file_path']
    )
    CONDA_PREFIX = f"conda run -n {config['compilation']['conda_env']} "
    command = (
      f"{CONDA_PREFIX} python {EVAL_SCRIPT} --candidate_path {CANDIDATE_SCRIPT} --data_path {config['paths']['data_path']} --problem_idx {config['evaluation']['problem_idx']}"
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
  RESULTS_PATH = os.path.expanduser(config['paths']['results_path'])
  TARGET_PATH = os.path.expanduser(config['paths']['src_path'])
  EVAL_SCRIPT = os.path.join(
      EVAL_PATH, config['evaluation']['eval_script_name']
  )
  CANDIDATE_SCRIPT = os.path.join(
    TARGET_PATH, config['paths']['target_file_path']
  )
  # BASELINE_DIFF_SCRIPT = os.path.join(EVAL_PATH, config['evaluation']['baseline_diff_script_name'])
  CONDA_PREFIX = f"conda run -n {config['compilation']['conda_env']} "
  results_dir = os.path.join(RESULTS_PATH, eval_config.dataset)
  output_file = os.path.join(results_dir, f"candidate_{candidate_id}.pickle")

  eval_command = (
    f"{CONDA_PREFIX} python {EVAL_SCRIPT} --candidate_path {CANDIDATE_SCRIPT} --data_path {config['paths']['data_path']} --problem_idx {config['evaluation']['problem_idx']}"
  )

  try:
    os.makedirs(results_dir, exist_ok=True)
  except OSError as e:
    logger.error(
      f"evaluate_dataset: Could not create results directory {results_dir}: {e}"
    )
    return CompletedProcess(
      args=eval_command,
      returncode=-1,
      stdout="",
      stderr=f"Could not create results directory {results_dir}: {e}"
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
) -> list[float] | float:
  if isinstance(eval_results, str):
    parsed_metrics = parse_eval_metrics(eval_results)
    if not parsed_metrics:
      return None
    return parsed_metrics.get("log10_nmse")

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


def _parse_candidate_dict(eval_result: str) -> dict | None:
  if not eval_result:
    return None
  pattern = r"Candidate:\s*({.+?})\s*"
  match = re.search(pattern, eval_result, re.DOTALL)
  if not match:
    return None

  captured_value_str = match.group(1)
  try:
    clean_str = re.sub(r"np\.float64\(([^)]+)\)", r"\1", captured_value_str)
    parsed = ast.literal_eval(clean_str)
    if isinstance(parsed, dict):
      return parsed
  except Exception as e:
    logger.error(f"_parse_candidate_dict: Failed to parse candidate metrics: {e}")
  return None


def parse_eval_metrics(eval_results: list[str] | str) -> dict[str, float]:
  if isinstance(eval_results, str):
    parsed = _parse_candidate_dict(eval_results)
    if not parsed:
      return {}
    metrics = {}
    for key, value in parsed.items():
      try:
        metrics[str(key)] = float(value)
      except Exception:
        continue
    return metrics

  if isinstance(eval_results, list):
    if len(eval_results) == 1:
      return parse_eval_metrics(eval_results[0])
    merged = {}
    for idx, eval_result in enumerate(eval_results):
      sub_metrics = parse_eval_metrics(eval_result)
      for key, value in sub_metrics.items():
        merged[f"{key}_ds{idx}"] = value
    return merged

  return {}
