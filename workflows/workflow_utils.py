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

import copy
import dataclasses
import traceback
import logging
import os
import re
import json

from concurrent.futures import ThreadPoolExecutor, as_completed

import llm_utils
import task_utils, idea_select_utils
import importlib
from copy import deepcopy

logger = logging.getLogger("controller")

Transcript = llm_utils.Transcript
ContentChunk = llm_utils.ContentChunk
CompilationConfig = task_utils.CompilationConfig
CompletedProcess = task_utils.CompletedProcess

NUM_CPU_CORES = 32


@dataclasses.dataclass
class AlgorithmTrial:
  algorithm_implementation: str = ""
  compile_success: bool = False
  eval_success: list[bool] = dataclasses.field(default_factory=list)
  eval_results: list[str] = dataclasses.field(default_factory=list)
  idea_id: int = -1
  compile_attempts: int = 0
  eval_attempts: int = 0
  compile_errors: list[str] = dataclasses.field(default_factory=list)
  eval_failures: list[str] = dataclasses.field(default_factory=list)
  analysis_success: bool = False
  analysis_results: str = ""
  analysis_attempts: int = 0
  analysis_errors: list[str] = dataclasses.field(default_factory=list)
  analysis_metrics: dict[str, float] = dataclasses.field(default_factory=dict)


def _truncate_error_message(message: str, max_chars: int = 1200) -> str:
  if message is None:
    return ""
  if len(message) <= max_chars:
    return message
  return message[: max_chars - 3] + "..."


def _summarize_compile_error(process: CompletedProcess, config: dict) -> str:
  summ_config = config['summarization']
  error_summary_lines = []
  lines = process.stdout.splitlines() + process.stderr.splitlines()
  context_lines_after_error = summ_config['context_lines_after_error']

  indices_with_errors = [i for i, line in enumerate(lines) if "error:" in line.lower()]

  if not indices_with_errors and process.stderr.strip():
    max_lines_fallback = 40
    fallback_summary = "\n".join(lines[-max_lines_fallback:])
    error_description = f"No specific 'error:' lines found. Showing last {max_lines_fallback} lines of output:\n{fallback_summary}"
  elif not indices_with_errors and not process.stderr.strip():
    error_description = "Compilation failed, but no stderr/stdout output was captured or it was empty."
  else:
    line_indices_to_include = set()
    for error_idx in indices_with_errors:
      for i in range(max(0, error_idx - 1), min(len(lines), error_idx + context_lines_after_error + 1)):
        line_indices_to_include.add(i)
    
    sorted_indices = sorted(list(line_indices_to_include))
    max_summary_lines = summ_config['max_summary_lines']
    last_printed_idx = -2
    for idx in sorted_indices:
      if idx > last_printed_idx + 1 and last_printed_idx != -2 :
        if len(error_summary_lines) < max_summary_lines: error_summary_lines.append("[...]")
      if len(error_summary_lines) < max_summary_lines: error_summary_lines.append(lines[idx])
      last_printed_idx = idx
      if len(error_summary_lines) >= max_summary_lines: break
    
    if len(error_summary_lines) >= max_summary_lines and sorted_indices[-1] > last_printed_idx:
      error_summary_lines.append("\n[Error summary truncated due to length...]")
    error_description = "\n".join(error_summary_lines)

    # Log full compile output to DEBUG to avoid flooding INFO, summary still INFO
    logger.debug(f"summarize_compile_error: Full compile output (stdout/stderr combined):\n{process.stderr}")
    logger.info(f"summarize_compile_error: Summarized compile error sent to LLM:\n{error_description}")

  return error_description


def attempt_compile(
  trial: AlgorithmTrial,
  compile_config: CompilationConfig,
  config: dict,
) -> tuple[AlgorithmTrial, str, str | None]:
  try:
    edit_library(
      compile_config.target_file_path,
      algorithm_implementation=trial.algorithm_implementation,
      config=config,
    )
  except ValueError as e:
    error_message = f"INTERNAL ERROR: Failed to edit library: {e}"
    logger.critical(f"attempt_compile: {error_message}")
    return trial, error_message, error_message
  
  task_id = config['experiment']['task_id']
  # Dynamically import task-specific eval_utils
  task_eval_utils = importlib.import_module(f"tasks.{task_id}.eval.eval_utils")

  compile_output = task_eval_utils.recompile_library(config)
  success = (compile_output.returncode == 0)

  output_message = "Code compiled successfully."
  error_description = None
  if not success:
    error_description = _summarize_compile_error(compile_output, config)
    output_message = (
      f"The code provided did not compile. Compiler output (summarized):\n{error_description}\n"
      "Please analyze these errors and provide a corrected version of the code. "
      "Ensure your response contains a complete code block. Focus on fixing the problem."
    )
  command_output = (
    compile_output.stdout.splitlines() + compile_output.stderr.splitlines()
  )
  for line in command_output:
    logger.debug(f"attempt_compile: {line}")
  output_trial = copy.deepcopy(trial)
  output_trial.compile_success = success
  return output_trial, output_message, error_description


def edit_until_compile(
  llm_name,
  trial: AlgorithmTrial,
  transcript: Transcript,
  compile_config: CompilationConfig,
  config: dict,
  loop_config: dict,
  use_idea_repo: bool=False,
) -> AlgorithmTrial:
  max_compile_attempts = loop_config['max_attempts']
  compile_loop_tag = loop_config['loop_tag']
  compile_summary_tag = loop_config['summary_tag']
  num_attempts = 0
  code_was_revised = False
  recovery_prompt = None
  trial = copy.deepcopy(trial)  # Do not modify the original trial object.
  existing_compile_attempts = trial.compile_attempts
  while num_attempts < max_compile_attempts:
    num_attempts += 1
    logger.info(f"edit_until_compile: {num_attempts}/{max_compile_attempts}")
    # If we have a recovery prompt from the last loop, re-prompt the LLM.
    if recovery_prompt:
      code_was_revised = True
      transcript.append(
        ContentChunk(recovery_prompt, "user", tags=[compile_loop_tag])
      )
      recovery_response = llm_utils.generate_completion(llm_name, transcript, config)
      transcript.append(
        ContentChunk(recovery_response, "model", tags=[compile_loop_tag])
      )

    if not transcript or transcript[-1].role != "model":
      # If we do not have a message from the model at the end of the transcript,
      # we need to re-prompt the LLM to provide the C++ code.
      logger.critical(
        "edit_unil_compile: Expected latest message to be from 'model'."
      )
      trial.compile_errors.append("No model response found at end of transcript.")
      recovery_prompt = "Error: No model response found. Please respond."
      continue

    current_llm_response = transcript[-1].content
    logger.debug(f"edit_until_compile: LLM Response:\n{current_llm_response}")
    # print(f"edit_until_compile: LLM Response:\n{current_llm_response}")
    if not current_llm_response:
      logger.warning("edit_until_compile: No response.")
      trial.compile_errors.append("Model returned an empty response.")
      recovery_prompt = (
        "Your output did not contain any markdown-formatted code blocks. "
        "Please provide one."
      )
      continue

    if use_idea_repo:
      idea_id = idea_select_utils.extract_idea_id(current_llm_response)
      if not idea_id:
        logger.warning("edit_until_compile: Idea ID not found in response.")
        trial.compile_errors.append("Missing Idea ID in model response.")
        recovery_prompt = (
          "Your output did not contain Idea ID for the selected idea. "
          "Please provide one."
        )
        continue
      else:
        logger.info(f"edit_until_compile: Setting idea ID to {idea_id}.")
        trial.idea_id = idea_id


    code_blocks = llm_utils.extract_code_blocks(current_llm_response)

    if not code_blocks:
      logger.warning("edit_until_compile: Code blocks not found in response.")
      trial.compile_errors.append("No markdown code block found in model response.")
      recovery_prompt = (
        "Your output did not contain any markdown-formatted code blocks. "
        "Please provide one."
      )
      continue

    trial.algorithm_implementation = code_blocks[0]  # Use the first block.
    trial, recovery_prompt, compile_error_summary = attempt_compile(trial, compile_config, config)
    if compile_error_summary:
      trial.compile_errors.append(_truncate_error_message(compile_error_summary))

    if trial.compile_success:
      logger.info(f"edit_until_compile: Attempt {num_attempts} successful")
      # All of the state is contained within the trial object.
      break

  trial.compile_attempts = existing_compile_attempts + num_attempts

  if trial.compile_success and code_was_revised:
    transcript.log_debug_message(
      "Code had compilation errors/was missing but compiled after revisions."
    )
    summary_text = (
      "Code had errors but compiled after revisions. The code that was finally "
      f"installed:\n```cpp\n{trial.algorithm_implementation}\n```"
    )
    transcript.append(
      ContentChunk(summary_text, "system", tags=[compile_summary_tag])
    )
  elif trial.compile_success and not code_was_revised:
    summary_text = (
      "Code compiled successfully and was installed."
    )
    transcript.log_debug_message("Code compiled on the first attempt.")
    transcript.append(
      ContentChunk(summary_text, "system", tags=[compile_summary_tag])
    )
  else:
    error_message = (
      f"Compilation failed even after {num_attempts} attempts to fix the code."
    )
    logger.warning(error_message)
    transcript.log_debug_message(error_message)
    transcript.append(
      ContentChunk(error_message, "system", tags=[compile_summary_tag])
    )

  return trial

# Map file extensions to their single-line comment markers
COMMENT_MARKERS: dict = {
    'cpp': '//', 'h': '//', 'cc': '//', 'cxx': '//', 'c': '//', 'java': '//',
    'js': '//', 'mjs': '//', 'cjs': '//', 'ts': '//', 'go': '//',
    'py': '#', 'sh': '#', 'bash': '#', 'zsh': '#',
    'rb': '#', 'pl': '#', 'pm': '#', 'r': '#', 'yaml': '#', 'yml': '#',
}

def get_comment_marker_for_file(file_path: str) -> str:
    """Infers the comment marker from the file extension."""
    _, ext = os.path.splitext(file_path)
    extension = ext.lower().lstrip('.')
    marker = COMMENT_MARKERS.get(extension)
    if marker is None:
        raise ValueError(
            f"Unsupported file extension '{extension}' for {file_path}. "
            "Cannot determine comment style."
        )
    return marker

def edit_library(
  target_file_path: str,
  algorithm_implementation: str,
  config: dict,
):
  """
  Edits a target file by replacing content between special comment tags.

  Determines the comment style based on the file extension.
  """
  start_tag: str = config['compilation']['edit_start_tag']
  end_tag: str = config['compilation']['edit_end_tag']

  try:
    comment_marker = get_comment_marker_for_file(target_file_path)
    logger.info(f"Using comment marker '{comment_marker}' for {target_file_path}")
  except ValueError as e:
    logger.critical(f"edit_library: {e}")
    raise

  logger.info(
    "edit_library: Preparing to write to the library file: "
    f"{target_file_path}"
  )
  logger.debug("edit_library: --- Code to be inserted: ---")
  logger.debug(f"CODE: \n{algorithm_implementation}")
  logger.debug("edit_library: --- End of code to be inserted ---")

  with open(target_file_path, 'r') as file:
    content = file.read()

  esc_comment = re.escape(comment_marker)
  esc_start_tag = re.escape(start_tag)
  esc_end_tag = re.escape(end_tag)

  # Regex to find the block:
  # Group 1: The entire start tag line, including the comment marker and any whitespace.
  # Group 2: The content between the start and end tags (what will be replaced).
  # Group 3: The entire end tag line.
  pattern = re.compile(
      f"(^[ \t]*{esc_comment}[ \t]*{esc_start_tag}[ \t]*\n)"  # Group 1: Start tag line
      f"(.*?)"  # Group 2: Content between tags
      f"(^[ \t]*{esc_comment}[ \t]*{esc_end_tag}[ \t]*$)",  # Group 3: End tag line
      re.DOTALL | re.MULTILINE
  )

  if not pattern.search(content):
      logger.critical(
          f"edit_library: Tag block not found for '{comment_marker} {start_tag}' "
          f"and '{comment_marker} {end_tag}' in {target_file_path}."
      )
      raise ValueError(f"Start/end tag block not found in {target_file_path}.")

  def replacer(match):
      return f"{match.group(1)}{algorithm_implementation}\n{match.group(3)}"

  new_content, num_subs = pattern.subn(replacer, content, count=1)

  if num_subs == 0:
      # This case should ideally not be reached if the search above passed.
      logger.critical(
          "edit_library: Tags found but pattern substitution failed for "
          f"'{start_tag}'/'{end_tag}' in {target_file_path}."
      )
      raise ValueError("Tags found, but pattern substitution failed.")

  with open(target_file_path, 'w') as file:
    file.write(new_content)

  logger.info(
    "edit_library: Successfully wrote to the library file: "
    f"{target_file_path}"
  )


def _extract_analysis_metrics(output_text: str) -> dict[str, float]:
  if not output_text:
    return {}
  pattern = re.compile(r"AnalysisMetrics:\s*(\{.*?\})", re.DOTALL)
  match = pattern.search(output_text)
  if not match:
    return {}

  metrics_raw = match.group(1)
  try:
    parsed = json.loads(metrics_raw)
  except Exception:
    return {}

  metrics = {}
  for key, value in parsed.items():
    try:
      metrics[str(key)] = float(value)
    except Exception:
      continue
  return metrics


def _default_pre_eval_analysis_prompt(candidate_code: str) -> str:
  return f"""
You are designing a lightweight pre-evaluation analysis module for an evolutionary workflow.

The candidate implementation to analyze is:
```python
{candidate_code}
```

Your job is to implement an analysis function:
`def analyze_candidate(candidate_source: str) -> dict[str, float]:`

Requirements:
- Return ONLY numeric metrics (floats/ints) in the dict.
- Metrics should be cheap to compute and deterministic.
- Include at least 8 metrics that help reason about complexity/risk.
- Prefix metric names with `analysis_`.
- Do not read files or call network; analyze only `candidate_source`.
- Output valid Python code in a markdown code block.
- The code must define `analyze_candidate` exactly once.
"""


def run_pre_eval_analysis(
  llm_name,
  trial: AlgorithmTrial,
  transcript: Transcript,
  config: dict,
  analysis_prompt: str | None = None,
  max_attempts: int = 2,
) -> AlgorithmTrial:
  """Generates and runs a pre-eval analyzer script for extra metrics."""
  trial = copy.deepcopy(trial)

  src_path = os.path.expanduser(config['paths']['src_path'])
  candidate_path = os.path.join(src_path, config['paths']['target_file_path'])
  harness_path = config['paths'].get(
    'analysis_harness_path',
    os.path.join(os.path.dirname(__file__), "pre_eval_analysis_harness.py")
  )
  harness_path = os.path.abspath(os.path.expanduser(harness_path))

  if not os.path.exists(harness_path):
    message = f"Pre-eval analysis harness not found at {harness_path}. Skipping."
    logger.warning(message)
    trial.analysis_success = False
    trial.analysis_errors.append(message)
    return trial

  loop_tag = "pre_eval_analysis_loop"
  summary_tag = "pre_eval_analysis_summary"

  if not analysis_prompt:
    analysis_prompt = _default_pre_eval_analysis_prompt(trial.algorithm_implementation)

  generated_code = None
  for attempt in range(max_attempts):
    trial.analysis_attempts += 1
    if attempt == 0:
      prompt_text = analysis_prompt
    else:
      prompt_text = (
        "Your previous analysis code was invalid or missing. "
        "Please provide a valid Python code block that defines analyze_candidate(candidate_source: str)."
      )

    transcript.append(ContentChunk(prompt_text, "user", tags=[loop_tag]))
    llm_response_text = llm_utils.generate_completion(llm_name, transcript, config)
    transcript.append(ContentChunk(llm_response_text, "model", tags=[loop_tag]))

    code_blocks = llm_utils.extract_code_blocks(llm_response_text)
    if not code_blocks:
      trial.analysis_errors.append("No code block found in pre-eval analysis response.")
      continue
    generated_code = code_blocks[0]

    if "def analyze_candidate" not in generated_code:
      trial.analysis_errors.append("Generated pre-eval analysis code did not define analyze_candidate.")
      generated_code = None
      continue
    break

  if generated_code:
    patch_config = copy.deepcopy(config)
    if 'compilation' not in patch_config:
      patch_config['compilation'] = {}
    patch_config['compilation']['edit_start_tag'] = "RegexTagPreEvalAnalysisStart"
    patch_config['compilation']['edit_end_tag'] = "RegexTagPreEvalAnalysisEnd"
    try:
      edit_library(
        target_file_path=harness_path,
        algorithm_implementation=generated_code,
        config=patch_config,
      )
    except Exception as e:
      trial.analysis_errors.append(f"Failed to write generated pre-eval analyzer: {e}")

  conda_env = config['compilation'].get('conda_env')
  if conda_env:
    command = f"conda run -n {conda_env} python {harness_path} --candidate_path {candidate_path}"
  else:
    command = f"python {harness_path} --candidate_path {candidate_path}"

  process_result = task_utils._call_shell_command(
    command,
    timeout=config.get('evaluation', {}).get('eval_timeout', 300),
    max_retries=1,
  )

  if not process_result:
    trial.analysis_success = False
    trial.analysis_errors.append("Pre-eval analysis process failed to complete.")
    summary = "Pre-eval analysis failed to complete."
    transcript.append(ContentChunk(summary, "system", tags=[summary_tag]))
    return trial

  result_text = "\n".join([process_result.stdout.strip(), process_result.stderr.strip()]).strip()
  trial.analysis_results = result_text
  trial.analysis_success = (process_result.returncode == 0)
  if not trial.analysis_success:
    trial.analysis_errors.append(_truncate_error_message(result_text, max_chars=800))
  trial.analysis_metrics = _extract_analysis_metrics(result_text)

  if trial.analysis_success:
    metrics_text = json.dumps(trial.analysis_metrics, sort_keys=True)
    transcript.append(
      ContentChunk(
        f"Pre-eval analysis metrics:\n```json\n{metrics_text}\n```",
        "system",
        tags=[summary_tag],
      )
    )
  else:
    transcript.append(
      ContentChunk(
        "Pre-eval analysis failed. Continue with evaluation using available metrics only.",
        "system",
        tags=[summary_tag],
      )
    )
  return trial


def attempt_evals(
  eval_configs: list,
  trial: AlgorithmTrial,
  candidate_id: int,
  baseline_id: int,
  config: dict,
  max_parallel_evals: int = 5,
) -> AlgorithmTrial:
  trial = copy.deepcopy(trial)  # Do not modify the original trial object.
  trial.eval_success = [False for _ in eval_configs]
  trial.eval_results = ["" for _ in eval_configs]
  # num_build_threads = max(1, config['experiment']['num_cpu_cores'] // max_parallel_evals)
  logger.info(
    "attempt_evals: Starting eval for datasets: "
    f"{','.join([config.dataset for config in eval_configs])} "
    f"(Cand ID: {candidate_id})"
  )
  num_workers = min(len(eval_configs), max_parallel_evals)
  task_id = config['experiment']['task_id']
  # Dynamically import task-specific eval_utils
  task_eval_utils = importlib.import_module(f"tasks.{task_id}.eval.eval_utils")
  with ThreadPoolExecutor(max_workers=num_workers) as executor:
    future_to_idx = {
      executor.submit(task_eval_utils.evaluate_dataset, candidate_id, baseline_id, cfg, config): i 
      for i, cfg in enumerate(eval_configs)
    }
    for future in as_completed(future_to_idx):
      idx = future_to_idx[future]
      try:
        completed_process = future.result()
        result_text = "\n".join([
          completed_process.stdout.strip(),
          completed_process.stderr.strip(),
        ])
        trial.eval_success[idx] = (completed_process.returncode == 0)
        trial.eval_results[idx] = result_text
      except Exception as exc:
        trial.eval_success[idx] = False
        trial.eval_results[idx] = (
          f"Eval for {eval_configs[idx].dataset} failed in thread: "
          f"{exc}\n{traceback.format_exc()}"
        )
  return trial


def edit_until_successful_eval(
  llm_name,
  trial: AlgorithmTrial,
  transcript: Transcript,
  compile_config: CompilationConfig,
  eval_configs: list,
  config: dict,
  candidate_id: int,
  baseline_id: int,
  loop_config: dict,
) -> AlgorithmTrial:
  trial = copy.deepcopy(trial)  # Do not modify the original trial object.
  # If the code was not been compiled yet, attempt to do so.
  if not trial.compile_success:
    raise ValueError("Trial must have been compiled before evals.")

  max_eval_attempts = loop_config['max_attempts']
  eval_loop_tag = loop_config['loop_tag']
  eval_summary_tag = loop_config['summary_tag']
  compile_loop_config = {
      'max_attempts': loop_config['max_compile_attempts'],
      'loop_tag': loop_config['compile_loop_tag'],
      'summary_tag': loop_config['compile_summary_tag'],
  }

  code_was_revised = False
  num_attempts = 0
  existing_eval_attempts = trial.eval_attempts
  while num_attempts < max_eval_attempts:
    num_attempts += 1
    logger.info(
      f"edit_until_successful_eval: {num_attempts}/{max_eval_attempts} "
      f"for candidate ID {candidate_id}."
    )

    # Attempt evals, assuming that the current code is installed to the library.
    trial = attempt_evals(
      eval_configs, trial, candidate_id, baseline_id, config, max_parallel_evals=5
    )
    # If all trials were successful, we can break out of the loop; eval is done.
    if all(trial.eval_success):
      logger.info(
        f"edit_until_successful_eval: Attempt {num_attempts} succeeded."
      )
      break
    # If we are out of attempts, we should break and not try to fix the code.
    if num_attempts >= max_eval_attempts:
      break

    # If we reach here, at least one eval failed and we still have attempts left
    # to fix it. We re-prompt the LLM to fix the code and attempt a re-compile.
    logger.info(
      f"edit_until_successful_eval: Eval failed for candidate {candidate_id} "
      f"on attempt {num_attempts}. Requesting LLM fix."
    )
    transcript.append(
      ContentChunk(
        "The code compiled, but evaluation has run-time errors.",
        "user",
        tags=[eval_loop_tag]
      )
    )
    failed_eval_messages = [
      msg for msg, flag in zip(trial.eval_results, trial.eval_success)
      if not flag
    ]
    for msg in failed_eval_messages:
      if len(trial.eval_failures) < 20:
        trial.eval_failures.append(_truncate_error_message(msg))
    transcript.append(
      ContentChunk(
        "\n".join(f"- {m}" for m in failed_eval_messages),
        "system",
        tags=[eval_loop_tag]
      )
    )
    transcript.append(
      ContentChunk(
        "Analyze the issue and provide corrected code in a markdown block.",
        "user",
        tags=[eval_loop_tag]
      )
    )
    # Generate a new code block from the LLM to fix the eval issues.
    llm_fix_response = llm_utils.generate_completion(llm_name, transcript, config)
    transcript.append(
      ContentChunk(llm_fix_response, "model", tags=[eval_loop_tag])
    )
    # Attempt to compile the new code provided by the LLM
    trial = edit_until_compile(
      llm_name, trial, transcript, compile_config, config,
      loop_config=compile_loop_config,
    )
    # Squash any compile edits from the transcript by hiding those chunks.
    transcript.hide_by_tag(tags=[compile_loop_config['loop_tag']])
    code_was_revised = True
    # If compile failed, then we will have the same eval issue and will loop
    # again. This is fine, because it gives the LLM another chance to fix it.

  # If we reach here, we either succeeded in evals or exhausted the attempts.
  final_success = all(trial.eval_success)
  trial.eval_attempts = existing_eval_attempts + num_attempts
  if final_success:
    logger.info(
      f"edit_until_successful_eval: All evals ran for candidate {candidate_id} "
      f"after {num_attempts} attempt(s)."
    )
    if not code_was_revised:
      summary_text = "Evals ran without modifications to the code."
    else:
      summary_text = (
        "Evals ran successfully after code revisions. The final code that "
        f"was evaluated:\n```cpp\n{trial.algorithm_implementation}\n```"
      )
  else:
    logger.error(
      f"edit_until_successful_eval: Failed for candidate {candidate_id} "
      f"after {num_attempts} attempt(s)."
    )
    summary_text = (
      "Evals failed even after multiple attempts to fix the code. "
    )
  transcript.append(
    ContentChunk(summary_text, "system", tags=[eval_summary_tag])
  )
  return trial


def extract_summary(response_text: str) -> list[str]:
  bullets = []
  lines = response_text.strip().split("\n")
  for line in lines:
    line = line.strip()
    if line.startswith(("- ", "* ")) :
      bullets.append(line)
  return bullets


def merge_ideas(
  llm_name: str, 
  transcript_name: str, 
  config: dict,
  new_idea_repo, 
  idea_cap: int,
  max_attempts: int = 3,
) -> None:
  for attempt in range(max_attempts):

    idea_dropping_transcript = Transcript(log_filename=transcript_name)
    idea_dropping_prompt = idea_select_utils.construct_idea_drop_prompt(new_idea_repo, idea_cap)
    idea_dropping_transcript.append(ContentChunk(idea_dropping_prompt, "user", tags=["drop_ideas"]))
    
    ideas_to_drop = llm_utils.generate_completion(llm_name, idea_dropping_transcript, config)

    if not ideas_to_drop:
      logger.error(f"LLM failed to produce ideas to drop on attempt {attempt + 1}. Moving on.")
      continue
    
    ideas_to_drop_list = idea_select_utils.parse_llm_idea_list(ideas_to_drop)
    if not ideas_to_drop_list:
      logger.error(f"Parsed output was not a valid list of idea IDs on attempt {attempt + 1}. Retrying.")
      continue  # Go to the next attempt in the loop
        
    ideas_to_keep = [idea for idea in new_idea_repo.ideas if idea.id not in ideas_to_drop_list]
    new_idea_repo.ideas = ideas_to_keep
    new_idea_repo.reindex_ideas()

    if len(new_idea_repo.ideas) <= idea_cap:
      logger.info(f"Successfully dropped ideas and met cap of {idea_cap} on attempt {attempt + 1}.")
      break  # Goal achieved, exit the loop
    else:
      logger.warning(f"Did not drop enough ideas. Current count: {len(new_idea_repo.ideas)}. Retrying...")

  if len(new_idea_repo.ideas) > idea_cap:
    logger.error(f"Failed to meet idea cap of {idea_cap} after {max_attempts} attempts. Current idea count is {len(new_idea_repo.ideas)}. Proceeding with extra ideas.")
