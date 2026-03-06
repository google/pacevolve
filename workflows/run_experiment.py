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

import sys
import time
import argparse
import logging
import os
import yaml
from copy import deepcopy

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)
# Get the path to the 'workflows' directory
workflows_dir = os.path.dirname(current_script_path)
# Get the path to the 'auto_evo' directory (parent of 'workflows')
project_root = os.path.dirname(workflows_dir)

# Add the project root to the beginning of the sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import llm_utils, workflow_utils, program_database, task_utils, idea_select_utils, analysis_utils
import importlib

# NOTE: LLM interactions are handled in llm_utils.py
# and are configured via the YAML config file.
# Ensure llm_utils.py supports different model backends
# based on the 'llm' section of the config.

AlgorithmTrial = workflow_utils.AlgorithmTrial
Transcript = llm_utils.Transcript
ContentChunk = llm_utils.ContentChunk
# EvalConfig = eval_utils.EvalConfig
CompilationConfig = task_utils.CompilationConfig
ProgramsDatabaseConfig = program_database.ProgramsDatabaseConfig
IdeaRepo = idea_select_utils.IdeaRepo
IdeaRepoDatabase = idea_select_utils.IdeaRepoDatabase

def load_configs(config_path) -> tuple[dict, CompilationConfig, list, str, object]:
  with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

  # Store the absolute path to the config file so it can be passed to other scripts.
  config['config_path'] = os.path.abspath(config_path)

  task_id = config['experiment']['task_id']
  # llm_name is used by llm_utils to determine which model/API to call.
  # The details of the model connection should be within the config['llm'] dict.
  llm_name = config['llm']['name']

  src_path = os.path.expanduser(config['paths']['src_path'])
  compile_config = CompilationConfig(
    target_file_path=os.path.join(src_path, config['paths']['target_file_path']),
    pip_path=None,
  )

  # Dynamically import task-specific EvalConfig
  try:
      task_eval_utils = importlib.import_module(f"tasks.{task_id}.eval.eval_utils")
      EvalConfig = task_eval_utils.EvalConfig
  except (ImportError, AttributeError) as e:
      logger.error(f"Could not load EvalConfig for task '{task_id}': {e}")
      sys.exit(1)

  eval_configs = []
  for eval_config_dict in config['evaluation']['eval_configs']:
      eval_configs.append(EvalConfig(**eval_config_dict))

  # prompts = importlib.import_module(f"tasks.{task_id}.config.prompts")

  return config, compile_config, eval_configs, llm_name


def record_iteration_analysis(
    analysis_manager: analysis_utils.AnalysisManager | None,
    iteration: int,
    island_id: int,
    trial: AlgorithmTrial | None,
    success: bool,
    eval_score: float | None = None,
    summary_bullets: list[str] | None = None,
    failure_reason: str | None = None,
    eval_results: list[str] | None = None,
    elapsed_seconds: float | None = None,
):
  trial = trial if trial is not None else AlgorithmTrial()
  record = analysis_utils.IterationAnalysisRecord(
    iteration=iteration,
    island_id=island_id,
    success=success,
    compile_success=trial.compile_success,
    eval_success=all(trial.eval_success) if trial.eval_success else False,
    compile_attempts=trial.compile_attempts,
    eval_attempts=trial.eval_attempts,
    analysis_attempts=trial.analysis_attempts,
    idea_id=trial.idea_id,
    eval_score=eval_score,
    analysis_success=trial.analysis_success,
    analysis_metrics=trial.analysis_metrics,
    summary_bullets=summary_bullets or [],
    compile_errors=trial.compile_errors[:20],
    eval_failures=trial.eval_failures[:20],
    analysis_errors=trial.analysis_errors[:20],
    eval_results=eval_results or trial.eval_results,
    failure_reason=failure_reason,
    elapsed_seconds=elapsed_seconds,
  )
  analysis_manager.record_iteration(record)

if __name__ == "__main__":
  # Set up logging.
  logger = logging.getLogger("controller")
  logger.setLevel(logging.DEBUG)
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  timestamp = time.strftime("%Y%m%d_%H%M%S")

  parser = argparse.ArgumentParser(description="Run evolutionary process")
  parser.add_argument(
      "--task_id",
      "-t",
      type=str,
      required=True,
      help="Path to the experiment config YAML file.",
  )
  parser.add_argument(
      "--dataset_id",
      "-d",
      type=str,
      required=False,
      default=".",
      help="Path to the experiment config YAML file.",
  )
  parser.add_argument(
      "--run_id",
      "-r",
      type=int,
      required=False,
      default=1,
      help="Path to the experiment config YAML file.",
  )
  parser.add_argument(
    "--use_idea_repo",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="If included, this flag will enable the idea repository."
  )
  parser.add_argument(
    "--summarize_freq",
    type=int,
    required=False,
    default=20,
    help="Frequency to summarize experiment context.",
  )
  parser.add_argument(
    "--idea_selection",
    type=str,
    required=False,
    default="llm",
    choices=['llm'],
    help="idea_selection method."
  )
  parser.add_argument(
    "--idea_cap",
    type=int,
    required=False,
    default=5,
    help="Max number of ideas."
  )
  parser.add_argument(
    "--backtrack_freq",
    type=int,
    required=False,
    default=-1,
    help="How often to perform backtracking."
  )
  parser.add_argument(
    "--backtrack_len",
    type=int,
    required=False,
    default=5,
    help="How long does the agent backtrack."
  )
  parser.add_argument(
    "--power_alpha",
    type=float,
    required=False,
    default=1.5,
    help="Alpha in power law distribution."
  )
  parser.add_argument(
    "--use_idea_filter",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="If included, this flag will enable the idea repository."
  )
  parser.add_argument(
    "--max_attempt",
    type=int,
    required=False,
    default=3,
    help="This argument stores the maximum attempt for each LLM call."
  )
  parser.add_argument(
    "--merge_freq",
    type=int,
    required=False,
    default=1,
    help="How long does the agent backtrack."
  )
  parser.add_argument(
    "--use_integrated_sampling",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable momentum based integrated sampling (default: True)."
  )
  parser.add_argument(
    "--freeze_period",
    type=int,
    required=False,
    default=15,
    help="How many iterations to not trigger crossover sampling again."
  )
  parser.add_argument(
    "--parallel",
    action="store_true",
    default=False,
    help="Enable true async multi-island parallel execution."
  )
  parser.add_argument(
    "--num_workers",
    type=int,
    required=False,
    default=8,
    help="Number of parallel worker processes (used with --parallel)."
  )
  parser.add_argument(
    "--disable_analysis",
    action="store_true",
    default=False,
    help="Disable the analysis module for ablation studies."
  )

  args = parser.parse_args()
  analysis_enabled = not args.disable_analysis

  # Load configurations
  CONFIG_PATH = os.path.abspath(f"../tasks/{args.task_id}/config/{args.dataset_id}/config_{args.run_id}.yaml")
  config, compile_config, eval_configs, llm_name = load_configs(CONFIG_PATH)

  logfile_dir = os.path.expanduser(config['paths']['log_dir'])
  logfile_path = os.path.join(logfile_dir, f"controller_verbose_{timestamp}.log")
  os.makedirs(logfile_dir, exist_ok=True)
  log_file_handler = logging.FileHandler(logfile_path)
  log_file_handler.setLevel(logging.DEBUG)
  log_file_handler.setFormatter(formatter)
  logger.addHandler(log_file_handler)
  # Set up transcript logging.
  transcript_dir = os.path.expanduser(config['paths']['transcript_dir'])
  os.makedirs(transcript_dir, exist_ok=True)
  transcript_file = os.path.join(transcript_dir, f"transcript_{timestamp}.txt")
  print("Transcript will be written to: ", transcript_file)

  # Main experiment loop.
  max_iters = config['experiment']['max_iters']
  max_hparam_iters = config['experiment']['max_hparam_iters']

  num_islands = config['database']['num_islands']
  metric_dir = config['evaluation']['metric_direction']
  idea_repo_db = IdeaRepoDatabase(num_islands=num_islands, target_score=config['evaluation']['target_score'], metric_direction=metric_dir)
  ablation_list = []

  baseline_id = config['experiment']['initial_baseline_id']
  task_id = config['experiment']['task_id']
  task_eval_utils = importlib.import_module(f"tasks.{task_id}.eval.eval_utils")
  # Dynamically import task-specific prompts
  prompt_filename = config['experiment'].get('prompts_file', 'prompts')
  if args.dataset_id == ".":
    prompts = importlib.import_module(f"tasks.{task_id}.config.{prompt_filename}")
  else:
    prompts = importlib.import_module(f"tasks.{task_id}.config.{args.dataset_id}.{prompt_filename}")
  sota_algo = getattr(prompts, config['experiment']['sota_algo_name'])

  last_crossover_idx = [0 for _ in range(num_islands)]
  per_island_count = [0 for _ in range(num_islands)]

  programs_db_config = ProgramsDatabaseConfig(
    num_islands=num_islands,
    tournament_size=config['database']['tournament_size'],
    top_k=config['database']['top_k'],
    max_queue_size=config['database']['max_queue_size'],
  )

  db = program_database.ProgramsDatabase(
    config=programs_db_config,
    template=sota_algo,
    function_to_evolve=task_id,
    metric_direction=metric_dir,
  )

  logger.info("Registering initial scores and solutions.")
  init_score = config['evaluation']['init_score']
  for temp_id in range(num_islands):
    db.register_program(program=sota_algo, island_id=temp_id, score=init_score)
    idea_repo_db.best_scores_history[temp_id].append(init_score)
    idea_repo_db.scheduler.update_score(temp_id, init_score)
    initial_repo = IdeaRepo()
    initial_repo.sota = sota_algo
    idea_repo_db.idea_repos[temp_id].append(initial_repo)

  analysis_manager = None
  analysis_jsonl_path = None
  analysis_report_path = None
  if analysis_enabled:
    analysis_cfg = config.get("analysis", {})
    analysis_dir = os.path.expanduser(
      config['paths'].get('analysis_dir', os.path.join(logfile_dir, "analysis"))
    )
    os.makedirs(analysis_dir, exist_ok=True)
    analysis_jsonl_path = os.path.join(analysis_dir, f"iteration_analysis_{timestamp}.jsonl")
    analysis_report_path = os.path.join(analysis_dir, f"postmortem_{timestamp}.md")
    analysis_manager = analysis_utils.AnalysisManager(
      metric_direction=metric_dir,
      jsonl_path=analysis_jsonl_path,
      report_path=analysis_report_path,
      task_eval_utils=task_eval_utils,
      history_window=analysis_cfg.get("history_window", 60),
      max_context_chars=analysis_cfg.get("max_context_chars", 2400),
      recent_analysis_window=analysis_cfg.get("recent_analysis_window", 3),
    )
    logger.info(f"Iteration analysis will be written to: {analysis_jsonl_path}")
    logger.info(f"Post-mortem report will be written to: {analysis_report_path}")
  else:
    logger.info("Analysis module disabled for this run.")

  logger.info(f"Backtrack frequency is {args.backtrack_freq}, Back track length is {args.backtrack_len}, alpha for power law is {args.power_alpha}")

  # --- Parallel mode dispatch ---
  if args.parallel:
    import parallel_runner
    import asyncio
    logger.info(f"Running in PARALLEL mode with {args.num_workers} workers")
    asyncio.run(parallel_runner.run_parallel_evolution(
        config=config,
        db=db,
        idea_repo_db=idea_repo_db,
        prompts_module=prompts,
        args=args,
        transcript_file=transcript_file,
        project_root=project_root,
        workflows_dir=workflows_dir,
        num_workers=args.num_workers,
        analysis_manager=analysis_manager,
        enable_analysis=analysis_enabled,
    ))
    logger.info("Parallel evolution finished.")
    if analysis_enabled:
      logger.info(f"Iteration analysis log: {analysis_jsonl_path}")
      logger.info(f"Post-mortem report: {analysis_report_path}")
    sys.exit(0)

  # --- Sequential mode (original) ---
  repo_idx_before_backtrack = 0
  backtrack_triggered_idx = -1
  island_id = 0
  for i in range(max_iters):
    iter_start_time = time.time()
    last_bt_iter = False
    logger.info(f"\n{'='*40} Iteration {i} {'='*40}")

    transcript = Transcript(log_filename=transcript_file)
    transcript.log_debug_message(f"### Starting iteration {i}")
    trial = AlgorithmTrial()

    trigger_backtrack = False

    if backtrack_triggered_idx > -1 and i - backtrack_triggered_idx < args.backtrack_len:
      logger.info(f"Backtrack ongoing")
      if i - backtrack_triggered_idx == args.backtrack_len - 1:
        last_bt_iter = True
        logger.info(f"Last backtrack iteration")
      sota_algo = idea_repo_db.idea_repos[island_id][-1].sota
      new_idea_repo = deepcopy(idea_repo_db.idea_repos[island_id][-1])
    else:
      sota_algo, island_id = db.get_candidate()
      if len(idea_repo_db.idea_repos[island_id]) == 0:
        new_idea_repo = IdeaRepo()
      else:
        new_idea_repo = deepcopy(idea_repo_db.idea_repos[island_id][-1])
      if args.use_integrated_sampling and per_island_count[island_id] - last_crossover_idx[island_id] > args.freeze_period:
        logger.info(f"Passed freeze period, momentum on island {island_id} is {idea_repo_db.scheduler.momentums[island_id]}")
        if idea_repo_db.scheduler.check_trigger(island_id):
          logger.info(f"Momentum below threshold")
          (action, cross_id) = idea_repo_db.scheduler.sample_action(island_id)
          logger.info(f"Action is {action}, cross_id is {cross_id}")
          if action == "CROSSOVER":
            last_crossover_idx[island_id] = per_island_count[island_id]
            best_idea_repo = idea_repo_db.get_best_idea_repo(cross_id)
            new_idea_repo.ideas.extend(best_idea_repo.ideas)
            new_idea_repo.reindex_ideas()
          elif action == "BACKTRACK":
            last_crossover_idx[island_id] = per_island_count[island_id]
            trigger_backtrack = True

      if (args.backtrack_freq != -1 and (i+1) % args.backtrack_freq == 0) or trigger_backtrack:
        # assert db._exp_hist == db._sota_hist
        backtrack_triggered_idx = i
        logger.info(f"Entering backtrack mode, selected island {island_id}")
        assert len(idea_repo_db.idea_repos[island_id]) != 0
        repo_idx_before_backtrack = len(idea_repo_db.idea_repos[island_id]) - 1
        sampled_idx = idea_select_utils.sample_power_law(len(idea_repo_db.idea_repos[island_id]), alpha=args.power_alpha)
        sota_algo = idea_repo_db.idea_repos[island_id][sampled_idx].sota
        new_idea_repo = deepcopy(idea_repo_db.idea_repos[island_id][sampled_idx])
    # elif args.backtrack_freq != -1 and (i+1) % args.backtrack_freq < args.backtrack_len and i >= args.backtrack_freq:

    new_idea_repo.sota = sota_algo
    if analysis_manager is not None:
      analysis_context = analysis_manager.build_reasoning_context(island_id)
      if analysis_context:
        transcript.append(ContentChunk(analysis_context, "system", tags=["analysis_context"]))

    per_island_count[island_id] += 1
    trigger_merge = False
    if args.idea_cap > -1 and len(new_idea_repo.ideas) > args.idea_cap and args.merge_freq > -1:
      if args.use_integrated_sampling and per_island_count[island_id] - last_crossover_idx[island_id] == args.freeze_period:
        trigger_merge = True
      elif args.backtrack_freq == -1 or (i+1) % args.backtrack_freq == args.backtrack_freq - 1:
        trigger_merge = True
    logger.info(f"Trigger merge is {trigger_merge} for island {island_id}.")

    if args.use_idea_repo:
      idea_gen_prompt_text = prompts.construct_idea_gen_prompt(sota_algo, new_idea_repo)
      new_hypo = idea_select_utils.scratch_pad(new_idea_repo, llm_name, transcript, config, idea_gen_prompt_text)
      if not new_hypo:
        logger.error(f"Iter {i} failed to generate new hypothesis. Skipping to next iteration.")
        trial.eval_failures.append("Failed to generate new hypothesis from scratch-pad stage.")
        if last_bt_iter and args.merge_freq > -1:
          logger.info(f"End of sequence, merge backtrack results and main results")
          new_idea_repo.ideas.extend(idea_repo_db.idea_repos[island_id][repo_idx_before_backtrack].ideas)
          new_idea_repo.reindex_ideas()
        if trigger_merge:
          workflow_utils.merge_ideas(llm_name, transcript_file, config, new_idea_repo, args.idea_cap)
        record_iteration_analysis(
          analysis_manager=analysis_manager,
          iteration=i,
          island_id=island_id,
          trial=trial,
          success=False,
          failure_reason="idea_generation_failed",
          elapsed_seconds=time.time() - iter_start_time,
        )
        continue

      # This is step 2: Idea selection.
      if args.idea_selection == "llm":
        if args.use_idea_filter:
          for gen_hypo in range(args.max_attempt):
            try:
              prompt_text = prompts.construct_idea_select_no_code_prompt(sota_algo, new_idea_repo)
              transcript.append(ContentChunk(prompt_text, "user", tags=[f"idea_selection_prompt_no_code_{gen_hypo}"]))
              llm_response_text = llm_utils.generate_completion(llm_name, transcript, config)
              transcript.append(ContentChunk(llm_response_text, "model", tags=[f"initial_response_no_code_{gen_hypo}"]))
              if llm_response_text is None:
                logger.error(f"LLM did not return anything in hypothesis gen in iter {i}, skipping to next iteration")
                transcript.hide_by_tag(tags=[f"idea_selection_prompt_no_code_{gen_hypo}", f"initial_response_no_code_{gen_hypo}"])
                continue
              else:
                idea_id, exp_description = idea_select_utils.parse_selected_idea(llm_response_text)
                if idea_id is None or exp_description is None:
                  logger.error(f"LLM failed to output correctly formatted ideas")
                  transcript.hide_by_tag(tags=[f"idea_selection_prompt_no_code_{gen_hypo}", f"initial_response_no_code_{gen_hypo}"])
                  continue
              idea_filtering_transcript = Transcript(log_filename=transcript_file)
              idea_filtering_prompt = idea_select_utils.construct_idea_repetition_detection_prompt(ablation_list, llm_response_text)
              idea_filtering_transcript.append(ContentChunk(idea_filtering_prompt, "user", tags=["filter_ideas"]))
              llm_repetition_detection_response = llm_utils.generate_completion(llm_name, idea_filtering_transcript, config)
              is_repetitive = idea_select_utils.parse_repetition_detection(llm_repetition_detection_response)
              if is_repetitive is None:
                logger.error(f"LLM response failed to produce answers in the correct format, skipping to next iteration")
                continue
              elif is_repetitive:
                transcript.hide_by_tag(tags=[f"idea_selection_prompt_no_code_{gen_hypo}", f"initial_response_no_code_{gen_hypo}"])
                continue
              break
            except:
              logger.error(f"Unexpected error in iter {i} attempt {gen_hypo+1} when detecting repetitive hypothesis")
              continue
        else:
          prompt_text = prompts.construct_idea_select_prompt(sota_algo, new_idea_repo)
          transcript.append(ContentChunk(prompt_text, "user", tags=["idea_selection_prompt"]))
          llm_response_text = llm_utils.generate_completion(llm_name, transcript, config)
          transcript.append(ContentChunk(llm_response_text, "model", tags=["initial_response"]))
    else:
      prompt_text = prompts.construct_mutation_prompt(sota_algo, ablation_list)
      transcript.append(ContentChunk(prompt_text, "user", tags=["initial_prompt"]))
      llm_response_text = llm_utils.generate_completion(llm_name, transcript, config)
      transcript.append(ContentChunk(llm_response_text, "model", tags=["initial_response"]))
    # Extract code blocks from the LLM response and compile the code.
    trial = workflow_utils.edit_until_compile(
      llm_name, trial, transcript, compile_config, config,
      loop_config=config['workflow_loops']['initial_compile'],
      use_idea_repo=args.use_idea_repo,
    )
    # Squash any compile edits from the transcript by hiding those chunks.
    transcript.hide_by_tag(tags=["initial_compile_loop"])
    if not trial.compile_success:
      logger.error(f"Iter {i}: Candidate failed eval. Skipping to next iteration.")
      if last_bt_iter and args.merge_freq > -1:
        logger.info(f"End of sequence, merge backtrack results and main results")
        new_idea_repo.ideas.extend(idea_repo_db.idea_repos[island_id][repo_idx_before_backtrack].ideas)
        new_idea_repo.reindex_ideas()
      if trigger_merge:
        workflow_utils.merge_ideas(llm_name, transcript_file, config, new_idea_repo, args.idea_cap)
      record_iteration_analysis(
        analysis_manager=analysis_manager,
        iteration=i,
        island_id=island_id,
        trial=trial,
        success=False,
        failure_reason="compile_failed",
        elapsed_seconds=time.time() - iter_start_time,
      )
      continue

    if analysis_enabled:
      pre_eval_prompt = workflow_utils.resolve_pre_eval_analysis_prompt(
        prompts, trial, transcript
      )
      trial = workflow_utils.run_pre_eval_analysis(
        llm_name=llm_name,
        trial=trial,
        transcript=transcript,
        config=config,
        analysis_prompt=pre_eval_prompt,
        max_attempts=max(1, min(args.max_attempt, 3)),
      )
      transcript.hide_by_tag(tags=["pre_eval_analysis_loop"])
      if not trial.analysis_success:
        logger.warning(f"Iter {i}: Pre-eval analysis did not complete successfully. Proceeding to eval.")

    # Run the evaluation process.
    trial = workflow_utils.edit_until_successful_eval(
      llm_name, trial, transcript, compile_config, eval_configs, config,
      i+1, baseline_id,
      loop_config=config['workflow_loops']['initial_eval'],
    )
    transcript.hide_by_tag(tags=["initial_eval_loop"])
    if not all(trial.eval_success):
      logger.error(f"Iter {i}: Candidate failed eval. Skipping to next iteration.")
      if last_bt_iter and args.merge_freq > -1:
        logger.info(f"End of sequence, merge backtrack results and main results")
        new_idea_repo.ideas.extend(idea_repo_db.idea_repos[island_id][repo_idx_before_backtrack].ideas)
        new_idea_repo.reindex_ideas()
      if trigger_merge:
        workflow_utils.merge_ideas(llm_name, transcript_file, config, new_idea_repo, args.idea_cap)
      record_iteration_analysis(
        analysis_manager=analysis_manager,
        iteration=i,
        island_id=island_id,
        trial=trial,
        success=False,
        failure_reason="eval_failed",
        elapsed_seconds=time.time() - iter_start_time,
      )
      continue
    # Log the initial eval results to the transcript.
    transcript.append(ContentChunk(prompts.EVAL_DESCRIPTION_PROMPT,"user", tags=["initial_eval_results"]))
    eval_results = "\n".join(["```"] + trial.eval_results + ["```"])
    transcript.append(ContentChunk(eval_results, "system", tags=["initial_eval_results"]))

    eval_score = task_eval_utils.parse_eval_results(trial.eval_results)
    logger.debug(f"My eval score is {eval_score}")
    if eval_score is None:
      logger.error(f"Program score is None, skipping program registration.")
    else:
      db.register_program(program=trial.algorithm_implementation, island_id=island_id, score=eval_score)
      idea_repo_db.best_scores_history[island_id].append(eval_score)
      idea_repo_db.scheduler.update_score(island_id, eval_score)

    # Summarize the experiment status.
    summary_prompt = prompts.SUMMARIZE_EVAL_PROMPT
    transcript.append(ContentChunk(summary_prompt, "user", tags=["final_summary_request"]))
    llm_summary = None
    bullets = []
    for idx in range(args.max_attempt):
      llm_summary = llm_utils.generate_completion(llm_name, transcript, config)
      if not llm_summary:
        logger.error(f"LLM failed to produce idea summary on attempt {idx + 1}. Try again.")
      else:
        transcript.append(ContentChunk(llm_summary, "model", tags=["final_summary_response"]))
        break

    try:
      if llm_summary:
        bullets = workflow_utils.extract_summary(llm_summary)
      ablation_list.extend(bullets)
      if args.use_idea_repo:
        if trial.idea_id == -1:
          logger.critical("Unexpected idea ID, should be handled already")
        else:
          idea = new_idea_repo.find_idea_by_id(trial.idea_id)
          if not idea:
            logger.error("LLM hallucinated a non-existing ID")
          else:
            idea.exp_history.extend(bullets)
            idea.exp_count += 1
            if idea.exp_count % args.summarize_freq == 0:
              idea_select_utils.summarize(idea, llm_name, config, Transcript(log_filename=transcript_file), Transcript(log_filename=transcript_file))
    except:
      logger.error(f"LLM failed to summarize ideas on attempt.")

    if last_bt_iter and args.merge_freq > -1:
      logger.info(f"End of sequence, merge backtrack results and main results")
      new_idea_repo.ideas.extend(idea_repo_db.idea_repos[island_id][repo_idx_before_backtrack].ideas)
      new_idea_repo.reindex_ideas()
    if trigger_merge:
      workflow_utils.merge_ideas(llm_name, transcript_file, config, new_idea_repo, args.idea_cap)

    idea_repo_db.idea_repos[island_id].append(new_idea_repo)
    # idea_repo = new_idea_repo

    failure_reason = None if eval_score is not None else "score_parse_failed"
    record_iteration_analysis(
      analysis_manager=analysis_manager,
      iteration=i,
      island_id=island_id,
      trial=trial,
      success=True,
      eval_score=eval_score,
      summary_bullets=bullets,
      failure_reason=failure_reason,
      eval_results=trial.eval_results,
      elapsed_seconds=time.time() - iter_start_time,
    )

    logger.info(f"Iter {i} summary:\n" + "\n".join(bullets))

    logger.info(f"{'='*40} Iteration {i} finished {'='*40}\n")

  logger.info(f"All {max_iters} iterations finished.")
  logger.info(f"LLM Transcript log: {transcript_file}")
  if analysis_enabled:
    logger.info(f"Iteration analysis log: {analysis_jsonl_path}")
    logger.info(f"Post-mortem report: {analysis_report_path}")
