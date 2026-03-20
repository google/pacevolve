"""Async parallel multi-island evolution runner for PACEvolve.

Uses concurrent.futures.ProcessPoolExecutor to run island iterations in
parallel. Each worker process independently runs the full pipeline:
  candidate selection -> LLM call -> compile -> eval -> return result

The main process coordinates database updates and crossover scheduling.
"""

import asyncio
import dataclasses
import importlib
import logging
import os
import sys
import time
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, Future
from copy import deepcopy
from typing import Optional
import analysis_utils
import record_utils

logger = logging.getLogger("controller")

# ---------------------------------------------------------------------------
# Worker-side globals (initialised per process via _worker_init)
# ---------------------------------------------------------------------------
_worker_config = None
_worker_compile_config = None
_worker_eval_configs = None
_worker_llm_name = None
_worker_prompts = None
_worker_task_eval_utils = None


@dataclasses.dataclass
class IterationResult:
    """Serialisable result returned from a worker process."""
    iteration: int
    island_id: int
    program_code: Optional[str] = None
    eval_score: Optional[float] = None
    eval_results: list = dataclasses.field(default_factory=list)
    summary_bullets: list = dataclasses.field(default_factory=list)
    idea_id: int = -1
    updated_idea_repo: Optional[object] = None  # IdeaRepo for appending to idea_repo_db
    success: bool = False
    error: Optional[str] = None
    elapsed: float = 0.0
    compile_success: bool = False
    eval_success: bool = False
    compile_attempts: int = 0
    eval_attempts: int = 0
    compile_errors: list = dataclasses.field(default_factory=list)
    eval_failures: list = dataclasses.field(default_factory=list)
    analysis_success: bool = False
    analysis_attempts: int = 0
    analysis_metrics: dict = dataclasses.field(default_factory=dict)
    analysis_errors: list = dataclasses.field(default_factory=list)
    analysis_mode: str = "disabled"
    analysis_results: str = ""
    analysis_script: str = ""
    cuda_visible_devices: Optional[str] = None


def _worker_init(config_dict: dict, project_root: str, workflows_dir: str):
    """Initialise heavy, non-picklable objects once per worker process."""
    global _worker_config, _worker_compile_config, _worker_eval_configs
    global _worker_llm_name, _worker_prompts, _worker_task_eval_utils

    if workflows_dir and workflows_dir not in sys.path:
        sys.path.insert(0, workflows_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import task_utils

    _worker_config = config_dict
    _worker_llm_name = config_dict['llm']['name']

    src_path = os.path.expanduser(config_dict['paths']['src_path'])
    _worker_compile_config = task_utils.CompilationConfig(
        target_file_path=os.path.join(src_path, config_dict['paths']['target_file_path']),
        pip_path=None,
    )

    task_id = config_dict['experiment']['task_id']
    _worker_task_eval_utils = importlib.import_module(f"tasks.{task_id}.eval.eval_utils")
    EvalConfig = _worker_task_eval_utils.EvalConfig
    _worker_eval_configs = [EvalConfig(**d) for d in config_dict['evaluation']['eval_configs']]

    prompt_filename = config_dict['experiment'].get('prompts_file', 'prompts')
    dataset_id = config_dict.get('_dataset_id', '.')
    if dataset_id == ".":
        _worker_prompts = importlib.import_module(f"tasks.{task_id}.config.{prompt_filename}")
    else:
        _worker_prompts = importlib.import_module(f"tasks.{task_id}.config.{dataset_id}.{prompt_filename}")


def _run_island_iteration(
    iteration: int,
    island_id: int,
    parent_code: str,
    idea_repo_snapshot,
    use_idea_repo: bool,
    use_idea_filter: bool,
    max_attempt: int,
    baseline_id: int,
    transcript_file: str,
    enable_analysis: bool = True,
    analysis_context: Optional[str] = None,
) -> IterationResult:
    """Worker function executed in a child process.

    Runs the full pipeline for one iteration on a given island:
    idea generation -> LLM code generation -> compile -> eval -> summarise.
    """
    t0 = time.time()
    result = IterationResult(iteration=iteration, island_id=island_id)
    worker_temp_dir = None

    try:
        import llm_utils, workflow_utils
        import task_utils

        Transcript = llm_utils.Transcript
        ContentChunk = llm_utils.ContentChunk
        AlgorithmTrial = workflow_utils.AlgorithmTrial

        config = deepcopy(_worker_config)
        llm_name = _worker_llm_name
        prompts = _worker_prompts
        eval_configs = _worker_eval_configs

        src_path = os.path.expanduser(config["paths"]["src_path"])
        worker_temp_dir = tempfile.mkdtemp(
            prefix=f"pacevolve_worker_{os.getpid()}_{iteration}_{island_id}_",
            dir="/tmp",
        )
        worker_src_path = os.path.join(worker_temp_dir, "src")
        shutil.copytree(src_path, worker_src_path)
        config["paths"]["src_path"] = worker_src_path
        compile_config = task_utils.CompilationConfig(
            target_file_path=os.path.join(
                worker_src_path,
                config["paths"]["target_file_path"],
            ),
            pip_path=_worker_compile_config.pip_path,
        )

        island_cuda_visible_devices = task_utils.resolve_cuda_visible_devices_for_island(
            config, island_id
        )
        if island_cuda_visible_devices is not None:
            config.setdefault("evaluation", {})
            config["evaluation"]["cuda_visible_devices"] = island_cuda_visible_devices
            result.cuda_visible_devices = island_cuda_visible_devices

        transcript = Transcript(log_filename=transcript_file)
        transcript.log_debug_message(f"### Starting parallel iteration {iteration} on island {island_id}")
        transcript.log_debug_message(
            f"### Worker-local source tree: {worker_src_path}"
        )
        if island_cuda_visible_devices is not None:
            transcript.log_debug_message(
                f"### Island {island_id} assigned CUDA_VISIBLE_DEVICES={island_cuda_visible_devices}"
            )
        if analysis_context:
            transcript.append(ContentChunk(analysis_context, "system", tags=["analysis_context"]))
        trial = AlgorithmTrial()

        def _persist_iteration_records(failure_reason: Optional[str]) -> None:
            records_run_dir = config.get("paths", {}).get("records_run_dir")
            if not records_run_dir:
                return
            try:
                record_utils.write_iteration_records(
                    records_run_dir=records_run_dir,
                    iteration=iteration,
                    island_id=island_id,
                    transcript=transcript,
                    candidate_code=trial.algorithm_implementation,
                    eval_results=trial.eval_results,
                    task_eval_utils=_worker_task_eval_utils,
                    success=result.success,
                    compile_success=trial.compile_success,
                    eval_success=all(trial.eval_success) if trial.eval_success else False,
                    compile_attempts=trial.compile_attempts,
                    eval_attempts=trial.eval_attempts,
                    analysis_attempts=trial.analysis_attempts,
                    idea_id=trial.idea_id,
                    eval_score=result.eval_score,
                    summary_bullets=result.summary_bullets,
                    compile_errors=trial.compile_errors,
                    eval_failures=trial.eval_failures,
                    analysis_errors=trial.analysis_errors,
                    analysis_success=trial.analysis_success,
                    analysis_metrics=trial.analysis_metrics,
                    failure_reason=failure_reason,
                    elapsed_seconds=result.elapsed,
                    cuda_visible_devices=result.cuda_visible_devices,
                    analysis_mode=result.analysis_mode,
                    analysis_results=result.analysis_results,
                    analysis_script=result.analysis_script,
                )
            except Exception as exc:
                logger.warning(
                    f"Failed to persist records for iteration {iteration}, island {island_id}: {exc}"
                )

        sota_algo = parent_code
        new_idea_repo = deepcopy(idea_repo_snapshot) if idea_repo_snapshot else None

        if use_idea_repo and new_idea_repo is not None:
            new_idea_repo.sota = sota_algo
            import idea_select_utils
            idea_gen_prompt_text = prompts.construct_idea_gen_prompt(sota_algo, new_idea_repo)
            new_hypo = idea_select_utils.scratch_pad(new_idea_repo, llm_name, transcript, config, idea_gen_prompt_text)
            if not new_hypo:
                result.error = "Failed to generate new hypothesis"
                result.eval_failures = ["Failed to generate new hypothesis from scratch-pad stage."]
                result.elapsed = time.time() - t0
                result.updated_idea_repo = new_idea_repo
                _persist_iteration_records("idea_generation_failed")
                return result

            if use_idea_filter:
                for gen_hypo in range(max_attempt):
                    try:
                        prompt_text = prompts.construct_idea_select_no_code_prompt(sota_algo, new_idea_repo)
                        transcript.append(ContentChunk(prompt_text, "user", tags=[f"idea_selection_prompt_no_code_{gen_hypo}"]))
                        llm_response_text = llm_utils.generate_completion(llm_name, transcript, config)
                        transcript.append(ContentChunk(llm_response_text, "model", tags=[f"initial_response_no_code_{gen_hypo}"]))
                        if llm_response_text is None:
                            transcript.hide_by_tag(tags=[f"idea_selection_prompt_no_code_{gen_hypo}", f"initial_response_no_code_{gen_hypo}"])
                            continue
                        else:
                            idea_id, _ = idea_select_utils.parse_selected_idea(llm_response_text)
                            if idea_id is None:
                                transcript.hide_by_tag(tags=[f"idea_selection_prompt_no_code_{gen_hypo}", f"initial_response_no_code_{gen_hypo}"])
                                continue
                        break
                    except Exception:
                        continue
            else:
                prompt_text = prompts.construct_idea_select_prompt(sota_algo, new_idea_repo)
                transcript.append(ContentChunk(prompt_text, "user", tags=["idea_selection_prompt"]))
                llm_response_text = llm_utils.generate_completion(llm_name, transcript, config)
                transcript.append(ContentChunk(llm_response_text, "model", tags=["initial_response"]))
        else:
            prompt_text = prompts.construct_mutation_prompt(sota_algo, [])
            transcript.append(ContentChunk(prompt_text, "user", tags=["initial_prompt"]))
            llm_response_text = llm_utils.generate_completion(llm_name, transcript, config)
            transcript.append(ContentChunk(llm_response_text, "model", tags=["initial_response"]))

        # Compile
        trial = workflow_utils.edit_until_compile(
            llm_name, trial, transcript, compile_config, config,
            loop_config=config['workflow_loops']['initial_compile'],
            use_idea_repo=use_idea_repo,
        )
        transcript.hide_by_tag(tags=["initial_compile_loop"])
        if not trial.compile_success:
            result.error = "Compilation failed"
            result.compile_success = False
            result.eval_success = False
            result.compile_attempts = trial.compile_attempts
            result.compile_errors = trial.compile_errors
            result.eval_attempts = trial.eval_attempts
            result.eval_failures = trial.eval_failures
            result.elapsed = time.time() - t0
            result.updated_idea_repo = new_idea_repo if use_idea_repo else None
            _persist_iteration_records("compile_failed")
            return result
        result.compile_success = True

        # Eval
        trial = workflow_utils.edit_until_successful_eval(
            llm_name, trial, transcript, compile_config, eval_configs, config,
            iteration + 1, baseline_id,
            loop_config=config['workflow_loops']['initial_eval'],
        )
        transcript.hide_by_tag(tags=["initial_eval_loop"])
        if enable_analysis:
            analysis_prompt = workflow_utils.resolve_post_eval_analysis_prompt(
                prompts,
                trial,
                transcript,
                config,
            )

            analysis_config = config
            worker_harness_path = None
            base_harness_path = config['paths'].get(
                'analysis_harness_path',
                os.path.join(os.path.dirname(workflow_utils.__file__), "post_eval_analysis_harness.py")
            )
            if os.path.exists(base_harness_path):
                worker_harness_path = os.path.join(
                    "/tmp",
                    f"post_eval_analysis_harness_{os.getpid()}_{iteration}_{island_id}.py",
                )
                try:
                    shutil.copyfile(base_harness_path, worker_harness_path)
                    analysis_config = deepcopy(config)
                    analysis_config.setdefault("paths", {})
                    analysis_config["paths"]["analysis_harness_path"] = worker_harness_path
                except Exception as copy_error:
                    logger.warning(f"Failed to create worker-local post-eval harness copy: {copy_error}")

            trial = workflow_utils.run_post_eval_analysis(
                llm_name=llm_name,
                trial=trial,
                transcript=transcript,
                config=analysis_config,
                analysis_prompt=analysis_prompt,
                max_attempts=max(1, min(max_attempt, 3)),
            )
            transcript.hide_by_tag(tags=["post_eval_analysis_loop"])
            result.analysis_success = trial.analysis_success
            result.analysis_attempts = trial.analysis_attempts
            result.analysis_metrics = trial.analysis_metrics
            result.analysis_errors = trial.analysis_errors
            result.analysis_mode = getattr(trial, "analysis_mode", "disabled")
            result.analysis_results = trial.analysis_results
            result.analysis_script = getattr(trial, "analysis_script", "")
            if worker_harness_path and os.path.exists(worker_harness_path):
                try:
                    os.remove(worker_harness_path)
                except OSError:
                    pass
        if not all(trial.eval_success):
            result.error = "Evaluation failed"
            result.eval_success = False
            result.compile_attempts = trial.compile_attempts
            result.compile_errors = trial.compile_errors
            result.eval_attempts = trial.eval_attempts
            result.eval_failures = trial.eval_failures
            result.analysis_success = trial.analysis_success
            result.analysis_attempts = trial.analysis_attempts
            result.analysis_metrics = trial.analysis_metrics
            result.analysis_errors = trial.analysis_errors
            result.analysis_mode = getattr(trial, "analysis_mode", "disabled")
            result.analysis_results = trial.analysis_results
            result.analysis_script = getattr(trial, "analysis_script", "")
            result.elapsed = time.time() - t0
            result.updated_idea_repo = new_idea_repo if use_idea_repo else None
            _persist_iteration_records("eval_failed")
            return result
        result.eval_success = True

        # Parse score
        eval_score = _worker_task_eval_utils.parse_eval_results(trial.eval_results)

        # Summarise
        transcript.append(ContentChunk(prompts.EVAL_DESCRIPTION_PROMPT, "user", tags=["initial_eval_results"]))
        eval_results_text = "\n".join(["```"] + trial.eval_results + ["```"])
        transcript.append(ContentChunk(eval_results_text, "system", tags=["initial_eval_results"]))

        summary_prompt = prompts.SUMMARIZE_EVAL_PROMPT
        transcript.append(ContentChunk(summary_prompt, "user", tags=["final_summary_request"]))
        bullets = []
        for idx in range(max_attempt):
            llm_summary = llm_utils.generate_completion(llm_name, transcript, config)
            if llm_summary:
                transcript.append(ContentChunk(llm_summary, "model", tags=["final_summary_response"]))
                try:
                    bullets = workflow_utils.extract_summary(llm_summary)
                except Exception:
                    pass
                break

        result.program_code = trial.algorithm_implementation
        result.eval_score = eval_score
        result.eval_results = trial.eval_results
        result.summary_bullets = bullets
        result.idea_id = trial.idea_id
        result.success = True
        result.elapsed = time.time() - t0
        result.compile_success = True
        result.eval_success = True
        result.compile_attempts = trial.compile_attempts
        result.compile_errors = trial.compile_errors
        result.eval_attempts = trial.eval_attempts
        result.eval_failures = trial.eval_failures
        result.analysis_success = trial.analysis_success
        result.analysis_attempts = trial.analysis_attempts
        result.analysis_metrics = trial.analysis_metrics
        result.analysis_errors = trial.analysis_errors
        result.analysis_mode = getattr(trial, "analysis_mode", "disabled")
        result.analysis_results = trial.analysis_results
        result.analysis_script = getattr(trial, "analysis_script", "")

        # Update idea exp_history and attach for main process to append
        if use_idea_repo and new_idea_repo is not None and trial.idea_id != -1:
            idea = new_idea_repo.find_idea_by_id(trial.idea_id)
            if idea:
                idea.exp_history.extend(bullets)
                idea.exp_count += 1
        result.updated_idea_repo = new_idea_repo
        _persist_iteration_records(None if result.eval_score is not None else "score_parse_failed")
        return result

    except Exception as e:
        result.error = str(e)
        result.elapsed = time.time() - t0
        try:
            if 'worker_harness_path' in locals() and worker_harness_path and os.path.exists(worker_harness_path):
                os.remove(worker_harness_path)
        except OSError:
            pass
        try:
            result.compile_attempts = trial.compile_attempts
            result.compile_errors = trial.compile_errors
            result.eval_attempts = trial.eval_attempts
            result.eval_failures = trial.eval_failures
        except Exception:
            pass
        try:
            result.updated_idea_repo = new_idea_repo if use_idea_repo else None
        except NameError:
            result.updated_idea_repo = None
        if 'trial' in locals() and 'transcript' in locals():
            _persist_iteration_records("worker_exception")
        return result
    finally:
        if worker_temp_dir and os.path.exists(worker_temp_dir):
            shutil.rmtree(worker_temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main-process coordinator
# ---------------------------------------------------------------------------

async def run_parallel_evolution(
    config: dict,
    db,
    idea_repo_db,
    prompts_module,
    args,
    transcript_file: str,
    project_root: str,
    workflows_dir: str,
    num_workers: int = 4,
    analysis_manager: analysis_utils.AnalysisManager | None = None,
    enable_analysis: bool = True,
):
    """Run multi-island evolution with true process-level parallelism.

    Submits island iterations to a ProcessPoolExecutor and processes results
    as they complete, updating the shared databases in the main process.
    """
    max_iters = config['experiment']['max_iters']
    baseline_id = config['experiment']['initial_baseline_id']
    num_islands = config['database']['num_islands']

    config_for_workers = deepcopy(config)
    config_for_workers['_dataset_id'] = args.dataset_id

    per_island_count = [0] * num_islands  # Increment on submit (for trigger_merge, matches sequential)
    per_island_completed = [0] * num_islands  # Increment on success (for crossover/backtrack trigger)
    last_crossover_idx = [0] * num_islands
    in_flight_per_island = [0] * num_islands
    forced_parent_for_island: list[Optional[str]] = [None] * num_islands
    backtrack_remaining_for_island = [0] * num_islands
    repo_idx_before_backtrack: list[Optional[int]] = [None] * num_islands

    import idea_select_utils
    import workflow_utils
    import llm_utils

    completed = 0
    submitted = 0
    next_island_rr = 0

    logger.info(
        f"Starting parallel evolution: {max_iters} iterations, "
        f"{num_workers} workers, {num_islands} islands"
    )

    executor = ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(config_for_workers, project_root, workflows_dir),
    )

    try:
        pending: dict[int, tuple[Future, int, bool, Optional[int]]] = {}  # iter_id -> (fut, island_id, is_last_backtrack, bt_repo_idx)

        def _choose_ready_island() -> Optional[int]:
            nonlocal next_island_rr
            for _ in range(num_islands):
                island_id = next_island_rr
                next_island_rr = (next_island_rr + 1) % num_islands
                if in_flight_per_island[island_id] == 0:
                    return island_id
            return None

        def _submit_one() -> bool:
            nonlocal submitted
            if submitted >= max_iters:
                return False
            if len(pending) >= num_workers:
                return False
            island_id_for_iter = _choose_ready_island()
            if island_id_for_iter is None:
                return False

            is_last_backtrack = False
            bt_repo_idx = None
            parent_code = None
            idea_snapshot = None

            # Backtrack mode: use power-law sampled parent from historical repos.
            if backtrack_remaining_for_island[island_id_for_iter] > 0:
                repos = idea_repo_db.idea_repos[island_id_for_iter]
                if not repos:
                    backtrack_remaining_for_island[island_id_for_iter] = 0
                else:
                    is_last_backtrack = backtrack_remaining_for_island[island_id_for_iter] == 1
                    bt_repo_idx = repo_idx_before_backtrack[island_id_for_iter] if is_last_backtrack else None
                    sampled_idx = idea_select_utils.sample_power_law(len(repos), alpha=args.power_alpha)
                    parent_code = repos[sampled_idx].sota
                    idea_snapshot = deepcopy(repos[sampled_idx])
                    backtrack_remaining_for_island[island_id_for_iter] -= 1

            # One-step off-policy crossover parent override, or normal candidate selection.
            if parent_code is None:
                if forced_parent_for_island[island_id_for_iter] is not None:
                    parent_code = forced_parent_for_island[island_id_for_iter]
                    forced_parent_for_island[island_id_for_iter] = None
                    if args.use_idea_repo and idea_repo_db.idea_repos[island_id_for_iter]:
                        idea_snapshot = deepcopy(idea_repo_db.idea_repos[island_id_for_iter][-1])
                else:
                    parent_code, _ = db.get_candidate_for_island(island_id_for_iter)
                    if args.use_idea_repo and idea_repo_db.idea_repos[island_id_for_iter]:
                        idea_snapshot = deepcopy(idea_repo_db.idea_repos[island_id_for_iter][-1])

            iter_id = submitted
            analysis_context = None
            if analysis_manager is not None:
                analysis_context = analysis_manager.build_reasoning_context(island_id_for_iter)
            fut = executor.submit(
                _run_island_iteration,
                iter_id,
                island_id_for_iter,
                parent_code,
                idea_snapshot,
                args.use_idea_repo,
                args.use_idea_filter,
                args.max_attempt,
                baseline_id,
                transcript_file,
                enable_analysis,
                analysis_context,
            )
            pending[iter_id] = (fut, island_id_for_iter, is_last_backtrack, bt_repo_idx)
            submitted += 1
            in_flight_per_island[island_id_for_iter] += 1
            # Match sequential: increment per_island_count at start of iteration (on submit)
            per_island_count[island_id_for_iter] += 1
            return True

        # Prime workers.
        while _submit_one():
            pass

        while completed < max_iters and pending:
            done_iter = None
            done_future = None
            done_island = None
            done_is_last_backtrack = False
            done_bt_repo_idx = None
            for it, (fut, island_id, is_last_bt, bt_repo_idx) in list(pending.items()):
                if fut.done():
                    done_iter = it
                    done_future = fut
                    done_island = island_id
                    done_is_last_backtrack = is_last_bt
                    done_bt_repo_idx = bt_repo_idx
                    break

            if done_iter is None:
                await asyncio.sleep(0.05)
                continue

            pending.pop(done_iter)
            in_flight_per_island[done_island] = max(0, in_flight_per_island[done_island] - 1)

            try:
                result: IterationResult = done_future.result()
            except Exception as e:
                logger.error(f"Iteration {done_iter} raised exception: {e}")
                completed += 1
                while _submit_one():
                    pass
                continue

            island_id = result.island_id

            def _compute_trigger_merge(repo, is_last_bt: bool, bt_repo_idx: Optional[int]) -> bool:
                """Match sequential trigger_merge logic (per-island)."""
                if args.idea_cap <= -1 or args.merge_freq <= -1:
                    return False
                if repo is None or len(repo.ideas) <= args.idea_cap:
                    return False
                if args.use_integrated_sampling and per_island_count[island_id] - last_crossover_idx[island_id] == args.freeze_period:
                    return True
                if args.backtrack_freq == -1 or (completed + 1) % args.backtrack_freq == args.backtrack_freq - 1:
                    return True
                return False

            def _apply_failure_merge(repo, is_last_bt: bool, bt_repo_idx: Optional[int]) -> None:
                """Merge backtrack results and call merge_ideas on failure (match sequential)."""
                if repo is None:
                    return
                if is_last_bt and args.merge_freq > -1 and bt_repo_idx is not None and idea_repo_db.idea_repos[island_id]:
                    repo.ideas.extend(idea_repo_db.idea_repos[island_id][bt_repo_idx].ideas)
                    repo.reindex_ideas()
                    logger.info(f"Island {island_id}: Merged backtrack results on failure")
                if _compute_trigger_merge(repo, is_last_bt, bt_repo_idx):
                    workflow_utils.merge_ideas(config["llm"]["name"], transcript_file, config, repo, args.idea_cap)

            def _record_iteration_analysis(result_obj: IterationResult, failure_reason: Optional[str]) -> None:
                if analysis_manager is None:
                    return
                analysis_manager.record_iteration(
                    analysis_utils.IterationAnalysisRecord(
                        iteration=result_obj.iteration,
                        island_id=result_obj.island_id,
                        success=result_obj.success,
                        compile_success=result_obj.compile_success,
                        eval_success=result_obj.eval_success,
                        compile_attempts=result_obj.compile_attempts,
                        eval_attempts=result_obj.eval_attempts,
                        analysis_attempts=result_obj.analysis_attempts,
                        idea_id=result_obj.idea_id,
                        eval_score=result_obj.eval_score,
                        analysis_success=result_obj.analysis_success,
                        analysis_metrics=result_obj.analysis_metrics,
                        summary_bullets=result_obj.summary_bullets,
                        compile_errors=result_obj.compile_errors,
                        eval_failures=result_obj.eval_failures,
                        analysis_errors=result_obj.analysis_errors,
                        eval_results=result_obj.eval_results,
                        failure_reason=failure_reason,
                        elapsed_seconds=result_obj.elapsed,
                    )
                )

            if result.success and result.eval_score is not None:
                db.register_program(
                    program=result.program_code,
                    island_id=island_id,
                    score=result.eval_score,
                )
                idea_repo_db.best_scores_history[island_id].append(result.eval_score)
                idea_repo_db.scheduler.update_score(island_id, result.eval_score)

                # Keep crossover trigger semantics close to sequential:
                # trigger after freeze period and after score update (use completed count)
                per_island_completed[island_id] += 1
                if (
                    args.use_integrated_sampling
                    and per_island_completed[island_id] - last_crossover_idx[island_id] > args.freeze_period
                    and idea_repo_db.scheduler.check_trigger(island_id)
                ):
                    action, cross_id = idea_repo_db.scheduler.sample_action(island_id)
                    if action == "CROSSOVER" and cross_id is not None:
                        last_crossover_idx[island_id] = per_island_completed[island_id]
                        # One-step off-policy: next sample on this island uses partner island SOTA.
                        _, cross_sota = db.get_best_program_for_island(cross_id)
                        if cross_sota is not None:
                            forced_parent_for_island[island_id] = cross_sota
                        if idea_repo_db.idea_repos[island_id]:
                            current_repo = deepcopy(idea_repo_db.idea_repos[island_id][-1])
                            best_repo = idea_repo_db.get_best_idea_repo(cross_id)
                            current_repo.ideas.extend(best_repo.ideas)
                            current_repo.reindex_ideas()
                            idea_repo_db.idea_repos[island_id].append(current_repo)
                    elif action == "BACKTRACK":
                        last_crossover_idx[island_id] = per_island_completed[island_id]
                        if idea_repo_db.idea_repos[island_id]:
                            backtrack_remaining_for_island[island_id] = args.backtrack_len
                            repo_idx_before_backtrack[island_id] = len(idea_repo_db.idea_repos[island_id]) - 1
                            logger.info(f"Island {island_id}: Entering backtrack mode for {args.backtrack_len} iterations")

                # Append updated idea repo (with merge on last backtrack iteration)
                if result.updated_idea_repo is not None:
                    repo_to_append = result.updated_idea_repo
                    if done_is_last_backtrack and done_bt_repo_idx is not None and idea_repo_db.idea_repos[island_id]:
                        repo_to_append.ideas.extend(idea_repo_db.idea_repos[island_id][done_bt_repo_idx].ideas)
                        repo_to_append.reindex_ideas()
                        logger.info(f"Island {island_id}: Merged backtrack results with main repo")

                    # Idea summarization (match sequential)
                    if result.idea_id != -1:
                        idea = repo_to_append.find_idea_by_id(result.idea_id)
                        if idea and idea.exp_count % args.summarize_freq == 0:
                            idea_select_utils.summarize(
                                idea, config["llm"]["name"], config,
                                llm_utils.Transcript(log_filename=transcript_file),
                                llm_utils.Transcript(log_filename=transcript_file),
                            )

                    # merge_ideas when trigger_merge (match sequential)
                    trigger_merge = _compute_trigger_merge(repo_to_append, done_is_last_backtrack, done_bt_repo_idx)
                    if trigger_merge:
                        workflow_utils.merge_ideas(config["llm"]["name"], transcript_file, config, repo_to_append, args.idea_cap)

                    idea_repo_db.idea_repos[island_id].append(repo_to_append)

                logger.info(
                    f"Iteration {result.iteration} (island {island_id}) completed in "
                    f"{result.elapsed:.1f}s  score={result.eval_score}"
                    + (
                        f" gpu={result.cuda_visible_devices}"
                        if result.cuda_visible_devices is not None else ""
                    )
                )
                if result.summary_bullets:
                    logger.info("Summary: " + " | ".join(result.summary_bullets[:3]))
                _record_iteration_analysis(result, None)
            else:
                # Failure: apply merge logic to match sequential (last_bt_iter merge, merge_ideas)
                _apply_failure_merge(result.updated_idea_repo, done_is_last_backtrack, done_bt_repo_idx)
                failure_reason = result.error or ("score_parse_failed" if result.success else "iteration_failed")
                logger.warning(
                    f"Iteration {result.iteration} (island {island_id}) failed: "
                    f"{result.error or 'unknown'}"
                    + (
                        f" gpu={result.cuda_visible_devices}"
                        if result.cuda_visible_devices is not None else ""
                    )
                )
                _record_iteration_analysis(result, failure_reason)

            completed += 1
            while _submit_one():
                pass

        logger.info(f"Parallel evolution finished. {completed}/{max_iters} iterations completed.")

    finally:
        executor.shutdown(wait=False)
