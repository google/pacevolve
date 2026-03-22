import copy
import pathlib
import sys
import types
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = REPO_ROOT / "workflows"
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))
if str(WORKFLOWS_DIR) not in sys.path:
  sys.path.insert(0, str(WORKFLOWS_DIR))

# These workflow imports bring in optional runtime dependencies that are not
# needed for these validation-focused tests.
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))

import task_utils
import workflow_utils


class CandidateValidationTest(unittest.TestCase):

  def test_parse_candidate_validation_response(self):
    verdict, feedback = workflow_utils._parse_candidate_validation_response(
      "Verdict: PASS\nReason: preserved semantics."
    )

    self.assertTrue(verdict)
    self.assertIn("preserved semantics", feedback)

  def test_resolve_candidate_validation_hook_for_parameter_golf(self):
    config = {
      "experiment": {"task_id": "parameter_golf"},
      "paths": {
        "src_path": str(REPO_ROOT / "tasks" / "parameter_golf" / "src"),
      },
      "candidate_validation": {"max_retries": 2},
    }

    hook = workflow_utils._resolve_candidate_validation_hook(config)

    self.assertIsNotNone(hook)
    self.assertEqual(hook.reference_file_label, "train_gpt_ref.py")
    self.assertEqual(hook.max_retries, 2)
    self.assertIn("load_validation_tokens", hook.reference_code)

  def test_edit_until_compile_retries_after_candidate_validation_fail(self):
    compile_config = task_utils.CompilationConfig(
      target_file_path=str(
        REPO_ROOT / "tasks" / "parameter_golf" / "src" / "train_gpt.py"
      )
    )
    config = {
      "experiment": {"task_id": "parameter_golf"},
      "paths": {
        "src_path": str(REPO_ROOT / "tasks" / "parameter_golf" / "src"),
      },
      "candidate_validation": {"max_retries": 2},
    }
    loop_config = {
      "max_attempts": 3,
      "loop_tag": "initial_compile_loop",
      "summary_tag": "initial_compile_summary",
    }

    transcript = workflow_utils.Transcript()
    transcript.append(
      workflow_utils.ContentChunk("```python\nprint('bad')\n```", "model")
    )
    trial = workflow_utils.AlgorithmTrial()

    llm_responses = iter([
      "Verdict: FAIL\nReason: the candidate altered protected evaluation logic.\nProtected areas:\n- evaluation\n- data order\n- reporting",
      "```python\nprint('good')\n```",
      "Verdict: PASS\nReason: protected semantics are preserved.\nProtected areas:\n- evaluation\n- data order\n- reporting",
    ])
    compiled_candidates = []

    original_generate_completion = workflow_utils.llm_utils.generate_completion
    original_attempt_compile = workflow_utils.attempt_compile

    def fake_generate_completion(*_args, **_kwargs):
      return next(llm_responses)

    def fake_attempt_compile(trial, _compile_config, _config):
      compiled_candidates.append(trial.algorithm_implementation)
      output_trial = copy.deepcopy(trial)
      output_trial.compile_success = True
      return output_trial, "Code compiled successfully.", None

    workflow_utils.llm_utils.generate_completion = fake_generate_completion
    workflow_utils.attempt_compile = fake_attempt_compile
    try:
      output_trial = workflow_utils.edit_until_compile(
        "fake-llm",
        trial,
        transcript,
        compile_config,
        config,
        loop_config=loop_config,
      )
    finally:
      workflow_utils.llm_utils.generate_completion = original_generate_completion
      workflow_utils.attempt_compile = original_attempt_compile

    self.assertTrue(output_trial.compile_success)
    self.assertEqual(compiled_candidates, ["print('good')"])
    self.assertIn("Candidate validation failed", "\n".join(output_trial.compile_errors))


if __name__ == "__main__":
  unittest.main()
