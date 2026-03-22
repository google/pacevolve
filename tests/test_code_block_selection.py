import pathlib
import sys
import types
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = REPO_ROOT / "workflows"
if str(WORKFLOWS_DIR) not in sys.path:
  sys.path.insert(0, str(WORKFLOWS_DIR))

# These workflow modules import optional runtime dependencies that are not
# needed for these parser-focused tests.
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))

import llm_utils
import workflow_utils


class CodeBlockSelectionTest(unittest.TestCase):

  def test_extract_code_blocks_prefers_requested_language(self):
    response = """Idea ID: 1
blabla

```text
Idea ID: 1
blabla
```

```python
print("real code")
```
"""

    code_blocks = llm_utils.extract_code_blocks(
      response,
      preferred_languages=["python", "py"],
    )

    self.assertEqual(code_blocks, ['print("real code")'])

  def test_extract_code_blocks_falls_back_to_original_order(self):
    response = """```text
Idea ID: 1
```

```python
print("real code")
```
"""

    code_blocks = llm_utils.extract_code_blocks(response)

    self.assertEqual(code_blocks[0], "Idea ID: 1")
    self.assertEqual(code_blocks[1], 'print("real code")')

  def test_python_targets_prefer_python_fences(self):
    response = """```text
Idea ID: 1
```

```python
print("real code")
```
"""

    preferred_languages = workflow_utils._preferred_fence_languages_for_file(
      "train_gpt.py"
    )
    code_blocks = llm_utils.extract_code_blocks(
      response,
      preferred_languages=preferred_languages,
    )

    self.assertEqual(preferred_languages, ["python", "py"])
    self.assertEqual(code_blocks, ['print("real code")'])


if __name__ == "__main__":
  unittest.main()
