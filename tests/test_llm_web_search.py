import importlib
import pathlib
import sys
import types
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = REPO_ROOT / "workflows"
if str(WORKFLOWS_DIR) not in sys.path:
  sys.path.insert(0, str(WORKFLOWS_DIR))


class _FakeEncoding:

  def encode(self, text):
    return list(text)


class _FakeChatCompletions:

  def __init__(self, calls):
    self.calls = calls

  def create(self, **kwargs):
    self.calls["chat.completions.create"] = kwargs
    return types.SimpleNamespace(
      choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="chat answer"))]
    )


class _FakeResponses:

  def __init__(self, calls):
    self.calls = calls

  def create(self, **kwargs):
    self.calls["responses.create"] = kwargs
    return types.SimpleNamespace(output_text="responses answer")


class _FakeOpenAI:
  instances = []

  def __init__(self, *args, **kwargs):
    self.calls = {}
    self.chat = types.SimpleNamespace(
      completions=_FakeChatCompletions(self.calls)
    )
    self.responses = _FakeResponses(self.calls)
    _FakeOpenAI.instances.append(self)


class _FakeMessages:

  def __init__(self, calls):
    self.calls = calls

  def count_tokens(self, **kwargs):
    self.calls["messages.count_tokens"] = kwargs
    return types.SimpleNamespace(input_tokens=42)

  def create(self, **kwargs):
    self.calls["messages.create"] = kwargs
    return types.SimpleNamespace(
      content=[types.SimpleNamespace(type="text", text="claude answer")]
    )


class _FakeAnthropic:
  instances = []

  def __init__(self, *args, **kwargs):
    self.calls = {}
    self.messages = _FakeMessages(self.calls)
    _FakeAnthropic.instances.append(self)


def _load_llm_utils():
  sys.modules["openai"] = types.SimpleNamespace(OpenAI=_FakeOpenAI)
  sys.modules["anthropic"] = types.SimpleNamespace(Anthropic=_FakeAnthropic)
  sys.modules["tiktoken"] = types.SimpleNamespace(
    encoding_for_model=lambda _model: _FakeEncoding(),
    get_encoding=lambda _name: _FakeEncoding(),
  )

  sys.modules.pop("llm_utils", None)
  return importlib.import_module("llm_utils")


class LLMWebSearchTest(unittest.TestCase):

  def setUp(self):
    _FakeOpenAI.instances.clear()
    _FakeAnthropic.instances.clear()
    self.llm_utils = _load_llm_utils()
    self.llm_utils._CLIENT_CACHE.clear()

  def test_openai_uses_responses_api_with_web_search(self):
    transcript = self.llm_utils.Transcript()
    transcript.append(self.llm_utils.ContentChunk("Find the latest papers.", "user"))

    config = {
      "llm": {
        "name": "gpt-5",
        "client_type": "openai",
        "reasoning_effort": "low",
        "max_output_tokens": 2048,
        "web_search": {
          "enabled": True,
          "allowed_domains": ["arxiv.org", "openreview.net"],
          "include_sources": True,
          "external_web_access": True,
        },
      }
    }

    response = self.llm_utils.generate_completion("gpt-5", transcript, config)

    self.assertEqual(response, "responses answer")
    params = _FakeOpenAI.instances[-1].calls["responses.create"]
    self.assertEqual(params["model"], "gpt-5")
    self.assertEqual(params["tool_choice"], "auto")
    self.assertEqual(params["reasoning"], {"effort": "low"})
    self.assertEqual(params["include"], ["web_search_call.action.sources"])
    self.assertEqual(params["tools"][0]["type"], "web_search")
    self.assertEqual(
      params["tools"][0]["filters"]["allowed_domains"],
      ["arxiv.org", "openreview.net"],
    )

  def test_openai_keeps_chat_completions_without_web_search(self):
    transcript = self.llm_utils.Transcript()
    transcript.append(self.llm_utils.ContentChunk("Say hi.", "user"))

    config = {
      "llm": {
        "name": "gpt-5",
        "client_type": "openai",
      }
    }

    response = self.llm_utils.generate_completion("gpt-5", transcript, config)

    self.assertEqual(response, "chat answer")
    calls = _FakeOpenAI.instances[-1].calls
    self.assertIn("chat.completions.create", calls)
    self.assertNotIn("responses.create", calls)

  def test_anthropic_adds_web_search_tool(self):
    transcript = self.llm_utils.Transcript()
    transcript.append(self.llm_utils.ContentChunk("Research the newest benchmark.", "user"))

    config = {
      "llm": {
        "name": "claude-sonnet-4-6",
        "client_type": "anthropic",
        "web_search": {
          "enabled": True,
          "tool_type": "web_search_20260209",
          "max_uses": 3,
          "allowed_domains": ["docs.anthropic.com"],
          "user_location": {
            "type": "approximate",
            "country": "US",
            "timezone": "America/Los_Angeles",
          },
        },
      }
    }

    response = self.llm_utils.generate_completion(
      "claude-sonnet-4-6",
      transcript,
      config,
    )

    self.assertEqual(response, "claude answer")
    params = _FakeAnthropic.instances[-1].calls["messages.create"]
    self.assertEqual(params["tools"][0]["type"], "web_search_20260209")
    self.assertEqual(params["tools"][0]["name"], "web_search")
    self.assertEqual(params["tools"][0]["max_uses"], 3)
    self.assertEqual(
      params["tools"][0]["allowed_domains"],
      ["docs.anthropic.com"],
    )

  def test_client_cache_varies_with_llm_config(self):
    config_a = {
      "llm": {
        "name": "gpt-5",
        "client_type": "openai",
      }
    }
    config_b = {
      "llm": {
        "name": "gpt-5",
        "client_type": "openai",
        "web_search": {"enabled": True},
      }
    }

    client_a = self.llm_utils.get_llm_client("gpt-5", config_a)
    client_b = self.llm_utils.get_llm_client("gpt-5", config_b)

    self.assertIsNot(client_a, client_b)


if __name__ == "__main__":
  unittest.main()
