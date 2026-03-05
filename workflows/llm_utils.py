# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import MutableSequence
import dataclasses
import json
import re
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Union

# --- Logger Setup ---
logger = logging.getLogger("controller")

@dataclasses.dataclass
class ContentChunk:
    content: str
    role: str
    tags: list[str] = dataclasses.field(default_factory=list)
    hidden: bool = False

    def format(self):
        return f"<start_of_turn>{self.role}\n{self.content}<end_of_turn>\n"


class Transcript(MutableSequence):
    def __init__(self, log_filename: str | None = None):
        self._log_filename = log_filename
        self._list = list()

    def _log_to_file(self, v):
        if isinstance(v, ContentChunk) and self._log_filename:
            json_str = dataclasses.asdict(v)
            with open(self._log_filename, 'a') as f:
                f.write(json.dumps(json_str) + "\n")

    def hide_by_tag(self, tags: list[str]):
        for chunk in self._list:
            if any(tag in chunk.tags for tag in tags):
                chunk.hidden = True

    def unhide_all_tags(self):
        for chunk in self._list:
            chunk.hidden = False

    def format(self):
        output_text = []
        for chunk in self._list:
            if chunk.hidden:
                continue
            output_text.append(chunk.format())
        return "".join(output_text)

    def log_debug_message(self, message: str):
        c = ContentChunk(
            content=message,
            role="info",
            tags=["debug_message"]
        )
        self._log_to_file(c)

    def __len__(self): return len(self._list)
    def __getitem__(self, i): return self._list[i]
    def __delitem__(self, i): del self._list[i]
    def __str__(self): return str(self._list)
    def __setitem__(self, i, v):
        self._list[i] = v

    def insert(self, i, v):
        if i == len(self._list):
            self._log_to_file(v)
        self._list.insert(i, v)

# --- LLM Abstraction ---
class LLMClient(ABC):
    """Abstract Base Class for Language Model Clients."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass

    @abstractmethod
    def generate(self, prompt: str, generation_config: Dict[str, Any]) -> str:
        pass

# --- Gemini Client ---
try:
    import google.generativeai as genai
    from google.generativeai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

if GEMINI_AVAILABLE:
    class GeminiClient(LLMClient):
        def __init__(self, config: Dict[str, Any]):
            super().__init__(config)
            self.model_name = self.config.get("name", "gemini-3-pro-preview")
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                logger.warning("GOOGLE_API_KEY not found in environment.")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)

        def count_tokens(self, text: str) -> int:
            return self.model.count_tokens(text).total_tokens

        def generate(self, prompt: str, generation_config: Dict[str, Any]) -> str:
            # Map common keys to Gemini specific keys
            gen_config = genai_types.GenerationConfig(**generation_config)
            response = self.model.generate_content(prompt, generation_config=gen_config)
            return response.text

# --- OpenAI Client ---
try:
    from openai import OpenAI
    import tiktoken
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

if OPENAI_AVAILABLE:
    class OpenAIClient(LLMClient):
        def __init__(self, config: Dict[str, Any]):
            super().__init__(config)
            self.model_name = self.config.get("name", "gpt-5.2-pro")
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        def count_tokens(self, text: str) -> int:
            """Returns the number of tokens in a text string."""
            encoding = tiktoken.encoding_for_model(self.model_name)
            num_tokens = len(encoding.encode(text))
            return num_tokens

        def generate(self, prompt: str, generation_config: Dict[str, Any]) -> str:
            # Convert generation_config to OpenAI format
            params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": generation_config.get("temperature", 1.0),
                "top_p": generation_config.get("top_p", 0.95),
                "max_tokens": generation_config.get("max_output_tokens", 4096)
            }
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content
    
    
    class OllamaClient(LLMClient):
        """Ollama local LLM via OpenAI-compatible API (http://localhost:11434/v1)."""

        def __init__(self, config: Dict[str, Any]):
            super().__init__(config)
            self.model_name = self.config.get("name", "llama3.2")
            base_url = self.config.get("base_url", "http://localhost:11434/v1")
            self.client = OpenAI(
                base_url=base_url,
                api_key=self.config.get("api_key", "ollama"),
            )

        def count_tokens(self, text: str) -> int:
            """Approximate token count (~4 chars/token for most models)."""
            return max(1, len(text) // 4)

        def generate(self, prompt: str, generation_config: Dict[str, Any]) -> str:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=generation_config.get("temperature", 1.0),
                max_tokens=generation_config.get("max_output_tokens", 16384),
            )
            return response.choices[0].message.content or ""

# --- Anthropic Client ---
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

if ANTHROPIC_AVAILABLE:
    class AnthropicClient(LLMClient):
        def __init__(self, config: Dict[str, Any]):
            super().__init__(config)
            self.model_name = self.config.get("name", "claude-opus-4-6")
            self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        def count_tokens(self, text: str) -> int:
            """
            Calls the Anthropic API to get a precise token count for the input string.
            """
            response = self.client.messages.count_tokens(
                model=self.model_name,
                messages=[{"role": "user", "content": text}]
            )
            return response.input_tokens

        def generate(self, prompt: str, generation_config: Dict[str, Any]) -> str:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=generation_config.get("max_output_tokens", 4096),
                temperature=generation_config.get("temperature", 1.0),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

# --- Client Factory and Cache ---

_CLIENT_CACHE: Dict[str, LLMClient] = {}

def get_llm_client(llm_name: str, config: Dict[str, Any]) -> LLMClient:
    """Singleton pattern to retrieve or create LLM clients."""
    if llm_name in _CLIENT_CACHE:
        return _CLIENT_CACHE[llm_name]

    llm_config = config.get('llm', {})
    client_type = llm_config.get('client_type', 'gemini')
    
    # Override client_type based on model name if user just switched the name
    if 'gpt' in llm_name or 'openai' in llm_name:
        client_type = 'openai'
    elif 'claude' in llm_name or 'anthropic' in llm_name:
        client_type = 'anthropic'
    elif 'gemini' in llm_name:
        client_type = 'gemini'

    client = None
    if client_type == 'gemini' and GEMINI_AVAILABLE:
        client = GeminiClient(llm_config)
    elif client_type == 'openai' and OPENAI_AVAILABLE:
        client = OpenAIClient(llm_config)
    elif client_type == 'anthropic' and ANTHROPIC_AVAILABLE:
        client = AnthropicClient(llm_config)
    else:
        raise ValueError(f"Unsupported or missing client type: {client_type}")

    _CLIENT_CACHE[llm_name] = client
    logger.info(f"Initialized LLM Client: {client_type} for model {llm_name}")
    return client

# --- Completion Function ---

def generate_completion(
    llm_name_or_client: Union[str, LLMClient],
    transcript: Transcript,
    config: dict,
):
    """
    Generates a completion.
    Accepts either a string (model name) or an instantiated LLMClient.
    If a string is passed, it looks up/creates the client using the config.
    """
    
    # 1. Resolve the Client
    if isinstance(llm_name_or_client, str):
        llm_client = get_llm_client(llm_name_or_client, config)
    else:
        llm_client = llm_name_or_client

    prompt = transcript.format()
    llm_config = config.get('llm', {})

    # 2. Token Counting (Safe)
    try:
        token_count = llm_client.count_tokens(prompt)
    except Exception as e:
        logger.warning(f"Could not count tokens: {e}")
        token_count = 0

    # 3. Truncation Logic
    max_context = llm_config.get('context_size', 100000) # Default high for modern models
    if token_count > max_context:
        logger.warning(f"Prompt too long ({token_count}), truncating...")
        
    final_prompt = prompt 

    generation_config = {
        "temperature": llm_config.get("temperature", 1.0),
        "top_p": llm_config.get("top_p", 0.95),
        "max_output_tokens": llm_config.get("max_output_tokens", 32768)
    }

    # 5. Execution with Retry
    max_tries = llm_config.get('max_try_count', 3)
    output_text = None

    for i in range(max_tries):
        try:
            output_text = llm_client.generate(final_prompt, generation_config)
            break
        except Exception as e:
            logger.error(f"LLM generate call failed on try {i + 1}/{max_tries}: {e}")
            if i == max_tries - 1:
                return None

    if output_text is None:
        return None

    # Cleanup tags
    output_text = output_text.replace("<end_of_turn>", "").replace("<start_of_turn>", "").strip()
    return output_text


def extract_code_blocks(markdown_string: str) -> list[str]:
    if not markdown_string:
        return []
    code_blocks = re.findall(
        r'```(?:[a-zA-Z0-9_+\.-]+)?\n(.*?)\n```', markdown_string, re.DOTALL
    )
    return [block for block in code_blocks]