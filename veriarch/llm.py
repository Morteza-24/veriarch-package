"""Pluggable LLM backend.

architect.py and descriptors.py only depend on the LLMClient interface
below, so swapping between the Anthropic API and a local/free model
(Ollama running Qwen, Llama, etc., or any OpenAI-compatible server) is a
config change (see VeriArchConfig.llm_provider), not an edit to the
reasoning logic itself.
"""

import os
from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Return the raw text completion for a single-turn prompt."""


class AnthropicClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str = None):
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OllamaClient(LLMClient):
    """Talks to a local Ollama server (https://ollama.com). No API key
    needed. Run `ollama pull qwen2.5:7b` (or any other model) first, and
    `ollama serve` if it isn't already running as a background service.
    """

    def __init__(self, model: str = "qwen2.5:7b", base_url: str = None, timeout: int = 180):
        self.model = model
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        import requests

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.2},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["response"]


class OpenAICompatibleClient(LLMClient):
    """Talks to any OpenAI-chat-completions-compatible endpoint: vLLM,
    LM Studio, text-generation-webui, Ollama's own /v1 endpoint, a
    hosted free-tier Qwen endpoint, etc."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
        timeout: int = 180,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        import requests

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def get_llm_client(
    provider: str, model: str, api_key: str = None, base_url: str = None
) -> LLMClient:
    provider = provider.lower()
    if provider == "anthropic":
        return AnthropicClient(model=model, api_key=api_key)
    if provider == "ollama":
        return OllamaClient(model=model, base_url=base_url)
    if provider == "openai_compatible":
        return OpenAICompatibleClient(
            model=model, base_url=base_url or "http://localhost:8000/v1", api_key=api_key or "not-needed"
        )
    raise ValueError(
        f"Unknown LLM provider: {provider!r}. Expected one of: "
        "'anthropic', 'ollama', 'openai_compatible'."
    )
