"""The 'brain' of AutoExpert: a local model running in Ollama (no API key needed).

Everything else (the API, the benchmark) talks to the brain through ONE method:

    async for piece in provider.stream(system, messages): ...

`piece` is a piece of text while the answer is being written, and one final
`Usage` object at the very end (token counts).

Because of this shared "interface", you can later add a hosted model (OpenAI,
Gemini, ...) by writing one more class like OllamaProvider. Nothing else changes.
"""
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = "You are AutoExpert, a friendly expert. Give clear, short answers."


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


class LLMProvider(ABC):
    model: str

    @abstractmethod
    def stream(self, system: str, messages: list[dict]) -> AsyncIterator["str | Usage"]:
        """Yield text pieces as they arrive, then one Usage at the end."""


class OllamaProvider(LLMProvider):
    """An open model running on your own computer through Ollama."""

    def __init__(self, model: str):
        from ollama import AsyncClient

        self.model = model
        self.client = AsyncClient(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))

    async def stream(self, system, messages):
        msgs = [{"role": "system", "content": system}] + messages
        usage = Usage(0, 0)
        stream = await self.client.chat(model=self.model, messages=msgs, stream=True)
        async for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield token
            if chunk.get("done"):  # the last chunk carries the token counts
                usage = Usage(
                    chunk.get("prompt_eval_count") or 0,
                    chunk.get("eval_count") or 0,
                )
        yield usage


_cache: dict[str, LLMProvider] = {}


def get_provider(model: str | None = None) -> LLMProvider:
    """Get the brain for a local model name, e.g. 'llama3.2' or 'qwen2.5:3b'.

    If no name is given, use MODEL_NAME from .env (default: llama3.2).
    """
    model = model or os.getenv("MODEL_NAME", "llama3.2")
    if model not in _cache:
        _cache[model] = OllamaProvider(model)
    return _cache[model]
