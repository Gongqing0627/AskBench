"""Thin wrapper around OpenAI-style APIs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .base import ChatMessage, LLMClient, LLMResponse


@dataclass(slots=True)
class OpenAIClient(LLMClient):
    """Adapter for OpenAI-compatible APIs.

    This implementation is intentionally light-weight and leaves the concrete
    HTTP interaction to the provided callable hooks so the framework remains
    provider agnostic and easy to test.
    """

    model_name: str
    chat_callable: Any
    completion_callable: Any
    embedding_callable: Any

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        raw = self.completion_callable(model=self.model_name, prompt=prompt, **kwargs)
        content = raw["choices"][0]["text"] if isinstance(raw, dict) else str(raw)
        return LLMResponse(content=content, raw=raw)

    def chat(self, messages: Iterable[ChatMessage], **kwargs: Any) -> LLMResponse:
        raw = self.chat_callable(model=self.model_name, messages=list(messages), **kwargs)
        content = raw["choices"][0]["message"]["content"] if isinstance(raw, dict) else str(raw)
        return LLMResponse(content=content, raw=raw)

    def embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        raw = self.embedding_callable(model=self.model_name, input=texts, **kwargs)
        if isinstance(raw, dict) and "data" in raw:
            return [item["embedding"] for item in raw["data"]]
        raise ValueError("Embedding callable returned unsupported payload")

