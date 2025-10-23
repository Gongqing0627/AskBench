"""Base interfaces for LLM interactions."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Protocol


class ChatMessage(Protocol):
    """Represents a chat completion style message."""

    role: str
    content: str


@dataclass(slots=True)
class LLMResponse:
    """Normalized response from an LLM call."""

    content: str
    raw: Any | None = None


class LLMClient(ABC):
    """Abstract base class for all LLM provider clients."""

    model_name: str

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate a completion for a raw text prompt."""

    @abstractmethod
    def chat(self, messages: Iterable[ChatMessage], **kwargs: Any) -> LLMResponse:
        """Generate a completion for chat-formatted messages."""

    @abstractmethod
    def embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Create embeddings for the provided texts."""

