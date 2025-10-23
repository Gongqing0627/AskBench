"""LLM powered question generation."""
from __future__ import annotations

import json
import uuid
from typing import Iterable, Sequence

from ..config import GenerationConfig
from ..domain.models import DocumentChunk, QuestionCandidate
from ..llm.base import LLMClient


SYSTEM_PROMPT = """You are an expert question writer creating rigorous multiple-choice questions."""


class QuestionGenerator:
    """Generates multi-choice question candidates from document chunks."""

    def __init__(self, client: LLMClient, config: GenerationConfig) -> None:
        self.client = client
        self.config = config

    def build_prompt(self, chunk: DocumentChunk) -> list[dict[str, str]]:
        """Return chat-formatted prompt for the LLM."""

        instruction = {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
        user = {
            "role": "user",
            "content": (
                "Create up to {max_q} multiple-choice questions based on the provided context.\n"
                "Each question must have exactly {opt_count} answer options and a single correct answer.\n"
                "Return JSON with fields question, options, answer_index, rationale.\n"
                "Context:\n{context}"
            ).format(
                max_q=self.config.max_questions_per_chunk,
                opt_count=4,
                context=chunk.text,
            ),
        }
        return [instruction, user]

    def parse_response(self, payload: str, chunk: DocumentChunk) -> Sequence[QuestionCandidate]:
        """Parse the LLM response into structured candidates."""

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse LLM response: {exc}") from exc

        candidates: list[QuestionCandidate] = []
        for index, item in enumerate(data if isinstance(data, list) else [data]):
            options = list(item.get("options", []))
            candidate = QuestionCandidate(
                question_id=item.get("id") or str(uuid.uuid4()),
                prompt_id=chunk.chunk_id,
                question=item.get("question", "").strip(),
                options=options,
                correct_answer_index=int(item.get("answer_index", 0)),
                source_chunk_ids=[chunk.chunk_id],
                rationale=item.get("rationale"),
                metadata={"llm_index": str(index)},
            )
            candidates.append(candidate)
        return candidates

    def generate(self, chunk: DocumentChunk) -> Sequence[QuestionCandidate]:
        messages = self.build_prompt(chunk)
        response = self.client.chat(messages, temperature=self.config.temperature, top_p=self.config.top_p)
        return self.parse_response(response.content, chunk)

    def bulk_generate(self, chunks: Iterable[DocumentChunk]) -> Iterable[QuestionCandidate]:
        for chunk in chunks:
            yield from self.generate(chunk)

