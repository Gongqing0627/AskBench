"""Automated validation and filtering logic."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Protocol

from ..config import ValidationConfig
from ..domain.models import QuestionCandidate, ValidationResult, ValidatedQuestion
from ..llm.base import LLMClient


class CandidateScorer(Protocol):
    """Callable protocol for scoring questions."""

    def __call__(self, candidate: QuestionCandidate) -> float:
        ...


@dataclass(slots=True)
class SelfConsistencyScorer:
    """Use an LLM to self-evaluate correctness of generated questions."""

    client: LLMClient
    n_votes: int = 3

    def __call__(self, candidate: QuestionCandidate) -> float:
        prompt = (
            "Question: {question}\nOptions: {options}\nCorrect index: {answer}\n"
            "Does the correct answer align with the context? Answer yes or no."
        ).format(
            question=candidate.question,
            options=" | ".join(candidate.options),
            answer=candidate.correct_answer_index,
        )
        votes = []
        for _ in range(self.n_votes):
            response = self.client.complete(prompt, temperature=0.2)
            votes.append(1.0 if response.content.lower().startswith("y") else 0.0)
        return mean(votes)


class CandidateValidator:
    """Runs a sequence of validation checks and returns validation results."""

    def __init__(self, config: ValidationConfig, scorer: CandidateScorer | None = None) -> None:
        self.config = config
        self.scorer = scorer

    def validate(self, candidate: QuestionCandidate) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        if len(candidate.options) < self.config.min_option_count:
            errors.append("Insufficient number of answer options")

        if not candidate.question:
            errors.append("Question text is empty")

        if candidate.correct_answer_index >= len(candidate.options):
            errors.append("Correct answer index out of range")

        if self.config.require_rationale and not candidate.rationale:
            warnings.append("Missing rationale")

        score = None
        if self.scorer:
            score = self.scorer(candidate)
            if score < self.config.min_quality_score:
                errors.append(f"Quality score below threshold: {score:.2f}")

        return ValidationResult(
            question_id=candidate.question_id,
            is_valid=not errors,
            errors=errors,
            warnings=warnings,
            score=score,
        )

    def filter(self, candidates: Iterable[QuestionCandidate]) -> Iterable[ValidatedQuestion]:
        for candidate in candidates:
            result = self.validate(candidate)
            if result.is_valid:
                yield ValidatedQuestion(candidate=candidate, validation=result)

