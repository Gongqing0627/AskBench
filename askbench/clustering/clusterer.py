"""Question clustering and deduplication."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from ..config import ClusteringConfig
from ..domain.models import Cluster, ClusterSummary, ValidatedQuestion
from ..llm.base import LLMClient


@dataclass(slots=True)
class EmbeddingClusterer:
    """Clusters validated questions using vector similarity."""

    client: LLMClient
    config: ClusteringConfig

    def embed_questions(self, questions: Iterable[ValidatedQuestion]) -> list[list[float]]:
        texts = [item.candidate.question for item in questions]
        if not texts:
            return []
        return self.client.embed(texts, model=self.config.embedding_model)

    def cluster(self, questions: list[ValidatedQuestion]) -> list[Cluster]:
        if not questions:
            return []

        embeddings = self.embed_questions(questions)
        clusters: list[Cluster] = []
        assigned: list[int] = [-1] * len(questions)
        cluster_id = 0

        for idx, emb in enumerate(embeddings):
            if assigned[idx] != -1:
                continue
            cluster_id += 1
            members = [idx]
            for other_idx, other_emb in enumerate(embeddings):
                if other_idx == idx or assigned[other_idx] != -1:
                    continue
                similarity = cosine_similarity(emb, other_emb)
                if similarity >= self.config.similarity_threshold:
                    members.append(other_idx)
                    assigned[other_idx] = cluster_id
            assigned[idx] = cluster_id
            clusters.append(
                Cluster(
                    cluster_id=str(cluster_id),
                    question_ids=[questions[i].candidate.question_id for i in members],
                    centroid=_mean_vector([embeddings[i] for i in members]),
                )
            )
        return clusters


@dataclass(slots=True)
class ClusterSummarizer:
    """Uses an LLM to describe clusters for downstream curation."""

    client: LLMClient

    def summarize(self, questions: list[ValidatedQuestion], cluster: Cluster) -> ClusterSummary:
        question_map = {q.candidate.question_id: q for q in questions}
        context = "\n".join(question_map[qid].candidate.question for qid in cluster.question_ids)
        prompt = (
            "Summarize the shared topic of the following multiple-choice questions.\n"
            "Provide a short label and pick the most representative question id.\n"
            f"Questions:\n{context}\n"
            "Return JSON with fields summary and representative_id."
        )
        response = self.client.complete(prompt, temperature=0)
        try:
            payload = json.loads(response.content)
        except Exception:  # pragma: no cover - guardrail
            payload = {"summary": response.content.strip(), "representative_id": cluster.question_ids[0]}
        return ClusterSummary(
            cluster_id=cluster.cluster_id,
            summary=payload.get("summary", ""),
            representative_question_id=payload.get("representative_id", cluster.question_ids[0]),
        )


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""

    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _mean_vector(vectors: Iterable[list[float]]) -> list[float]:
    vectors = list(vectors)
    if not vectors:
        return []
    length = len(vectors[0])
    sums = [0.0] * length
    for vector in vectors:
        for i, value in enumerate(vector):
            sums[i] += value
    return [value / len(vectors) for value in sums]

