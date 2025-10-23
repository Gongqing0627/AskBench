"""High-level orchestration for the AskBench QnA pipeline."""
from __future__ import annotations

from typing import Iterable

from .config import PipelineConfig
from .domain.models import ClusterSummary, ExportRecord, ValidatedQuestion
from .exporters.writer import ExportWriter
from .generation.generator import QuestionGenerator
from .ingestion.loader import iter_document_chunks
from .llm.base import LLMClient
from .validation.validators import CandidateValidator, SelfConsistencyScorer
from .clustering.clusterer import ClusterSummarizer, EmbeddingClusterer


class Pipeline:
    """Coordinates the full TeleQnA-style automated pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        llm_for_generation: LLMClient,
        llm_for_validation: LLMClient,
        llm_for_clustering: LLMClient,
    ) -> None:
        self.config = config
        self.generator = QuestionGenerator(llm_for_generation, config.generation)
        scorer = SelfConsistencyScorer(llm_for_validation)
        self.validator = CandidateValidator(config.validation, scorer)
        self.clusterer = EmbeddingClusterer(llm_for_clustering, config.clustering)
        self.summarizer = ClusterSummarizer(llm_for_clustering)
        self.exporter = ExportWriter(config.export)
        self.working_dir = config.working_dir
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        chunks = list(iter_document_chunks(self.config.ingestion))
        candidates = list(self.generator.bulk_generate(chunks))
        validated = list(self.validator.filter(candidates))
        clusters = self.clusterer.cluster(validated)
        summaries = {cluster.cluster_id: self.summarizer.summarize(validated, cluster) for cluster in clusters}
        question_cluster_map = {
            qid: cluster.cluster_id for cluster in clusters for qid in cluster.question_ids
        }
        records = list(self._build_export_records(validated, summaries, question_cluster_map))
        self.exporter.write(records)

    def _build_export_records(
        self,
        validated: Iterable[ValidatedQuestion],
        summaries: dict[str, ClusterSummary],
        question_cluster_map: dict[str, str],
    ) -> Iterable[ExportRecord]:
        for item in validated:
            cluster_id = question_cluster_map.get(item.candidate.question_id)
            cluster_summary = summaries.get(cluster_id) if cluster_id else None
            metadata = None
            if self.config.export.include_metadata:
                metadata = {
                    "cluster_id": cluster_id,
                    "cluster_summary": cluster_summary.summary if cluster_summary else None,
                    "validation_score": item.validation.score,
                    "warnings": item.validation.warnings,
                }
            yield ExportRecord(
                question_id=item.candidate.question_id,
                question=item.candidate.question,
                options=item.candidate.options,
                correct_answer_index=item.candidate.correct_answer_index,
                rationale=item.candidate.rationale,
                source_documents=item.candidate.source_chunk_ids,
                metadata=metadata,
            )

