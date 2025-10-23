"""Configuration objects for the AskBench pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence


@dataclass(slots=True)
class IngestionConfig:
    """Parameters controlling document loading and chunking."""

    input_paths: Sequence[Path]
    max_chunk_tokens: int = 512
    chunk_overlap_tokens: int = 64
    page_level_metadata: bool = True


@dataclass(slots=True)
class GenerationConfig:
    """Settings for LLM-based question generation."""

    llm_model: str
    temperature: float = 0.2
    top_p: float = 0.95
    max_questions_per_chunk: int = 5
    multi_select: bool = False
    use_structured_outputs: bool = True


@dataclass(slots=True)
class ValidationConfig:
    """Settings for automatic validation heuristics."""

    min_option_count: int = 4
    require_rationale: bool = True
    correctness_check_mode: Literal["self_consistency", "reask", "none"] = "self_consistency"
    min_quality_score: float = 0.7


@dataclass(slots=True)
class ClusteringConfig:
    """Settings for embedding-based clustering and summarization."""

    embedding_model: str
    similarity_threshold: float = 0.8
    min_cluster_size: int = 2


@dataclass(slots=True)
class ExportConfig:
    """Settings for dataset export."""

    output_path: Path
    format: Literal["jsonl", "csv"] = "jsonl"
    include_metadata: bool = True


@dataclass(slots=True)
class PipelineConfig:
    """Aggregates all component configuration."""

    ingestion: IngestionConfig
    generation: GenerationConfig
    validation: ValidationConfig
    clustering: ClusteringConfig
    export: ExportConfig
    working_dir: Path = field(default_factory=lambda: Path(".askbench"))

