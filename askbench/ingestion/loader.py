"""Document loading and chunking utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from ..config import IngestionConfig
from ..domain.models import Document, DocumentChunk


class DocumentLoader:
    """Base class for converting input files into :class:`Document` objects."""

    def load(self, path: Path) -> Document:
        raise NotImplementedError


class PlainTextLoader(DocumentLoader):
    """Simple loader that treats the file as plain UTF-8 text."""

    def load(self, path: Path) -> Document:
        return Document(doc_id=path.stem, title=path.name, content=path.read_text(encoding="utf-8"))


class Chunker:
    """Breaks documents into semantically coherent chunks."""

    def __init__(self, config: IngestionConfig) -> None:
        self.config = config

    def chunk(self, document: Document) -> Iterable[DocumentChunk]:
        text = document.content
        max_tokens = self.config.max_chunk_tokens
        overlap = self.config.chunk_overlap_tokens
        words = text.split()
        chunk_id = 0
        step = max_tokens - overlap if max_tokens > overlap else max_tokens
        for start in range(0, len(words), step):
            end = min(start + max_tokens, len(words))
            chunk_text = " ".join(words[start:end])
            if not chunk_text.strip():
                continue
            chunk_id += 1
            yield DocumentChunk(
                document_id=document.doc_id,
                chunk_id=f"{document.doc_id}-{chunk_id}",
                text=chunk_text,
                metadata=document.metadata,
            )


def iter_document_chunks(config: IngestionConfig, loader: DocumentLoader | None = None) -> Iterator[DocumentChunk]:
    """Convenience generator that yields chunks for all configured inputs."""

    loader = loader or PlainTextLoader()
    chunker = Chunker(config)

    for path in config.input_paths:
        document = loader.load(Path(path))
        yield from chunker.chunk(document)

