"""Dataset export utilities."""
from __future__ import annotations

import csv
import json
from typing import Iterable

from ..config import ExportConfig
from ..domain.models import ExportRecord


class ExportWriter:
    """Writes validated QnA records to disk in various formats."""

    def __init__(self, config: ExportConfig) -> None:
        self.config = config
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, records: Iterable[ExportRecord]) -> None:
        match self.config.format:
            case "jsonl":
                self._write_jsonl(records)
            case "csv":
                self._write_csv(records)
            case _ as fmt:
                raise ValueError(f"Unsupported export format: {fmt}")

    def _write_jsonl(self, records: Iterable[ExportRecord]) -> None:
        with self.config.output_path.open("w", encoding="utf-8") as fh:
            for record in records:
                payload = record.__dict__ if self.config.include_metadata else {
                    key: value for key, value in record.__dict__.items() if key != "metadata"
                }
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_csv(self, records: Iterable[ExportRecord]) -> None:
        fieldnames = ["question_id", "question", "options", "correct_answer_index", "rationale", "source_documents"]
        if self.config.include_metadata:
            fieldnames.append("metadata")
        with self.config.output_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                row = record.__dict__.copy()
                row["options"] = json.dumps(record.options, ensure_ascii=False)
                row["source_documents"] = json.dumps(record.source_documents, ensure_ascii=False)
                if not self.config.include_metadata:
                    row.pop("metadata", None)
                writer.writerow(row)

