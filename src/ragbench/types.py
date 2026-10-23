"""Core data types shared across ingestion, retrieval and (later) generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # deterministic: "<doc_id>:p<page>:c<index>"
    doc_id: str
    doc_title: str
    role: str  # "core" | "distractor"
    page: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int  # 1-based
    similarity: float  # cosine similarity in [-1, 1]
    chunk: Chunk
