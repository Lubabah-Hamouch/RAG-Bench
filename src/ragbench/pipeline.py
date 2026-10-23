"""Run one question through the strategies with a single, shared retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ragbench.llm.client import ChatClient
from ragbench.strategies.base import StrategyResult
from ragbench.strategies.registry import ALL_NAMES, get_strategy
from ragbench.types import RetrievedChunk


class Retriever(Protocol):
    def query(self, text: str, top_k: int = ...) -> list[RetrievedChunk]: ...


@dataclass
class PipelineResult:
    question: str
    chunks: list[RetrievedChunk]
    results: dict[str, StrategyResult] = field(default_factory=dict)


def run_question(
    question: str,
    retriever: Retriever,
    client: ChatClient,
    *,
    strategies: Sequence[str] | None = None,
    top_k: int = 4,
    temperature: float = 0.5,
    max_tokens: int = 700,
    seed: int | None = None,
    trial: int = 0,
) -> PipelineResult:
    """Retrieve once, then hand the *same* chunks to every strategy (identical retrieval)."""
    chunks = retriever.query(question, top_k)
    out = PipelineResult(question=question, chunks=chunks)
    for name in strategies or ALL_NAMES:
        out.results[name] = get_strategy(name).run(
            question,
            chunks,
            client=client,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            trial=trial,
        )
    return out
