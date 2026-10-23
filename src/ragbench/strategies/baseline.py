"""Baseline: one call, a direct extractive answer."""

from __future__ import annotations

from collections.abc import Sequence

from ragbench.llm.client import ChatClient
from ragbench.prompts import render
from ragbench.strategies.base import Strategy, StrategyResult, format_context, system_user
from ragbench.types import RetrievedChunk


class Baseline(Strategy):
    name = "baseline"
    label = "Baseline"

    def run(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        *,
        client: ChatClient,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
        trial: int = 0,
    ) -> StrategyResult:
        rec = self._recorder(client, temperature, max_tokens, seed, trial)
        context = format_context(chunks)
        answer = rec.call(
            "answer",
            system_user(
                render("baseline.system"),
                render("baseline.user", context=context, question=question),
            ),
        )
        return StrategyResult(self.name, answer, rec.trace, rec.calls)
