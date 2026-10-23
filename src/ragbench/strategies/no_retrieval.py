"""Control condition: the same question with no retrieved context at all."""

from __future__ import annotations

from collections.abc import Sequence

from ragbench.llm.client import ChatClient
from ragbench.prompts import render
from ragbench.strategies.base import Strategy, StrategyResult, system_user
from ragbench.types import RetrievedChunk


class NoRetrieval(Strategy):
    name = "no_retrieval"
    label = "No retrieval (control)"
    uses_retrieval = False

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
        answer = rec.call(
            "answer",
            system_user(render("control.system"), render("control.user", question=question)),
        )
        return StrategyResult(self.name, answer, rec.trace, rec.calls)
