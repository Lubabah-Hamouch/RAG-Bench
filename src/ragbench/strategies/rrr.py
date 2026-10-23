"""RRR (Retrieve-Read-Refine): shared retrieval, then three sequential calls."""

from __future__ import annotations

from collections.abc import Sequence

from ragbench.llm.client import ChatClient
from ragbench.prompts import render
from ragbench.strategies.base import Strategy, StrategyResult, format_context, user_only
from ragbench.types import RetrievedChunk


class RRR(Strategy):
    name = "rrr"
    label = "RRR"

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

        facts = rec.call(
            "identify",
            user_only(render("rrr.identify", context=context, question=question)),
        )
        draft = rec.call(
            "synthesize",
            user_only(
                render("rrr.synthesize", context=context, question=question, stage1_output=facts)
            ),
        )
        final = rec.call(
            "refine",
            user_only(
                render("rrr.refine", context=context, question=question, stage2_output=draft)
            ),
        )
        return StrategyResult(self.name, final, rec.trace, rec.calls)
