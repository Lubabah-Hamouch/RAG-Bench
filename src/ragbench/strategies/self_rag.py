"""Self-RAG (prompted): generate, self-critique against the context, then refine.

This emulates the *idea* of Self-RAG (critique relevance, support and utility) with prompts. It is
not the fine-tuned model from the paper, which emits learned reflection tokens.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ragbench.llm.client import ChatClient
from ragbench.prompts import render
from ragbench.strategies.base import Strategy, StrategyResult, format_context, user_only
from ragbench.types import RetrievedChunk

_FIELDS = {
    "answerable": r"Answerable from context:\s*(.+)",
    "factual_accuracy": r"Factual accuracy:\s*(.+)",
    "completeness": r"Completeness:\s*(.+)",
    "utility": r"Utility \(1-5\):\s*(\d)",
}


def parse_critique(text: str) -> dict[str, str | list[str]]:
    """Pull the labelled fields out of a critique (missing fields are simply absent)."""
    out: dict[str, str | list[str]] = {}
    for key, pattern in _FIELDS.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            out[key] = m.group(1).strip()
    out["passage_relevance"] = re.findall(r"^\[\d\]\s*(Relevant|Irrelevant)", text, re.M | re.I)
    return out


class SelfRAG(Strategy):
    name = "self_rag"
    label = "Self-RAG (prompted)"

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

        draft = rec.call(
            "draft",
            user_only(render("selfrag.draft", context=context, question=question)),
        )
        critique = rec.call(
            "critique",
            user_only(
                render("selfrag.critique", context=context, question=question, stage1_output=draft)
            ),
        )
        final = rec.call(
            "refine",
            user_only(
                render(
                    "selfrag.refine",
                    context=context,
                    question=question,
                    stage1_output=draft,
                    stage2_output=critique,
                )
            ),
        )
        return StrategyResult(self.name, final, rec.trace, rec.calls, critique=critique)
