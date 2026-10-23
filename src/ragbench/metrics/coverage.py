"""Coverage of gold facts, which guards groundedness against being gamed by short answers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel

from ragbench.metrics.judge import Judge
from ragbench.prompts import is_refusal, render
from ragbench.strategies.base import CallRecord


class _Item(BaseModel):
    fact_id: int
    covered: bool


class _Coverage(BaseModel):
    coverage: list[_Item]


@dataclass
class CoverageResult:
    covered: list[bool] = field(default_factory=list)  # one per gold fact, in order
    judge_calls: list[CallRecord] = field(default_factory=list)

    @property
    def coverage(self) -> float | None:
        """C = covered facts / gold facts. None when the question has no gold facts."""
        return sum(self.covered) / len(self.covered) if self.covered else None


def score_coverage(
    question: str, answer: str, gold_facts: Sequence[str], judge: Judge
) -> CoverageResult:
    if not gold_facts:
        return CoverageResult()
    if is_refusal(answer) and len(answer) <= 300:
        return CoverageResult(covered=[False] * len(gold_facts))  # a refusal states no fact

    numbered = "\n".join(f"{i}. {fact}" for i, fact in enumerate(gold_facts, start=1))
    parsed, calls = judge.ask(
        "coverage",
        render("judge.coverage", question=question, answer=answer, numbered_gold_facts=numbered),
        _Coverage,
    )
    flags = {item.fact_id: item.covered for item in parsed.coverage}
    # a fact the judge did not mention is treated as not covered
    return CoverageResult(
        covered=[flags.get(i, False) for i in range(1, len(gold_facts) + 1)], judge_calls=calls
    )


def faithful_coverage(groundedness: float | None, coverage: float | None) -> float | None:
    """FCS: harmonic mean of groundedness and coverage, so an answer must be both to score well."""
    if groundedness is None or coverage is None:
        return None
    if groundedness == 0 or coverage == 0:
        return 0.0
    return 2 * groundedness * coverage / (groundedness + coverage)


def abstention_correct(expected_abstain: bool, abstained: bool) -> bool:
    """For out-of-corpus questions, declining is success; for answerable ones it is a miss."""
    return expected_abstain == abstained


class _SupportItem(BaseModel):
    fact_id: int
    supported: bool


class _Support(BaseModel):
    support: list[_SupportItem]


@dataclass
class ContextSupportResult:
    supported: list[bool] = field(default_factory=list)  # one per gold fact, in order
    judge_calls: list[CallRecord] = field(default_factory=list)

    @property
    def recall(self) -> float | None:
        """Share of gold facts the retrieved passages contain (a ceiling on achievable coverage)."""
        return sum(self.supported) / len(self.supported) if self.supported else None


def score_context_support(
    question: str, context: str, gold_facts: Sequence[str], judge: Judge
) -> ContextSupportResult:
    """Does the retrieved context contain each gold fact? Separates retrieval from generation."""
    if not gold_facts:
        return ContextSupportResult()
    numbered = "\n".join(f"{i}. {fact}" for i, fact in enumerate(gold_facts, start=1))
    parsed, calls = judge.ask(
        "context_support",
        render(
            "judge.context_support",
            question=question,
            context=context,
            numbered_gold_facts=numbered,
        ),
        _Support,
    )
    flags = {item.fact_id: item.supported for item in parsed.support}
    return ContextSupportResult(
        supported=[flags.get(i, False) for i in range(1, len(gold_facts) + 1)], judge_calls=calls
    )
