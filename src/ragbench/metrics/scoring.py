"""Score one strategy answer: lexical, groundedness, coverage, FCS, latency and cost."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ragbench.metrics.cost import Pricing, cost_per_supported_claim, total_cost
from ragbench.metrics.coverage import CoverageResult, faithful_coverage, score_coverage
from ragbench.metrics.groundedness import GroundednessResult, score_groundedness
from ragbench.metrics.judge import Judge
from ragbench.metrics.lexical import LexicalMetrics, lexical_metrics
from ragbench.strategies.base import StrategyResult
from ragbench.types import RetrievedChunk


@dataclass
class AnswerMetrics:
    strategy: str
    lexical: LexicalMetrics
    groundedness: GroundednessResult
    coverage: CoverageResult
    fcs: float | None
    abstained: bool
    gen_latency_s: float
    gen_prompt_tokens: int
    gen_completion_tokens: int
    gen_cost_usd: float | None
    judge_prompt_tokens: int
    judge_completion_tokens: int
    judge_cost_usd: float | None  # tracked separately; never added to a strategy's cost
    cost_per_supported_claim: float | None


def evaluate_answer(
    question: str,
    result: StrategyResult,
    chunks: Sequence[RetrievedChunk],
    *,
    judge: Judge,
    gold_facts: Sequence[str],
    pricing: Pricing,
    mattr_window: int = 25,
) -> AnswerMetrics:
    """Metrics are computed on `result.answer` only (for Self-RAG that excludes the critique)."""
    grounded = score_groundedness(question, result.answer, chunks, judge)
    covered = score_coverage(question, result.answer, gold_facts, judge)
    judge_calls = grounded.judge_calls + covered.judge_calls
    gen_cost = total_cost(result.calls, pricing)
    return AnswerMetrics(
        strategy=result.strategy,
        lexical=lexical_metrics(result.answer, mattr_window),
        groundedness=grounded,
        coverage=covered,
        fcs=faithful_coverage(grounded.groundedness, covered.coverage),
        abstained=grounded.abstained,
        gen_latency_s=result.latency_s,
        gen_prompt_tokens=result.prompt_tokens,
        gen_completion_tokens=result.completion_tokens,
        gen_cost_usd=gen_cost,
        judge_prompt_tokens=sum(c.prompt_tokens for c in judge_calls),
        judge_completion_tokens=sum(c.completion_tokens for c in judge_calls),
        judge_cost_usd=total_cost(judge_calls, pricing),
        cost_per_supported_claim=cost_per_supported_claim(gen_cost, grounded.n_supported),
    )
