"""Turn benchmark/live records into the API's response models.

Replay reads records from runs.jsonl and live builds the same record shape with
`ragbench.benchmark.build_record`, so one code path produces both and the two cannot drift.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ragbench.api.schemas import (
    AnswerOut,
    ChunkOut,
    ClaimOut,
    ConsistencyOut,
    CoverageOut,
    GoldFactOut,
    GroundednessOut,
    LexicalOut,
    MetricsOut,
    QuestionOut,
    StageOut,
    TrialAnswerOut,
)
from ragbench.metrics.cost import Pricing, total_cost
from ragbench.metrics.lexical import lexical_metrics
from ragbench.questions import Question
from ragbench.strategies.base import CallRecord
from ragbench.strategies.registry import REGISTRY


def question_out(q: Question, *, free_text: bool = False) -> QuestionOut:
    return QuestionOut(
        id=q.id,
        type="free_text" if free_text else q.type,
        question=q.question,
        expected_abstain=q.expected_abstain,
        gold_facts=[GoldFactOut(text=g.text, doc_id=g.doc_id, page=g.page) for g in q.gold_facts],
    )


def chunk_out(chunk: Mapping[str, Any], titles: Mapping[str, str]) -> ChunkOut:
    return ChunkOut(
        rank=chunk["rank"],
        similarity=chunk["similarity"],
        chunk_id=chunk["chunk_id"],
        doc_id=chunk["doc_id"],
        doc_title=titles.get(chunk["doc_id"]),
        role=chunk["role"],
        page=chunk["page"],
        text=chunk["text"],
    )


def _lexical(rec: Mapping[str, Any]) -> LexicalOut:
    lex = rec.get("lexical")
    if lex:
        return LexicalOut(**lex)
    # unjudged answers carry no stored lexical metrics: compute them from the answer text
    m = lexical_metrics(rec["answer"])
    return LexicalOut(
        chars=m.chars,
        words=m.words,
        sentences=m.sentences,
        unique_vocab=m.unique_vocab,
        avg_word_length=m.avg_word_length,
        words_per_sentence=m.words_per_sentence,
        ttr=m.ttr,
        mattr=m.mattr,
    )


def metrics_from_record(
    rec: Mapping[str, Any],
    q: Question,
    pricing: Pricing,
    consistency: ConsistencyOut | None = None,
) -> MetricsOut:
    calls = [CallRecord(**c) for c in rec["calls"]]
    judged = bool(rec.get("judged"))
    grounded = coverage = None
    fcs: float | None = None
    judge_tokens = judge_cost = None
    if judged:
        g = rec["groundedness"]
        claims = [
            ClaimOut(text=t, label=label, reason=reason or "", evidence_passage=ev)
            for t, label, reason, ev in zip(
                g["claims"], g["labels"], g["reasons"], g["evidence_passages"], strict=True
            )
        ]
        grounded = GroundednessOut(
            G=g["G"],
            H=g["H"],
            X=g["X"],
            n_claims=len(g["claims"]),
            n_supported=g["n_supported"],
            n_unsupported=g["n_unsupported"],
            n_contradicted=g["n_contradicted"],
            abstained=bool(g["abstained"]),
            claims=claims,
        )
        if q.gold_facts:
            cov = rec["coverage"]
            coverage = CoverageOut(C=cov["C"], covered=cov["covered"], gold_facts=q.gold_fact_texts)
        fcs = rec.get("fcs")
        refused = bool(rec.get("abstained") or g["abstained"])
        if q.gold_facts and not q.expected_abstain and refused:
            fcs = 0.0  # declining an answerable question is a miss, as in the benchmark report
        judge_tokens = rec.get("judge_prompt_tokens", 0) + rec.get("judge_completion_tokens", 0)
        judge_cost = rec.get("judge_cost_usd")
    return MetricsOut(
        judged=judged,
        groundedness=grounded,
        coverage=coverage,
        fcs=fcs,
        lexical=_lexical(rec),
        latency_s=rec["gen_latency_s"],
        prompt_tokens=rec["gen_prompt_tokens"],
        completion_tokens=rec["gen_completion_tokens"],
        calls=len(calls),
        cost_usd=total_cost(calls, pricing),
        judge_tokens=judge_tokens,
        judge_cost_usd=judge_cost,
        consistency=consistency,
    )


def answer_from_record(
    rec: Mapping[str, Any],
    q: Question,
    pricing: Pricing,
    *,
    consistency: ConsistencyOut | None = None,
    other_trials: Sequence[TrialAnswerOut] = (),
) -> AnswerOut:
    strategy = rec["strategy"]
    cls = REGISTRY[strategy]
    return AnswerOut(
        strategy=strategy,
        label=cls.label,
        uses_retrieval=cls.uses_retrieval,
        answer=rec["answer"],
        critique=rec.get("critique"),
        trace=[StageOut(name=s["stage"], text=s["text"]) for s in rec.get("trace", [])],
        abstained=bool(rec["abstained"]),
        trial=rec["trial"],
        metrics=metrics_from_record(rec, q, pricing, consistency),
        other_trials=list(other_trials),
    )
