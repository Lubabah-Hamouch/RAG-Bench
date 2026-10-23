"""Response models: the typed contract shared with the (future) frontend."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class GoldFactOut(BaseModel):
    text: str
    doc_id: str
    page: int


class QuestionOut(BaseModel):
    id: str
    type: Literal["lookup", "multi_hop", "out_of_corpus", "free_text"]
    question: str
    expected_abstain: bool = False
    gold_facts: list[GoldFactOut] = []


class StrategyInfo(BaseModel):
    name: str
    label: str
    uses_retrieval: bool
    calls: int  # model calls per answer


class ChunkOut(BaseModel):
    rank: int
    similarity: float
    chunk_id: str
    doc_id: str
    doc_title: str | None = None
    role: str
    page: int
    text: str


class RetrievalOut(BaseModel):
    """The top-k passages, retrieved once and shared by every strategy."""

    chunks: list[ChunkOut]
    context_recall_judge: float | None = None  # share of gold facts the context contains


class ClaimOut(BaseModel):
    text: str
    label: Literal["supported", "unsupported", "contradicted"]
    reason: str = ""
    evidence_passage: int | None = None


class GroundednessOut(BaseModel):
    G: float | None  # supported / claims; None when the answer made no claims (abstained)
    H: float | None  # hallucination rate, 1 - G
    X: float | None  # contradiction rate
    n_claims: int
    n_supported: int
    n_unsupported: int
    n_contradicted: int
    abstained: bool
    claims: list[ClaimOut]


class CoverageOut(BaseModel):
    C: float | None
    covered: list[bool]
    gold_facts: list[str]


class LexicalOut(BaseModel):
    chars: int
    words: int
    sentences: int
    unique_vocab: int
    avg_word_length: float | None = None
    words_per_sentence: float | None = None
    ttr: float | None = None
    mattr: float | None = None  # None for answers shorter than the 25-word window


class ConsistencyOut(BaseModel):
    """Replay only: needs several trials of the same question."""

    semantic: float | None
    structural: float | None
    n_samples: int


class MetricsOut(BaseModel):
    judged: bool
    groundedness: GroundednessOut | None = None
    coverage: CoverageOut | None = None
    # FCS: harmonic mean of G and C. A refusal on an answerable question counts as 0.0.
    fcs: float | None = None
    lexical: LexicalOut
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    calls: int
    cost_usd: float | None = None  # generation only, list prices
    judge_tokens: int | None = None
    judge_cost_usd: float | None = None
    consistency: ConsistencyOut | None = None  # always None for live runs (single trial)


class StageOut(BaseModel):
    name: str
    text: str


class TrialAnswerOut(BaseModel):
    trial: int
    answer: str


class AnswerOut(BaseModel):
    strategy: str
    label: str
    uses_retrieval: bool
    answer: str  # the final answer; metrics are computed on this text only
    critique: str | None = None  # Self-RAG: shown to the user, never scored as the answer
    trace: list[StageOut] = []
    abstained: bool
    trial: int = 0
    metrics: MetricsOut
    other_trials: list[TrialAnswerOut] = []  # replay: the same question's other trials


class QuestionResultOut(BaseModel):
    mode: Literal["replay", "live"]
    n_trials: int
    judged_trials: int
    models: dict[str, str]
    question: QuestionOut
    retrieval: RetrievalOut
    answers: dict[str, AnswerOut]


class HealthOut(BaseModel):
    status: Literal["ok"]
    replay_available: bool
    questions: int
    models: dict[str, str]
    require_https: bool
    live_ready: bool  # false while the embedding model is still loading at startup
    live_error: str | None = None  # set if loading it failed (live runs are unavailable)


class ValidateOut(BaseModel):
    valid: bool
    generator_model: str
    generator_available: bool
    judge_model: str
    judge_available: bool
    message: str


class ErrorOut(BaseModel):
    error: str
    message: str
    retry_after_s: float | None = None


SummaryOut = dict[str, Any]
