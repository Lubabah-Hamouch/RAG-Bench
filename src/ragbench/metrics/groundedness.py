"""Claim-level groundedness: extract atomic claims, verify each against the retrieved passages."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, field_validator

from ragbench.metrics.judge import Judge
from ragbench.prompts import is_refusal, render
from ragbench.strategies.base import CallRecord, format_context
from ragbench.types import RetrievedChunk

Label = Literal["supported", "unsupported", "contradicted"]
REFUSAL_MAX_CHARS = 300  # a short answer containing the refusal sentence is a refusal, not a claim


class _Claims(BaseModel):
    claims: list[str]


class ClaimVerdict(BaseModel):
    claim_id: int
    label: Label
    evidence_passage: int | None = None
    reason: str = ""

    @field_validator("label", mode="before")
    @classmethod
    def _normalise(cls, v: object) -> object:
        return v.strip().lower() if isinstance(v, str) else v


class _Verdicts(BaseModel):
    verdicts: list[ClaimVerdict]


@dataclass
class GroundednessResult:
    claims: list[str] = field(default_factory=list)
    verdicts: list[ClaimVerdict] = field(default_factory=list)  # one per claim, in claim order
    n_missing_verdicts: int = 0  # claims the judge skipped; counted as unsupported
    abstained: bool = False  # the answer made no factual claims (it declined to answer)
    judge_calls: list[CallRecord] = field(default_factory=list)

    @property
    def n_claims(self) -> int:
        return len(self.claims)

    def _count(self, label: Label) -> int:
        return sum(v.label == label for v in self.verdicts)

    @property
    def n_supported(self) -> int:
        return self._count("supported")

    @property
    def n_unsupported(self) -> int:
        return self._count("unsupported")

    @property
    def n_contradicted(self) -> int:
        return self._count("contradicted")

    @property
    def groundedness(self) -> float | None:
        """G = supported / claims. None when there are no claims (undefined, not zero)."""
        return self.n_supported / self.n_claims if self.n_claims else None

    @property
    def hallucination_rate(self) -> float | None:
        """H = (unsupported + contradicted) / claims = 1 - G."""
        g = self.groundedness
        return None if g is None else 1.0 - g

    @property
    def contradiction_rate(self) -> float | None:
        """X = contradicted / claims, the worst failure, reported on its own."""
        return self.n_contradicted / self.n_claims if self.n_claims else None


def _numbered(items: Sequence[str]) -> str:
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=1))


def score_groundedness(
    question: str,
    answer: str,
    chunks: Sequence[RetrievedChunk],
    judge: Judge,
) -> GroundednessResult:
    """Score one answer against the retrieved passages (always the same top-k, per decision D5)."""
    if is_refusal(answer) and len(answer) <= REFUSAL_MAX_CHARS:
        return GroundednessResult(abstained=True)  # saves two judge calls and cannot add claims

    extracted, calls = judge.ask(
        "extract_claims",
        render("judge.extract_claims", question=question, answer=answer),
        _Claims,
    )
    claims = [c.strip() for c in extracted.claims if c.strip()]
    if not claims:
        return GroundednessResult(abstained=True, judge_calls=calls)

    verdicts_out, verify_calls = judge.ask(
        "verify_claims",
        render(
            "judge.verify_claims",
            context=format_context(chunks),
            numbered_claims=_numbered(claims),
        ),
        _Verdicts,
    )
    by_id = {v.claim_id: v for v in verdicts_out.verdicts}
    aligned: list[ClaimVerdict] = []
    missing = 0
    for i in range(1, len(claims) + 1):
        verdict = by_id.get(i)
        if verdict is None:
            missing += 1
            verdict = ClaimVerdict(claim_id=i, label="unsupported", reason="no verdict returned")
        aligned.append(verdict)
    return GroundednessResult(
        claims=claims,
        verdicts=aligned,
        n_missing_verdicts=missing,
        judge_calls=calls + verify_calls,
    )
