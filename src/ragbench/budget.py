"""Project the benchmark's call and token volume against provider rate limits.

Per-answer token profiles are *measured* (see configs/budget.yaml) rather than guessed, and the
projection adds an explicit safety margin for what the profiles cannot capture (longer multi-hop
answers, retried or enlarged judge calls).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ragbench.config import repo_root


@dataclass(frozen=True)
class Profile:
    """Average usage for producing (or judging) one answer."""

    calls: float
    prompt_tokens: float
    completion_tokens: float

    @property
    def tokens(self) -> float:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class Limits:
    rpm: int
    rpd: int
    tpm: int
    tpd: int


@dataclass
class BudgetConfig:
    generator_model: str
    judge_model: str
    generator_limits: Limits
    judge_limits: Limits
    # generation[strategy][kind] where kind is "answerable" or "abstain"
    generation: dict[str, dict[str, Profile]]
    judge: Profile  # per judged answer (claim extraction + verification + coverage)
    conditions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelLoad:
    model: str
    calls: float
    tokens: float
    limits: Limits

    @property
    def tpd_use(self) -> float:
        return self.tokens / self.limits.tpd

    @property
    def rpd_use(self) -> float:
        return self.calls / self.limits.rpd

    @property
    def days(self) -> int:
        """Days needed if each daily quota can be used in full."""
        return max(1, math.ceil(max(self.tpd_use, self.rpd_use)))

    @property
    def fits_in_one_day(self) -> bool:
        return self.tpd_use <= 1.0 and self.rpd_use <= 1.0

    @property
    def min_minutes(self) -> float:
        """Lower bound on wall time from the per-minute limits alone."""
        return max(self.tokens / self.limits.tpm, self.calls / self.limits.rpm)


@dataclass(frozen=True)
class Projection:
    n_questions: int
    trials: int
    judge_trials: int
    margin: float
    answers_generated: int
    answers_judged: int
    generator: ModelLoad
    judge: ModelLoad


def _limits(raw: dict[str, int]) -> Limits:
    return Limits(raw["rpm"], raw["rpd"], raw["tpm"], raw["tpd"])


def _profile(raw: dict[str, float]) -> Profile:
    return Profile(raw["calls"], raw["prompt_tokens"], raw["completion_tokens"])


def load_budget_config(path: Path | None = None) -> BudgetConfig:
    path = path or repo_root() / "configs" / "budget.yaml"
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    generation = {
        strategy: {kind: _profile(p) for kind, p in kinds.items()}
        for strategy, kinds in raw["generation_per_answer"].items()
    }
    return BudgetConfig(
        generator_model=raw["generator"]["model"],
        judge_model=raw["judge"]["model"],
        generator_limits=_limits(raw["generator"]["limits"]),
        judge_limits=_limits(raw["judge"]["limits"]),
        generation=generation,
        judge=_profile(raw["judge_per_answer"]),
        conditions=list(generation),
    )


def project(
    cfg: BudgetConfig,
    *,
    n_questions: int = 13,
    n_abstain_questions: int = 1,
    trials: int = 3,
    judge_trials: int | None = None,
    margin: float = 0.25,
    abstain_judged_conditions: int = 1,
) -> Projection:
    """Projected load for the full benchmark.

    `judge_trials` is how many of the `trials` per (question, condition) get groundedness-judged
    (consistency needs all trials but no judge, so judging fewer is a real saving).
    On an out-of-corpus question the retrieval strategies refuse and refusals skip the judge, so
    only `abstain_judged_conditions` (the no-retrieval control) are judged there.
    """
    judge_trials = trials if judge_trials is None else judge_trials
    answerable = n_questions - n_abstain_questions
    n_cond = len(cfg.conditions)

    gen_calls = gen_tokens = 0.0
    for kinds in cfg.generation.values():
        for kind, count in (("answerable", answerable), ("abstain", n_abstain_questions)):
            gen_calls += trials * count * kinds[kind].calls
            gen_tokens += trials * count * kinds[kind].tokens

    judged = judge_trials * (answerable * n_cond + n_abstain_questions * abstain_judged_conditions)
    scale = 1.0 + margin
    return Projection(
        n_questions=n_questions,
        trials=trials,
        judge_trials=judge_trials,
        margin=margin,
        answers_generated=trials * n_questions * n_cond,
        answers_judged=judged,
        generator=ModelLoad(
            cfg.generator_model, gen_calls * scale, gen_tokens * scale, cfg.generator_limits
        ),
        judge=ModelLoad(
            cfg.judge_model,
            judged * cfg.judge.calls * scale,
            judged * cfg.judge.tokens * scale,
            cfg.judge_limits,
        ),
    )


def render(projections: dict[str, Projection]) -> str:
    """Markdown table comparing scenarios."""
    lines = [
        "| scenario | answers generated | answers judged | generator calls | generator tokens "
        "(% of daily) | judge calls | judge tokens (% of daily) | days needed | min. run time |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, p in projections.items():
        days = max(p.generator.days, p.judge.days)
        minutes = max(p.generator.min_minutes, p.judge.min_minutes)
        lines.append(
            f"| {name} | {p.answers_generated} | {p.answers_judged} | {p.generator.calls:,.0f} | "
            f"{p.generator.tokens:,.0f} ({p.generator.tpd_use:.0%}) | {p.judge.calls:,.0f} | "
            f"{p.judge.tokens:,.0f} ({p.judge.tpd_use:.0%}) | {days} | ~{minutes:,.0f} min |"
        )
    return "\n".join(lines)
