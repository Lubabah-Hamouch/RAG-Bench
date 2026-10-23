import math

import pytest

from ragbench.budget import (
    BudgetConfig,
    Limits,
    ModelLoad,
    Profile,
    load_budget_config,
    project,
    render,
)

LIMITS = Limits(rpm=30, rpd=1000, tpm=8000, tpd=200_000)


def cfg(judge_tokens=3000):
    a = Profile(calls=1, prompt_tokens=800, completion_tokens=200)  # 1000 tokens
    b = Profile(calls=1, prompt_tokens=800, completion_tokens=100)  # 900 tokens
    return BudgetConfig(
        generator_model="gen",
        judge_model="jdg",
        generator_limits=LIMITS,
        judge_limits=LIMITS,
        generation={"s1": {"answerable": a, "abstain": b}, "s2": {"answerable": a, "abstain": b}},
        judge=Profile(calls=3, prompt_tokens=judge_tokens - 600, completion_tokens=600),
        conditions=["s1", "s2"],
    )


def test_generation_arithmetic_without_margin():
    p = project(cfg(), n_questions=13, n_abstain_questions=1, trials=3, margin=0.0)
    # per condition and trial: 12 answerable x 1000 + 1 abstain x 900 = 12,900 tokens
    assert p.generator.tokens == pytest.approx(2 * 3 * 12_900)
    assert p.generator.calls == pytest.approx(2 * 3 * 13)
    assert p.answers_generated == 3 * 13 * 2


def test_judge_arithmetic_skips_refusals_on_out_of_corpus_questions():
    p = project(cfg(), trials=3, margin=0.0, abstain_judged_conditions=1)
    assert p.answers_judged == 3 * (12 * 2 + 1 * 1)
    assert p.judge.tokens == pytest.approx(p.answers_judged * 3000)
    assert p.judge.calls == pytest.approx(p.answers_judged * 3)


def test_judging_fewer_trials_cuts_judge_load_but_not_generation():
    full = project(cfg(), trials=3, margin=0.0)
    one = project(cfg(), trials=3, judge_trials=1, margin=0.0)
    assert one.generator.tokens == full.generator.tokens
    assert one.judge.tokens == pytest.approx(full.judge.tokens / 3)


def test_margin_scales_both_models():
    base = project(cfg(), margin=0.0)
    padded = project(cfg(), margin=0.25)
    assert padded.generator.tokens == pytest.approx(base.generator.tokens * 1.25)
    assert padded.judge.calls == pytest.approx(base.judge.calls * 1.25)


def test_daily_fit_and_days_needed():
    load = ModelLoad("m", calls=500, tokens=150_000, limits=LIMITS)
    assert load.fits_in_one_day and load.days == 1
    over = ModelLoad("m", calls=500, tokens=450_000, limits=LIMITS)
    assert not over.fits_in_one_day and over.days == math.ceil(450_000 / 200_000)
    many_calls = ModelLoad("m", calls=2500, tokens=100_000, limits=LIMITS)
    assert many_calls.days == 3  # the request cap binds, not the token cap


def test_min_minutes_uses_the_tighter_per_minute_limit():
    token_bound = ModelLoad("m", calls=10, tokens=80_000, limits=LIMITS)
    assert token_bound.min_minutes == pytest.approx(10.0)  # 80k / 8k per minute
    request_bound = ModelLoad("m", calls=300, tokens=1000, limits=LIMITS)
    assert request_bound.min_minutes == pytest.approx(10.0)  # 300 / 30 per minute


def test_render_lists_every_scenario():
    table = render({"A": project(cfg()), "B": project(cfg(), judge_trials=1)})
    assert "| A |" in table and "| B |" in table and table.count("\n") == 3


def test_shipped_budget_config_loads_and_covers_every_strategy():
    c = load_budget_config()
    assert c.conditions == ["baseline", "structured", "rrr", "self_rag", "no_retrieval"]
    assert c.generator_limits.tpd == 200_000 and c.judge_limits.tpm == 8000
    assert c.judge.calls > 0 and c.judge.tokens > 0
