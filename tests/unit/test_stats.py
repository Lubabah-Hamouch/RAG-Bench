import numpy as np
import pytest

from ragbench.metrics.stats import (
    _bootstrap_means,
    aggregate,
    bootstrap_ci,
    paired_difference,
    question_means,
)


def test_constant_values_give_a_zero_width_interval():
    est = bootstrap_ci([0.7] * 13, n_resamples=500)
    assert (
        est.mean == pytest.approx(0.7)
        and est.lo == pytest.approx(0.7)
        and est.hi == pytest.approx(0.7)
    )


def test_interval_brackets_the_mean_and_shrinks_with_more_questions():
    narrow = bootstrap_ci([0.4, 0.5, 0.6, 0.5] * 10, n_resamples=2000, seed=1)
    wide = bootstrap_ci([0.1, 0.9, 0.2, 0.8], n_resamples=2000, seed=1)
    for est in (narrow, wide):
        assert est.lo <= est.mean <= est.hi
    assert (narrow.hi - narrow.lo) < (wide.hi - wide.lo)


def test_bootstrap_is_reproducible_for_a_seed_and_differs_across_seeds():
    values = [0.1, 0.4, 0.5, 0.9, 0.3]
    assert bootstrap_ci(values, 1000, seed=3) == bootstrap_ci(values, 1000, seed=3)
    a = _bootstrap_means(np.asarray(values), 1000, seed=3)
    b = _bootstrap_means(np.asarray(values), 1000, seed=4)
    assert not np.array_equal(a, b)


def test_known_interval_for_two_values():
    # resampling [0, 1] gives means in {0, 0.5, 1}; a 95% interval spans the full range
    est = bootstrap_ci([0.0, 1.0], n_resamples=5000, seed=0)
    assert (est.lo, est.hi) == (0.0, 1.0) and est.mean == 0.5


def test_empty_input_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_question_means_average_trials_and_skip_undefined_values():
    out = question_means({"q1": [1.0, 0.0, None], "q2": [None, None, None], "q3": [0.5]})
    assert out == {"q1": 0.5, "q3": 0.5}


def test_aggregate_uses_questions_not_trials_as_the_unit():
    # q1 has 3 trials, q2 has 1: each question still counts once
    est = aggregate({"q1": [0.0, 0.0, 0.0], "q2": [1.0]}, n_resamples=200)
    assert est is not None and est.mean == 0.5 and est.n == 2


def test_aggregate_returns_none_when_nothing_is_defined():
    assert aggregate({"q1": [None, None]}) is None


def test_paired_difference_on_shared_questions_only():
    a = {"q1": 0.9, "q2": 0.8, "q3": 0.7, "only_a": 1.0}
    b = {"q1": 0.5, "q2": 0.4, "q3": 0.3, "only_b": 0.0}
    cmp = paired_difference(a, b, n_resamples=500)
    assert cmp is not None and cmp.diff.n == 3
    assert cmp.diff.mean == pytest.approx(0.4) and cmp.diff.lo > 0


def test_wilcoxon_detects_a_consistent_improvement():
    a = {f"q{i}": 0.5 + 0.01 * i + 0.2 for i in range(10)}
    b = {f"q{i}": 0.5 + 0.01 * i for i in range(10)}
    cmp = paired_difference(a, b, n_resamples=200)
    assert cmp is not None and cmp.wilcoxon_p is not None and cmp.wilcoxon_p < 0.01


def test_identical_strategies_have_no_test_statistic():
    cmp = paired_difference({"q1": 0.5, "q2": 0.6}, {"q1": 0.5, "q2": 0.6}, n_resamples=200)
    assert cmp is not None and cmp.wilcoxon_p is None and cmp.diff.mean == 0


def test_no_shared_questions_gives_none():
    assert paired_difference({"q1": 1.0}, {"q2": 1.0}) is None
