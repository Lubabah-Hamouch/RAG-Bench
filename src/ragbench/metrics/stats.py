"""Cluster-bootstrap confidence intervals. The unit of analysis is the question, not the trial."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sps


@dataclass(frozen=True)
class Estimate:
    mean: float
    lo: float
    hi: float
    n: int  # number of questions contributing


def question_means(per_question: Mapping[str, Sequence[float | None]]) -> dict[str, float]:
    """Average the trials within each question, ignoring undefined (None) values.

    A question with no defined value (for example every trial abstained) is dropped.
    """
    out: dict[str, float] = {}
    for qid, values in per_question.items():
        defined = [v for v in values if v is not None]
        if defined:
            out[qid] = float(np.mean(defined))
    return out


def _bootstrap_means(
    values: NDArray[np.float64], n_resamples: int, seed: int
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    return values[idx].mean(axis=1)


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> Estimate:
    """Percentile bootstrap CI of the mean, resampling questions with replacement."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("no values to bootstrap")
    means = _bootstrap_means(arr, n_resamples, seed)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Estimate(float(arr.mean()), float(lo), float(hi), int(arr.size))


def aggregate(
    per_question: Mapping[str, Sequence[float | None]],
    n_resamples: int = 10_000,
    seed: int = 0,
) -> Estimate | None:
    """Per-question trial means -> mean across questions with a 95% bootstrap CI."""
    means = question_means(per_question)
    if not means:
        return None
    return bootstrap_ci(list(means.values()), n_resamples, seed)


@dataclass(frozen=True)
class PairedComparison:
    diff: Estimate  # mean of (a - b) over shared questions, with CI
    wilcoxon_p: float | None  # None when fewer than 2 questions or all differences are zero


def paired_difference(
    a: Mapping[str, float],
    b: Mapping[str, float],
    n_resamples: int = 10_000,
    seed: int = 0,
) -> PairedComparison | None:
    """Compare two strategies on the questions where both have a defined value."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    diffs = np.array([a[q] - b[q] for q in shared], dtype=float)
    estimate = bootstrap_ci(diffs.tolist(), n_resamples, seed)
    p: float | None = None
    if len(diffs) >= 2 and np.any(diffs != 0):
        p = float(sps.wilcoxon(diffs).pvalue)
    return PairedComparison(estimate, p)
