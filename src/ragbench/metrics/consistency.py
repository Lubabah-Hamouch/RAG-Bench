"""Run-to-run consistency of repeated answers to the same question."""

from __future__ import annotations

import itertools
import statistics
from collections.abc import Sequence

import numpy as np

from ragbench.metrics.lexical import words
from ragbench.retrieval.embedder import EmbedderLike


def semantic_consistency(texts: Sequence[str], embedder: EmbedderLike) -> float | None:
    """Mean pairwise cosine similarity between the embeddings of `texts`.

    With the 3 benchmark trials this is the mean of 3 pairs. Returns None for fewer than 2 texts.
    The embedder must return L2-normalised vectors, so a dot product is the cosine.
    """
    if len(texts) < 2:
        return None
    vectors = embedder.encode(list(texts))
    sims = [float(vectors[i] @ vectors[j]) for i, j in itertools.combinations(range(len(texts)), 2)]
    return float(np.mean(sims))


def structural_consistency(texts: Sequence[str]) -> float | None:
    """1 - coefficient of variation of the word count across trials, clipped to [0, 1].

    A secondary metric that tracks how stable the answer *shape* is (the report's notion of
    consistency). None for fewer than 2 texts or when every answer is empty.
    """
    if len(texts) < 2:
        return None
    counts = [len(words(t)) for t in texts]
    mean = statistics.fmean(counts)
    if mean == 0:
        return None
    cv = statistics.pstdev(counts) / mean
    return max(0.0, min(1.0, 1.0 - cv))
