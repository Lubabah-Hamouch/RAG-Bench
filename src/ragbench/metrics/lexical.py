"""Surface metrics from the original report, plus MATTR as a length-robust diversity measure."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import pysbd

_WORD = re.compile(r"\w+")


def words(text: str) -> list[str]:
    """Lower-cased word tokens (`\\w+`)."""
    return _WORD.findall(text.lower())


@lru_cache(maxsize=1)
def _segmenter() -> pysbd.Segmenter:
    return pysbd.Segmenter(language="en", clean=False)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _segmenter().segment(text) if s.strip()]


def mattr(tokens: list[str], window: int = 25) -> float | None:
    """Moving-average type-token ratio: mean of unique/window over every sliding window.

    Unlike plain TTR it does not fall as texts get longer, and it does not hand a perfect 1.0 to
    very short answers (the report's 6-word Baseline). Returns None when the text is shorter than
    one window, because the measure is not defined there.
    """
    n = len(tokens)
    if window <= 0 or n < window:
        return None
    counts: Counter[str] = Counter(tokens[:window])
    unique = len(counts)
    total = float(unique)
    for i in range(window, n):
        leaving, entering = tokens[i - window], tokens[i]
        counts[leaving] -= 1
        if counts[leaving] == 0:
            unique -= 1
            del counts[leaving]
        counts[entering] += 1
        if counts[entering] == 1:
            unique += 1
        total += unique
    return total / (window * (n - window + 1))


@dataclass(frozen=True)
class LexicalMetrics:
    chars: int
    words: int
    sentences: int
    avg_word_length: float | None
    words_per_sentence: float | None
    unique_vocab: int
    ttr: float | None  # unique / total, as in the original report (length-sensitive)
    mattr: float | None  # headline diversity; None for answers shorter than the window


def lexical_metrics(text: str, window: int = 25) -> LexicalMetrics:
    toks = words(text)
    n = len(toks)
    n_sent = len(sentences(text))
    unique = len(set(toks))
    return LexicalMetrics(
        chars=len(text),
        words=n,
        sentences=n_sent,
        avg_word_length=sum(map(len, toks)) / n if n else None,
        words_per_sentence=n / n_sent if n_sent else None,
        unique_vocab=unique,
        ttr=unique / n if n else None,
        mattr=mattr(toks, window),
    )
