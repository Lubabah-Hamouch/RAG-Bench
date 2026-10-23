"""Name -> strategy lookup. Adding a strategy means one new file and one line here."""

from __future__ import annotations

from ragbench.strategies.base import Strategy
from ragbench.strategies.baseline import Baseline
from ragbench.strategies.no_retrieval import NoRetrieval
from ragbench.strategies.rrr import RRR
from ragbench.strategies.self_rag import SelfRAG
from ragbench.strategies.structured import Structured

_ORDER: list[type[Strategy]] = [Baseline, Structured, RRR, SelfRAG, NoRetrieval]
REGISTRY: dict[str, type[Strategy]] = {cls.name: cls for cls in _ORDER}
ALL_NAMES: list[str] = list(REGISTRY)


def get_strategy(name: str) -> Strategy:
    try:
        return REGISTRY[name]()
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; choose from {ALL_NAMES}") from None
