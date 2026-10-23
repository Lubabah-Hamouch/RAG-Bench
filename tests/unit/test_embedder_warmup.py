"""Warm-up must really load the embedding model, so `live_ready` means a live run will not stall.

Regression: the warm-up used to build only the store. `Embedder` imports sentence-transformers and
loads the weights lazily on the first encode (the import alone took 50-67 s cold), so /api/health
reported live_ready while the first live run still paid the whole cold start inside the run.
"""

from __future__ import annotations

import sys
import threading
import types
from typing import Any

import numpy as np

from ragbench.api.live import LazyRetriever, _default_retriever
from ragbench.retrieval.embedder import Embedder


class _Model:
    def __init__(self) -> None:
        self.encoded: list[list[str]] = []

    def encode(self, texts: list[str], **_: Any) -> Any:
        self.encoded.append(list(texts))
        return np.ones((len(texts), 4), dtype=np.float32)


def test_embedder_warm_runs_an_encode_so_the_model_is_really_loaded() -> None:
    e = Embedder()
    model = _Model()
    e._model = model
    e.warm()
    assert model.encoded == [["warm up"]]


def _fake_sentence_transformers(
    monkeypatch: Any, calls: list[dict[str, Any]], *, cached: bool
) -> None:
    def factory(name: str, device: str = "cpu", **kwargs: Any) -> _Model:
        calls.append({"name": name, **kwargs})
        if kwargs.get("local_files_only") and not cached:
            raise OSError("not in the local cache")
        return _Model()

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_model_loads_from_the_local_cache_first(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []
    _fake_sentence_transformers(monkeypatch, calls, cached=True)
    Embedder().warm()
    assert calls == [{"name": "all-MiniLM-L6-v2", "local_files_only": True}]  # no download


def test_model_falls_back_to_downloading_when_it_is_not_cached(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []
    _fake_sentence_transformers(monkeypatch, calls, cached=False)
    Embedder().warm()
    assert [bool(c.get("local_files_only")) for c in calls] == [True, False]


def test_default_retriever_is_not_ready_until_the_embedder_has_loaded(monkeypatch: Any) -> None:
    order: list[str] = []
    gate = threading.Event()

    class FakeEmbedder:
        def __init__(self, name: str) -> None:
            self.name = name

        def warm(self) -> None:
            gate.wait(5)  # stands in for the minute-long import + weight load
            order.append("embedder loaded")

    class FakeStore:
        def __init__(self, path: Any, embedder: Any) -> None:
            order.append("store built")

        def query(self, text: str, top_k: int = 4) -> list[Any]:
            return []

    monkeypatch.setattr("ragbench.retrieval.embedder.Embedder", FakeEmbedder)
    monkeypatch.setattr("ragbench.retrieval.store.VectorStore", FakeStore)

    lazy = LazyRetriever(_default_retriever)
    lazy.warm()
    assert lazy.ready is False  # still loading the model: health must not say live_ready yet
    gate.set()
    lazy.query("q")
    assert lazy.ready is True
    assert order == ["embedder loaded", "store built"]  # the model is loaded before the store
