"""VectorStore behaviour with a fast deterministic embedder (no model download)."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from ragbench.retrieval.store import VectorStore
from ragbench.types import Chunk

DIM = 128


class HashingEmbedder:
    """Bag-of-words hashing into DIM buckets, L2-normalised. Similar words -> similar vectors."""

    name = "fake-hashing"

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


def chunk(cid: str, doc: str, text: str, role: str = "core", page: int = 1) -> Chunk:
    return Chunk(cid, doc, f"Title {doc}", role, page, text)


CHUNKS = [
    chunk("a:p1:c0", "a", "dense retrieval encodes queries and passages as vectors"),
    chunk("b:p2:c0", "b", "reflection tokens critique generated text for support", page=2),
    chunk("c:p1:c0", "c", "point cloud augmentation rotates scanned lidar points", "distractor"),
    chunk("d:p3:c0", "d", "vector databases index embeddings for similarity search", page=3),
]


@pytest.fixture
def store(tmp_path):
    s = VectorStore(tmp_path / "chroma", HashingEmbedder())
    s.rebuild(CHUNKS)
    return s


def test_rebuild_indexes_all_chunks(store):
    assert store.count() == len(CHUNKS)


def test_query_returns_best_match_first_with_metadata(store):
    hits = store.query("reflection tokens that critique text", top_k=2)
    assert hits[0].chunk.chunk_id == "b:p2:c0"
    assert hits[0].chunk.page == 2 and hits[0].chunk.role == "core"
    assert [h.rank for h in hits] == [1, 2]


def test_similarity_descends_and_matches_numpy_cosine(store):
    emb = HashingEmbedder()
    query = "vectors for dense retrieval of passages"
    hits = store.query(query, top_k=4)
    sims = [h.similarity for h in hits]
    assert sims == sorted(sims, reverse=True)
    q = emb.encode([query])[0]
    for h in hits:
        expected = float(q @ emb.encode([h.chunk.text])[0])
        assert h.similarity == pytest.approx(expected, abs=1e-4)


def test_top_k_is_respected(store):
    assert len(store.query("anything", top_k=3)) == 3


def test_rebuild_replaces_rather_than_appends(store):
    store.rebuild(CHUNKS[:2])
    assert store.count() == 2


def test_distractor_role_is_preserved(store):
    hits = store.query("lidar point cloud", top_k=1)
    assert hits[0].chunk.role == "distractor"


def test_retrieval_is_deterministic(store):
    a = store.query("vector search", top_k=4)
    b = store.query("vector search", top_k=4)
    assert [(h.chunk.chunk_id, h.similarity) for h in a] == [
        (h.chunk.chunk_id, h.similarity) for h in b
    ]
