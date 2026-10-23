"""Phase 1 acceptance: real corpus and real embedding model (slow; first run downloads)."""

from __future__ import annotations

import numpy as np
import pytest

from ragbench.config import load_experiment_settings
from ragbench.ingest.chunking import chunk_corpus, length_stats
from ragbench.ingest.corpus import load_cleaned_documents
from ragbench.retrieval.embedder import Embedder
from ragbench.retrieval.store import VectorStore

pytestmark = pytest.mark.slow

DISTRACTOR = "zhu2023_pointcloud_survey"

# (question, document that must appear in the top-4). Draft questions from the plan (Q01-Q08).
LOOKUP_CASES = [
    (
        "What are the parametric and non-parametric memory components of the RAG model?",
        "lewis2020_rag",
    ),
    ("What is the difference between RAG-Sequence and RAG-Token?", "lewis2020_rag"),
    (
        "Which reflection tokens does Self-RAG use, and what does each one assess?",
        "asai2023_selfrag",
    ),
    ("Why does Self-RAG retrieve on demand instead of always retrieving?", "asai2023_selfrag"),
    (
        "What is the difference between Assert and Suggest in DSPy Assertions?",
        "singhvi2023_dspy_assertions",
    ),
    (
        "What is a vector database and how does it differ from a traditional database?",
        "pinecone_vector_db",
    ),
    ("What is Weaviate?", "weaviate_docs"),
    ("What is LangChain and what does it help developers build?", "langchain_overview"),
]
MULTI_HOP_QUESTIONS = [
    "How do the original RAG model and Self-RAG differ in deciding when and what to retrieve?",
    "Both Self-RAG and DSPy Assertions improve an output after a first attempt. "
    "How do their mechanisms differ?",
    "How does the vector search described in the Pinecone article relate to the retriever in RAG?",
    "Which approaches in these documents aim to reduce factual errors in language models?",
]


@pytest.fixture(scope="module")
def chunks():
    r = load_experiment_settings().retrieval
    return chunk_corpus(load_cleaned_documents(), r.chunk_chars, r.chunk_overlap)


@pytest.fixture(scope="module")
def embedder():
    return Embedder("all-MiniLM-L6-v2")


@pytest.fixture(scope="module")
def store(chunks, embedder, tmp_path_factory):
    s = VectorStore(tmp_path_factory.mktemp("chroma"), embedder)
    s.rebuild(chunks)
    return s


def test_chunk_length_matches_configured_cap(chunks):
    stats = length_stats(chunks)
    assert stats["max"] <= load_experiment_settings().retrieval.chunk_chars
    assert stats["median"] >= 500, stats
    assert stats["p10"] >= 350, stats


def test_every_document_is_chunked_with_pages(chunks):
    ids = {c.doc_id for c in chunks}
    assert len(ids) == 7 and DISTRACTOR in ids
    assert all(c.page >= 1 for c in chunks)


def test_index_size_matches_chunks(store, chunks):
    assert store.count() == len(chunks)


@pytest.mark.parametrize(("question", "expected"), LOOKUP_CASES, ids=lambda v: str(v)[:40])
def test_lookup_question_retrieves_its_source_document(store, question, expected):
    hits = store.query(question, top_k=4)
    assert expected in {h.chunk.doc_id for h in hits}, [h.chunk.doc_id for h in hits]


def test_distractor_never_appears_for_lookup_questions(store):
    for question, _ in LOOKUP_CASES:
        assert DISTRACTOR not in {h.chunk.doc_id for h in store.query(question, 4)}


def test_distractor_never_appears_for_multi_hop_questions(store):
    for question in MULTI_HOP_QUESTIONS:
        assert DISTRACTOR not in {h.chunk.doc_id for h in store.query(question, 4)}


def test_distractor_is_retrievable_for_its_own_topic(store):
    hits = store.query("How is point cloud data augmentation done for LiDAR detection?", 4)
    assert hits[0].chunk.doc_id == DISTRACTOR


def test_store_similarity_equals_direct_cosine(store, embedder):
    question = "What is the difference between RAG-Sequence and RAG-Token?"
    q = embedder.encode([question])[0]
    for h in store.query(question, 4):
        direct = float(q @ embedder.encode([h.chunk.text])[0])
        assert h.similarity == pytest.approx(direct, abs=1e-3)


def test_embeddings_are_normalised_384d(embedder):
    v = embedder.encode(["hello world", "retrieval augmented generation"])
    assert v.shape == (2, 384)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
