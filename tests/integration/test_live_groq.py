"""Live checks against the real Groq API. Skipped unless RAGBENCH_LIVE=1 and a key is configured.

Run with:  RAGBENCH_LIVE=1 pytest -m live -s
Everything else in the suite uses the fake client and never touches the network.
"""

from __future__ import annotations

import os

import pytest

from ragbench.config import EnvSettings, load_experiment_settings, resolve_chroma_dir
from ragbench.llm.roles import make_client
from ragbench.pipeline import run_question
from ragbench.retrieval.embedder import Embedder
from ragbench.retrieval.store import VectorStore

_env = EnvSettings()
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RAGBENCH_LIVE") != "1", reason="set RAGBENCH_LIVE=1"),
    pytest.mark.skipif(_env.generator_api_key is None, reason="no GENERATOR_API_KEY configured"),
]

ANSWERABLE = "What is the difference between Assert and Suggest in DSPy Assertions?"
OUT_OF_CORPUS = "What results did Self-RAG report on the Llama 3 model family?"


@pytest.fixture(scope="module")
def client():
    return make_client("generator")


@pytest.fixture(scope="module")
def store():
    s = load_experiment_settings()
    return VectorStore(resolve_chroma_dir(s), Embedder(s.retrieval.embedding_model))


def run(question, store, client, strategies=None):
    s = load_experiment_settings()
    return run_question(
        question,
        store,
        client,
        strategies=strategies,
        top_k=s.retrieval.top_k,
        temperature=0.0,  # deterministic enough to check prompt behaviour
        max_tokens=s.generation.max_tokens,
    )


@pytest.mark.parametrize("role", ["generator", "judge"])
def test_configured_model_is_offered_by_the_provider(role):
    """Fails loudly when a model id has been retired, listing what is available instead."""
    client = make_client(role, use_cache=False)
    available = sorted(m.id for m in client._sdk.models.list().data)
    assert client.model in available, f"{client.model!r} not offered; available: {available}"


def test_all_five_conditions_run_and_record_metrics(store, client):
    result = run(ANSWERABLE, store, client)
    assert len(result.chunks) == 4
    expected_calls = {"baseline": 1, "structured": 1, "rrr": 3, "self_rag": 3, "no_retrieval": 1}
    for name, n in expected_calls.items():
        r = result.results[name]
        assert r.n_calls == n and r.answer.strip(), name
        assert r.prompt_tokens > 0 and r.completion_tokens > 0 and r.latency_s > 0, name
    assert "## Summary" in result.results["structured"].answer
    assert "Answerable from context" in (result.results["self_rag"].critique or "")


def test_answerable_question_is_not_refused_by_retrieval_strategies(store, client):
    result = run(ANSWERABLE, store, client, ["baseline", "structured", "rrr", "self_rag"])
    refused = [n for n, r in result.results.items() if r.abstained]
    assert not refused, refused


def test_out_of_corpus_question_is_refused_by_most_retrieval_strategies(store, client):
    result = run(OUT_OF_CORPUS, store, client, ["baseline", "structured", "rrr", "self_rag"])
    refused = {n: r.abstained for n, r in result.results.items()}
    print("abstention on out-of-corpus question:", refused)
    assert sum(refused.values()) >= 2, refused


# ---- Judge validity: does it tell a grounded answer from a fabricated one? -------------------

GROUNDED = (
    "For both Assert and Suggest, if the maximum retry attempts are not exceeded, the handler "
    "returns control to the failing module with an updated prompt. After exceeding the maximum "
    "retries, the handler halts execution for a failing Assert, while for a failing Suggest it "
    "progresses to the next module in the pipeline."
)
SWAPPED = (
    "After exceeding the maximum retries, the handler halts execution for a failing Suggest, "
    "while for a failing Assert it silently continues to the next module. Both constructs were "
    "introduced in 2015 by the TensorFlow team."
)


@pytest.fixture(scope="module")
def judge():
    from ragbench.metrics.judge import Judge

    s = load_experiment_settings()
    return Judge(
        make_client("judge"),
        max_tokens=s.judge.max_tokens,
        max_tokens_cap=s.judge.max_tokens_cap,
        max_retries=s.judge.max_retries,
    )


def test_judge_scores_a_grounded_answer_high_and_a_fabricated_one_low(store, judge):
    from ragbench.metrics.groundedness import score_groundedness

    chunks = store.query(ANSWERABLE, 4)
    good = score_groundedness(ANSWERABLE, GROUNDED, chunks, judge)
    bad = score_groundedness(ANSWERABLE, SWAPPED, chunks, judge)
    print(f"grounded G={good.groundedness} fabricated G={bad.groundedness}", end=" ")
    print(f"X={bad.contradiction_rate}")
    assert good.groundedness is not None and good.groundedness >= 0.8
    assert bad.groundedness is not None and bad.groundedness <= 0.5
    assert good.groundedness - bad.groundedness >= 0.4
    assert bad.n_contradicted + bad.n_unsupported >= 1
