"""Replay endpoints: they serve recorded results, with no key and no model calls."""

from __future__ import annotations

import json

import pytest
from api_fakes import (
    MODELS,
    PRICING,
    QUESTIONS,
    TITLES,
    FakeFactory,
    HashEmbedder,
    StubRetriever,
    judge_responder,
)
from fastapi.testclient import TestClient

from ragbench.analysis import summarize
from ragbench.api.main import create_app
from ragbench.api.schemas import QuestionResultOut
from ragbench.api.settings import ApiSettings
from ragbench.benchmark import run_benchmark
from ragbench.config import load_experiment_settings, repo_root
from ragbench.llm.fake import FakeLLMClient
from ragbench.metrics.judge import Judge
from ragbench.prompts import REFUSAL
from ragbench.questions import QuestionSet, load_questions
from ragbench.report import write_outputs

SETTINGS = load_experiment_settings()
META = {
    "n_questions": 3,
    "n_answerable": 2,
    "composition": "x",
    "trials": 2,
    "judge_trials": 1,
    "generator": "fake-llm",
    "judge": "fake-judge",
    "log": None,
}


def build_results(path, gen=None):
    run_benchmark(
        QUESTIONS,
        StubRetriever(),
        gen or FakeLLMClient(),
        Judge(FakeLLMClient(judge_responder, model="fake-judge")),
        PRICING,
        SETTINGS,
        path,
        trials=2,
        judge_trials=1,
        log=lambda _: None,
    )
    summary = summarize(QUESTIONS, path, HashEmbedder(), PRICING, n_resamples=100, seed=1)
    write_outputs(summary, META, path)
    (path / "context_support.jsonl").write_text(
        json.dumps({"qid": "L1", "supported": [True], "recall": 0.5}) + "\n", encoding="utf-8"
    )


def make_client(results_dir):
    app = create_app(
        settings=ApiSettings(_env_file=None),
        experiment=SETTINGS,
        results_dir=results_dir,
        questions=QuestionSet(questions=QUESTIONS),
        retriever=StubRetriever(),
        client_factory=FakeFactory(),
        pricing=PRICING,
        titles=TITLES,
        models=MODELS,
    )
    return TestClient(app)


@pytest.fixture(scope="module")
def results_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("results")
    build_results(path)
    return path


@pytest.fixture
def client(results_dir):
    return make_client(results_dir)


def test_health_reports_replay_and_models(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["replay_available"] and body["questions"] == 3
    assert body["models"] == MODELS and body["require_https"] is False


def test_questions_endpoint_lists_questions_with_gold_facts(client):
    qs = client.get("/api/questions").json()
    assert [q["id"] for q in qs] == ["L1", "M1", "O1"]
    assert qs[1]["type"] == "multi_hop" and len(qs[1]["gold_facts"]) == 2
    assert qs[2]["expected_abstain"] is True and qs[2]["gold_facts"] == []


def test_strategies_endpoint(client):
    infos = client.get("/api/strategies").json()
    assert [s["name"] for s in infos] == [
        "baseline",
        "structured",
        "rrr",
        "self_rag",
        "no_retrieval",
    ]
    assert {s["name"]: s["calls"] for s in infos}["rrr"] == 3
    assert [s["uses_retrieval"] for s in infos][-1] is False


def test_result_has_all_five_conditions_with_metrics(client):
    r = client.get("/api/results/L1").json()
    assert r["mode"] == "replay" and r["n_trials"] == 2 and r["judged_trials"] == 1
    assert list(r["answers"]) == ["baseline", "structured", "rrr", "self_rag", "no_retrieval"]
    a = r["answers"]["baseline"]
    m = a["metrics"]
    assert a["trial"] == 0 and m["judged"] is True
    assert m["groundedness"]["G"] == 0.5 and m["groundedness"]["n_claims"] == 2
    assert [c["label"] for c in m["groundedness"]["claims"]] == ["supported", "unsupported"]
    assert m["coverage"]["C"] == 1.0 and m["coverage"]["gold_facts"] == ["f1"]
    assert m["fcs"] == pytest.approx(2 * 0.5 * 1.0 / 1.5)
    assert m["calls"] == 1 and m["latency_s"] >= 0 and m["prompt_tokens"] > 0
    assert m["cost_usd"] == pytest.approx(
        (m["prompt_tokens"] * 1.0 + m["completion_tokens"] * 2.0) / 1e6
    )
    assert m["judge_tokens"] > 0 and m["lexical"]["words"] > 0


def test_multi_call_strategies_expose_their_trace_and_self_rag_critique(client):
    r = client.get("/api/results/L1").json()["answers"]
    assert [s["name"] for s in r["rrr"]["trace"]] == ["identify", "synthesize", "refine"]
    assert r["rrr"]["metrics"]["calls"] == 3
    assert r["self_rag"]["critique"] and "Passage relevance" in r["self_rag"]["critique"]
    assert r["baseline"]["critique"] is None


def test_consistency_comes_from_the_recorded_trials(client):
    a = client.get("/api/results/L1").json()["answers"]["baseline"]
    c = a["metrics"]["consistency"]
    assert c["n_samples"] == 2 and c["structural"] is not None
    assert [t["trial"] for t in a["other_trials"]] == [1] and a["other_trials"][0]["answer"]


def test_retrieval_is_shared_and_carries_titles_and_judge_recall(client):
    r = client.get("/api/results/L1").json()
    chunks = r["retrieval"]["chunks"]
    assert [c["rank"] for c in chunks] == [1, 2, 3, 4]
    assert chunks[0]["doc_title"] == "Document D" and chunks[0]["role"] == "core"
    assert r["retrieval"]["context_recall_judge"] == 0.5
    assert client.get("/api/results/M1").json()["retrieval"]["context_recall_judge"] is None


def test_out_of_corpus_question_has_no_coverage_fcs_or_consistency(client):
    a = client.get("/api/results/O1").json()["answers"]["baseline"]["metrics"]
    assert a["coverage"] is None and a["consistency"] is None
    assert a["fcs"] is None  # no gold facts, so no coverage and no FCS


def test_a_refusal_on_an_answerable_question_scores_fcs_zero(tmp_path):
    class Refuser(FakeLLMClient):
        def __init__(self):
            super().__init__(lambda m: REFUSAL)

    build_results(tmp_path, gen=Refuser())
    m = make_client(tmp_path).get("/api/results/L1").json()["answers"]["baseline"]
    assert m["abstained"] is True and m["metrics"]["fcs"] == 0.0
    assert (
        m["metrics"]["groundedness"]["abstained"] is True
        and m["metrics"]["groundedness"]["G"] is None
    )


def test_summary_endpoint_returns_the_aggregate_report(client):
    s = client.get("/api/results/summary").json()
    assert {"meta", "strategies", "paired", "abstention", "retrieval", "totals"} <= set(s)
    assert s["strategies"]["baseline"]["FCS"]["n"] == 2


def test_unknown_question_is_a_404_with_a_json_error(client):
    r = client.get("/api/results/NOPE")
    assert r.status_code == 404 and r.json()["error"] == "unknown_question"


def test_missing_results_give_503_with_guidance(tmp_path):
    c = make_client(tmp_path)  # empty directory
    assert c.get("/api/health").json()["replay_available"] is False
    for url in ("/api/results/L1", "/api/results/summary"):
        r = c.get(url)
        assert r.status_code == 503 and r.json()["error"] == "replay_unavailable"
    assert len(c.get("/api/questions").json()) == 3  # questions do not need results


def test_replay_needs_no_key_and_makes_no_model_calls(results_dir):
    factory = FakeFactory()
    app = create_app(
        settings=ApiSettings(_env_file=None),
        experiment=SETTINGS,
        results_dir=results_dir,
        questions=QuestionSet(questions=QUESTIONS),
        retriever=StubRetriever(),
        client_factory=factory,
        pricing=PRICING,
        titles=TITLES,
        models=MODELS,
    )
    c = TestClient(app)
    for url in ("/api/health", "/api/questions", "/api/results/L1", "/api/results/summary"):
        assert c.get(url).status_code == 200
    assert factory.requests == [] and not factory.generator.calls


def test_results_are_reloaded_when_the_files_change(tmp_path):
    build_results(tmp_path)
    c = make_client(tmp_path)
    first = c.get("/api/results/L1").json()["answers"]["baseline"]["answer"]
    runs = tmp_path / "runs.jsonl"
    rows = [json.loads(x) for x in runs.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        if r["qid"] == "L1" and r["strategy"] == "baseline" and r["trial"] == 0:
            r["answer"] = "CHANGED ANSWER"
    runs.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    import os
    import time

    os.utime(runs, (time.time() + 5, time.time() + 5))
    assert first != "CHANGED ANSWER"
    assert c.get("/api/results/L1").json()["answers"]["baseline"]["answer"] == "CHANGED ANSWER"


def test_static_export_matches_what_the_api_serves(results_dir, tmp_path):
    c = make_client(results_dir)
    store = c.app.state.ctx.replay
    written = store.export_static(tmp_path / "replay")
    names = {p.name for p in written}
    assert {
        "questions.json",
        "strategies.json",
        "summary.json",
        "L1.json",
        "M1.json",
        "O1.json",
    } <= names
    exported = json.loads((tmp_path / "replay" / "results" / "L1.json").read_text(encoding="utf-8"))
    assert exported == c.get("/api/results/L1").json()
    assert json.loads((tmp_path / "replay" / "questions.json").read_text(encoding="utf-8")) == (
        c.get("/api/questions").json()
    )


# ---- against the real recorded benchmark ------------------------------

REAL = repo_root() / "evaluation" / "results"
needs_real = pytest.mark.skipif(not (REAL / "runs.jsonl").exists(), reason="no benchmark results")


@pytest.fixture(scope="module")
def real_client():
    return TestClient(create_app(settings=ApiSettings(_env_file=None)))


@needs_real
def test_real_results_cover_the_ten_questions_and_five_conditions(real_client):
    qs = real_client.get("/api/questions").json()
    assert len(qs) == 10
    for q in qs:
        r = QuestionResultOut.model_validate(real_client.get(f"/api/results/{q['id']}").json())
        assert r.mode == "replay" and r.n_trials == 2 and r.judged_trials == 1
        assert list(r.answers) == ["baseline", "structured", "rrr", "self_rag", "no_retrieval"]
        assert len(r.retrieval.chunks) == 4
        for a in r.answers.values():
            assert a.answer and a.metrics.judged and a.metrics.calls >= 1
            if r.question.type != "out_of_corpus":
                assert a.metrics.consistency is not None and a.metrics.consistency.n_samples == 2


@needs_real
def test_real_per_question_fcs_averages_back_to_the_published_summary(real_client):
    summary = real_client.get("/api/results/summary").json()
    answerable = [
        q["id"] for q in real_client.get("/api/questions").json() if not q["expected_abstain"]
    ]
    for strategy, row in summary["strategies"].items():
        vals = []
        for qid in answerable:
            fcs = real_client.get(f"/api/results/{qid}").json()["answers"][strategy]["metrics"][
                "fcs"
            ]
            assert fcs is not None
            vals.append(fcs)
        assert sum(vals) / len(vals) == pytest.approx(row["FCS"]["mean"], abs=1e-9), strategy


@needs_real
def test_real_known_answer_and_metrics_are_served_unchanged(real_client):
    r = real_client.get("/api/results/Q13").json()
    for name in ("baseline", "structured", "rrr", "self_rag"):
        assert r["answers"][name]["abstained"] is True
    assert r["answers"]["no_retrieval"]["abstained"] is False
    assert real_client.get("/api/results/summary").json()["sensitivity"]["excluded"] == ["Q11"]
    assert load_questions().questions[0].id == real_client.get("/api/questions").json()[0]["id"]
