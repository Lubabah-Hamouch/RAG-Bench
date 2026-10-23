"""The benchmark runner on fakes: persistence, resume, ordering, quota stop, judge errors."""

from __future__ import annotations

import json

import pytest

from ragbench.benchmark import (
    LOG_FILE,
    RETRIEVAL_FILE,
    RUNS_FILE,
    load_retrieval,
    load_runs,
    run_benchmark,
)
from ragbench.config import load_experiment_settings
from ragbench.llm.client import LLMError
from ragbench.llm.fake import FakeLLMClient
from ragbench.metrics.cost import ModelPrice
from ragbench.metrics.judge import Judge
from ragbench.questions import GoldFact, Question
from ragbench.types import Chunk, RetrievedChunk

PRICING = {"fake-llm": ModelPrice(1.0, 2.0), "fake-judge": ModelPrice(0.1, 0.2)}
SETTINGS = load_experiment_settings()


def make_chunks():
    return [
        RetrievedChunk(i, 0.8, Chunk(f"d:p1:c{i}", "d", "T", "core", 1, f"passage text {i}"))
        for i in range(1, 5)
    ]


class StubRetriever:
    def __init__(self):
        self.queries = 0

    def query(self, text, top_k=4):
        self.queries += 1
        return make_chunks()


def judge_responder(messages):
    text = messages[-1]["content"]
    if "Decompose the answer" in text:
        return json.dumps({"claims": ["claim one", "claim two"]})
    if "checking claims" in text:
        return json.dumps(
            {
                "verdicts": [
                    {"claim_id": 1, "label": "supported", "evidence_passage": 1, "reason": "r"},
                    {
                        "claim_id": 2,
                        "label": "unsupported",
                        "evidence_passage": None,
                        "reason": "r",
                    },
                ]
            }
        )
    return json.dumps({"coverage": [{"fact_id": 1, "covered": True}]})


def make_judge(responder=judge_responder):
    return Judge(FakeLLMClient(responder, model="fake-judge"))


QUESTIONS = [
    Question(
        id="QA",
        type="lookup",
        question="First question?",
        gold_facts=[GoldFact(text="fact", doc_id="d", page=1, quote="q")],
    ),
    Question(
        id="QB",
        type="lookup",
        question="Second question?",
        gold_facts=[GoldFact(text="fact", doc_id="d", page=1, quote="q")],
    ),
]


def run(tmp_path, client=None, judge=None, retriever=None, log=None, **kw):
    lines: list[str] = []
    outcome = run_benchmark(
        QUESTIONS,
        retriever or StubRetriever(),
        client or FakeLLMClient(),
        judge or make_judge(),
        PRICING,
        SETTINGS,
        tmp_path,
        strategies=kw.pop("strategies", ["baseline", "rrr"]),
        trials=kw.pop("trials", 2),
        judge_trials=kw.pop("judge_trials", 1),
        log=log or lines.append,
        **kw,
    )
    return outcome, lines


def test_every_question_strategy_trial_is_recorded(tmp_path):
    outcome, _ = run(tmp_path)
    runs = load_runs(tmp_path)
    assert outcome.completed == 8 and outcome.skipped == 0 and not outcome.quota_stopped
    assert set(runs) == {
        (q, s, t) for q in ("QA", "QB") for s in ("baseline", "rrr") for t in (0, 1)
    }


def test_only_the_first_judge_trials_are_judged(tmp_path):
    run(tmp_path)
    runs = load_runs(tmp_path)
    for (_, _, trial), rec in runs.items():
        assert rec["judged"] == (trial == 0)
        assert ("groundedness" in rec) == (trial == 0)


def test_judged_record_contents(tmp_path):
    run(tmp_path)
    rec = load_runs(tmp_path)[("QA", "baseline", 0)]
    assert rec["groundedness"]["G"] == 0.5 and rec["groundedness"]["labels"] == [
        "supported",
        "unsupported",
    ]
    assert rec["coverage"]["C"] == 1.0 and rec["fcs"] == pytest.approx(2 * 0.5 / 1.5)
    assert rec["gen_cost_usd"] is not None and rec["judge_cost_usd"] is not None
    assert rec["lexical"]["words"] > 0 and len(rec["calls"]) == 1
    assert len(load_runs(tmp_path)[("QA", "rrr", 0)]["calls"]) == 3
    json.dumps(rec)  # fully serialisable


def test_retrieval_is_recorded_once_per_question_and_shared_across_trials(tmp_path):
    retriever = StubRetriever()
    run(tmp_path, retriever=retriever)
    assert retriever.queries == 2  # once per question, reused for both trials
    ret = load_retrieval(tmp_path)
    assert set(ret) == {"QA", "QB"} and len(ret["QA"]["chunks"]) == 4


def test_trial_zero_finishes_before_trial_one_starts(tmp_path):
    _, lines = run(tmp_path)
    trials = [ln.split()[0] for ln in lines if ln.strip().startswith("t")]
    assert trials == ["t0"] * 4 + ["t1"] * 4


def test_rerun_skips_everything_already_done(tmp_path):
    run(tmp_path)
    client = FakeLLMClient()
    outcome, _ = run(tmp_path, client=client)
    assert outcome.completed == 0 and outcome.skipped == 8 and not client.calls


def test_resume_only_redoes_missing_records(tmp_path):
    run(tmp_path)
    path = tmp_path / RUNS_FILE
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-2]) + "\n", encoding="utf-8")
    outcome, _ = run(tmp_path)
    assert outcome.completed == 2 and outcome.skipped == 6
    assert len(load_runs(tmp_path)) == 8


def test_an_unjudged_record_in_a_judged_trial_is_redone_and_upgraded(tmp_path):
    run(tmp_path, judge_trials=0)  # nothing judged
    assert not any(r["judged"] for r in load_runs(tmp_path).values())
    outcome, _ = run(tmp_path, judge_trials=1)
    runs = load_runs(tmp_path)
    assert outcome.completed == 4 and all(runs[(q, s, 0)]["judged"] for q, s, _ in runs)


class QuotaAfter(FakeLLMClient):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def chat(self, messages, **kw):
        if len(self.calls) >= self.n:
            raise LLMError("rate-limited; provider asks to wait 900s", code="rate_limit_long")
        return super().chat(messages, **kw)


def test_quota_stop_saves_progress_and_resumes_to_completion(tmp_path):
    outcome, lines = run(tmp_path, client=QuotaAfter(3))
    assert outcome.quota_stopped and 0 < outcome.completed < 8
    assert any("STOPPED" in ln for ln in lines)
    saved = len(load_runs(tmp_path))
    assert saved == outcome.completed
    final, _ = run(tmp_path)
    assert not final.quota_stopped and len(load_runs(tmp_path)) == 8


def test_other_llm_errors_propagate(tmp_path):
    class Broken(FakeLLMClient):
        def chat(self, messages, **kw):
            raise LLMError("auth failed", code="invalid_api_key")

    with pytest.raises(LLMError, match="auth failed"):
        run(tmp_path, client=Broken())


def test_a_judge_failure_is_recorded_and_the_run_continues(tmp_path):
    def bad_judge(messages):
        return "not json at all"

    outcome, lines = run(tmp_path, judge=make_judge(bad_judge))
    assert outcome.judge_errors == 4 and outcome.completed == 8
    rec = load_runs(tmp_path)[("QA", "baseline", 0)]
    assert not rec["judged"] and rec["judge_error"]
    assert any("JUDGE-ERROR" in ln for ln in lines)
    # the failed judgements are retried on the next run
    retry, _ = run(tmp_path)
    assert retry.completed == 4 and retry.judge_errors == 0
    assert load_runs(tmp_path)[("QA", "baseline", 0)]["judged"]


def test_run_log_is_appended_per_invocation(tmp_path):
    run(tmp_path)
    run(tmp_path)
    entries = [
        json.loads(x) for x in (tmp_path / LOG_FILE).read_text(encoding="utf-8").splitlines()
    ]
    assert [e["completed"] for e in entries] == [8, 0]
    assert entries[0]["trials"] == 2 and entries[0]["judge_trials"] == 1


def test_files_are_jsonl_in_the_output_directory(tmp_path):
    run(tmp_path)
    assert (tmp_path / RETRIEVAL_FILE).exists() and (tmp_path / RUNS_FILE).exists()


def test_context_support_run_is_resumable_and_skips_questions_without_gold_facts(tmp_path):
    from ragbench.benchmark import CONTEXT_SUPPORT_FILE, _read_jsonl, run_context_support

    run(tmp_path, judge_trials=0)  # writes retrieval.jsonl
    no_gold = Question(id="QC", type="out_of_corpus", question="Unknowable?", expected_abstain=True)

    def responder(messages):
        return json.dumps({"support": [{"fact_id": 1, "supported": True}]})

    client = FakeLLMClient(responder, model="fake-judge")
    judge = Judge(client)
    qs = [*QUESTIONS, no_gold]
    assert run_context_support(qs, tmp_path, judge, log=lambda _: None) == 2
    rows = _read_jsonl(tmp_path / CONTEXT_SUPPORT_FILE)
    assert {r["qid"] for r in rows} == {"QA", "QB"} and all(r["recall"] == 1.0 for r in rows)
    calls_before = len(client.calls)
    assert run_context_support(qs, tmp_path, judge, log=lambda _: None) == 0
    assert len(client.calls) == calls_before
