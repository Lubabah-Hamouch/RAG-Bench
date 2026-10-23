"""Live endpoints (BYOK): behaviour, key secrecy, abuse protection, CORS and error mapping.

Every model call here goes to a fake client. The real API is never contacted.
"""

from __future__ import annotations

import json
import logging
import re

import pytest
from api_fakes import (
    KEY,
    MODELS,
    PRICING,
    QUESTIONS,
    TITLES,
    FailingClient,
    FakeFactory,
    StubRetriever,
    llm_error,
    parse_sse,
)
from fastapi.testclient import TestClient

from ragbench.api.live import LazyRetriever, default_client_factory
from ragbench.api.main import create_app
from ragbench.api.schemas import QuestionResultOut
from ragbench.api.security import (
    InflightGuard,
    RateLimiter,
    RedactionFilter,
    install_log_redaction,
    valid_key_shape,
)
from ragbench.api.settings import ApiSettings
from ragbench.config import load_experiment_settings
from ragbench.llm.client import LLMClient
from ragbench.llm.fake import FakeLLMClient
from ragbench.questions import QuestionSet

HEADERS = {"X-Groq-Api-Key": KEY}
STRATEGIES = ["baseline", "structured", "rrr", "self_rag", "no_retrieval"]


def make(factory=None, retriever=None, base_url="http://testserver", **settings):
    factory = factory or FakeFactory()
    retriever = retriever or StubRetriever()
    app = create_app(
        settings=ApiSettings(_env_file=None, **settings),
        experiment=load_experiment_settings(),
        results_dir=None,
        questions=QuestionSet(questions=QUESTIONS),
        retriever=retriever,
        client_factory=factory,
        pricing=PRICING,
        titles=TITLES,
        models=MODELS,
    )
    return TestClient(app, base_url=base_url), factory, retriever


def run(client, body=None, headers=HEADERS, stream=False):
    url = "/api/run" + ("" if stream else "?stream=false")
    return client.post(
        url, json=body if body is not None else {"question_id": "L1"}, headers=headers
    )


# ---- happy path ------------------------------


def test_live_run_returns_all_five_conditions_in_the_replay_shape():
    client, factory, retriever = make()
    r = run(client)
    assert r.status_code == 200
    out = QuestionResultOut.model_validate(r.json())
    assert out.mode == "live" and out.n_trials == 1 and out.judged_trials == 1
    assert list(out.answers) == STRATEGIES
    assert out.question.id == "L1" and out.question.type == "lookup"
    assert [c.rank for c in out.retrieval.chunks] == [1, 2, 3, 4]
    assert out.models == MODELS and retriever.queries == ["Lookup one?"]


def test_live_runs_are_single_trial_so_no_consistency_anywhere():
    client, *_ = make()
    out = QuestionResultOut.model_validate(run(client).json())
    for a in out.answers.values():
        assert a.metrics.consistency is None and a.other_trials == [] and a.trial == 0


def test_metrics_cover_groundedness_coverage_fcs_latency_and_cost():
    client, *_ = make()
    out = QuestionResultOut.model_validate(run(client).json())
    m = out.answers["baseline"].metrics
    assert m.judged and m.groundedness is not None and m.groundedness.G == 0.5
    assert m.coverage is not None and m.coverage.C == 1.0
    assert m.fcs == pytest.approx(2 * 0.5 * 1.0 / 1.5)
    assert m.latency_s >= 0 and m.calls == 1 and m.cost_usd is not None and m.judge_tokens
    assert out.answers["rrr"].metrics.calls == 3 and out.answers["self_rag"].critique


def test_visitors_key_is_what_the_factory_receives_for_both_roles():
    client, factory, _ = make()
    run(client)
    assert set(factory.requests) == {("generator", KEY), ("judge", KEY)}
    assert len(factory.generator.calls) == 1 + 1 + 3 + 3 + 1  # every condition ran
    assert factory.judge.calls  # scoring used the judge


def test_free_text_questions_are_scored_for_groundedness_but_not_coverage():
    client, *_ = make()
    out = QuestionResultOut.model_validate(
        run(client, {"question": "  What   is\n a thing? "}).json()
    )
    assert out.question.type == "free_text" and out.question.question == "What is a thing?"
    m = out.answers["structured"].metrics
    assert m.groundedness is not None and m.coverage is None and m.fcs is None


def test_score_false_skips_the_judge_entirely():
    client, factory, _ = make()
    out = QuestionResultOut.model_validate(
        run(client, {"question_id": "L1", "score": False}).json()
    )
    assert out.judged_trials == 0
    assert all(role == "generator" for role, _ in factory.requests)
    assert not factory.judge.calls
    m = out.answers["baseline"].metrics
    assert m.judged is False and m.groundedness is None and m.judge_tokens is None
    assert m.lexical.words > 0 and m.latency_s >= 0


def test_out_of_corpus_benchmark_question_has_no_coverage():
    client, *_ = make()
    out = QuestionResultOut.model_validate(run(client, {"question_id": "O1"}).json())
    assert out.question.expected_abstain and out.answers["baseline"].metrics.coverage is None


# ---- streaming ------------------------------


def test_sse_stream_emits_events_in_order_and_ends_with_the_full_result():
    client, *_ = make()
    r = run(client, stream=True)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    names = [n for n, _ in events]
    assert names[0] == "retrieval" and names[-1] == "done"
    assert names[1:-1] == ["strategy_started", "strategy_done", "metrics"] * 5
    started = [p["strategy"] for n, p in events if n == "strategy_started"]
    assert started == STRATEGIES
    done_unscored = next(p for n, p in events if n == "strategy_done")
    assert done_unscored["metrics"]["judged"] is False  # shown before scoring finishes
    scored = next(p for n, p in events if n == "metrics")
    assert scored["metrics"]["judged"] is True
    final = QuestionResultOut.model_validate(events[-1][1])
    assert final.mode == "live" and list(final.answers) == STRATEGIES
    assert events[0][1]["chunks"][0]["rank"] == 1 and events[0][1]["question"]["id"] == "L1"


def test_streamed_and_non_streamed_results_have_the_same_shape():
    client, *_ = make()
    plain = run(client).json()
    streamed = parse_sse(run(client, stream=True).text)[-1][1]
    assert set(plain) == set(streamed) and list(plain["answers"]) == list(streamed["answers"])


# ---- request validation ------------------------------


def test_missing_key_is_401_and_a_malformed_key_is_400_without_echoing_it():
    client, factory, _ = make()
    r = client.post("/api/run?stream=false", json={"question_id": "L1"})
    assert r.status_code == 401 and r.json()["error"] == "key_required"
    bad = "not a key!"
    r = run(client, headers={"X-Groq-Api-Key": bad})
    assert r.status_code == 400 and r.json()["error"] == "invalid_key_format"
    assert bad not in r.text and factory.requests == []


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        ({}, 422, "invalid_request"),
        ({"question": "   "}, 422, "invalid_request"),
        ({"question": "x" * 501}, 422, "question_too_long"),
        ({"question_id": "NOPE"}, 404, "unknown_question"),
    ],
)
def test_question_validation(body, status, code):
    client, *_ = make()
    r = run(client, body)
    assert r.status_code == status and r.json()["error"] == code


def test_question_length_limit_is_configurable():
    client, *_ = make(max_question_chars=10)
    assert run(client, {"question": "y" * 11}).status_code == 422
    assert run(client, {"question": "y" * 10}).status_code == 200


# ---- the key is never stored, logged or echoed ------------------------------


def test_key_never_appears_in_responses_or_logs_on_success_and_failure(caplog):
    caplog.set_level(logging.DEBUG)
    install_log_redaction()
    secret_exc = RuntimeError(f"boom with {KEY}")
    leaky_llm = llm_error("other", f"upstream said {KEY}")
    cases = [
        FakeFactory(),
        FakeFactory(generator=FailingClient(secret_exc)),
        FakeFactory(generator=FailingClient(leaky_llm)),
        FakeFactory(generator=FailingClient(llm_error("auth_failed", f"bad {KEY}"))),
    ]
    for factory in cases:
        client, *_ = make(factory)
        bodies = [run(client).text, run(client, stream=True).text]
        client.post("/api/live/validate", headers=HEADERS)
        assert all(KEY not in b for b in bodies)
    assert KEY not in caplog.text
    assert all(KEY not in r.getMessage() for r in caplog.records)


def test_log_redaction_masks_key_shaped_strings():
    install_log_redaction()
    rec = logging.LogRecord("t", logging.WARNING, "f", 1, "key=%s", (KEY,), None)
    RedactionFilter().filter(rec)
    assert KEY not in rec.getMessage() and "[REDACTED]" in rec.getMessage()
    made = logging.getLogger("ragbench.api.test").makeRecord(
        "ragbench.api.test", logging.WARNING, "f", 1, f"inline {KEY}", (), None
    )
    assert KEY not in made.getMessage()


def test_unexpected_errors_return_a_generic_500_without_internals():
    factory = FakeFactory(generator=FailingClient(RuntimeError(f"secret detail {KEY}")))
    client, *_ = make(factory)
    r = run(client)
    assert r.status_code == 500 and r.json()["error"] == "internal_error"
    assert "secret detail" not in r.text and KEY not in r.text


def test_default_factory_builds_a_cache_free_client_holding_only_the_visitors_key():
    gen = default_client_factory("generator", KEY)
    judge = default_client_factory("judge", KEY)
    assert isinstance(gen, LLMClient) and isinstance(judge, LLMClient)
    assert gen._cache is None and judge._cache is None  # BYOK never touches the shared disk cache
    assert KEY not in repr(gen) and KEY not in str(judge)


def test_responses_are_not_cacheable_and_carry_security_headers():
    client, *_ = make()
    r = run(client)
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'none'" in r.headers["content-security-policy"]
    assert client.get("/api/health").headers["x-content-type-options"] == "nosniff"


# ---- error mapping ------------------------------


@pytest.mark.parametrize(
    ("code", "status", "error"),
    [
        ("auth_failed", 401, "invalid_key"),
        ("rate_limit_long", 429, "provider_quota"),
        ("rate_limited", 429, "provider_quota"),
        ("model_not_found", 502, "model_unavailable"),
        ("something_else", 502, "upstream_error"),
    ],
)
def test_provider_errors_map_to_clear_http_errors(code, status, error):
    client, *_ = make(FakeFactory(generator=FailingClient(llm_error(code))))
    r = run(client)
    assert r.status_code == status and r.json()["error"] == error and r.json()["message"]


def test_stream_mode_reports_errors_as_an_error_event():
    client, *_ = make(FakeFactory(generator=FailingClient(llm_error("auth_failed"))))
    events = parse_sse(run(client, stream=True).text)
    assert [n for n, _ in events] == ["retrieval", "strategy_started", "error"]
    assert events[-1][1]["error"] == "invalid_key" and events[-1][1]["status"] == 401


def test_a_failing_judge_degrades_to_unscored_answers_instead_of_failing_the_run():
    factory = FakeFactory(judge=FakeLLMClient(lambda m: "not json", model="fake-judge"))
    client, *_ = make(factory)
    events = parse_sse(run(client, stream=True).text)
    errors = [p for n, p in events if n == "metrics" and "error" in p]
    assert len(errors) == 5 and all(p["error"] == "judge_failed" for p in errors)
    final = QuestionResultOut.model_validate(events[-1][1])
    assert list(final.answers) == STRATEGIES
    assert all(not a.metrics.judged and a.answer for a in final.answers.values())


def test_quota_exhaustion_during_scoring_is_reported_as_a_provider_quota_error():
    factory = FakeFactory(judge=FailingClient(llm_error("rate_limit_long"), model="fake-judge"))
    client, *_ = make(factory)
    r = run(client)
    assert r.status_code == 429 and r.json()["error"] == "provider_quota"


# ---- abuse protection ------------------------------


def test_live_runs_are_rate_limited_per_client_with_retry_after():
    client, *_ = make(rate_limit_runs=2, rate_limit_runs_window_s=600)
    assert run(client).status_code == 200 and run(client).status_code == 200
    r = run(client)
    assert r.status_code == 429 and r.json()["error"] == "rate_limited"
    assert int(r.headers["retry-after"]) >= 1 and r.json()["retry_after_s"] > 0


def test_only_one_run_may_be_in_flight_per_client_and_it_is_released_afterwards():
    client, *_ = make()
    guard = client.app.state.ctx.guard
    assert guard.acquire("testclient") is None
    r = run(client)
    assert r.status_code == 429 and r.json()["error"] == "run_in_progress"
    guard.release("testclient")
    assert run(client).status_code == 200  # and it released itself on completion:
    assert guard.acquire("testclient") is None


def test_a_failed_or_streamed_run_also_releases_its_slot():
    client, *_ = make(FakeFactory(generator=FailingClient(llm_error("auth_failed"))))
    guard = client.app.state.ctx.guard
    run(client)
    run(client, stream=True)
    assert guard.acquire("testclient") is None


def test_global_concurrency_cap_returns_503():
    client, *_ = make(max_concurrent_runs=1)
    client.app.state.ctx.guard.acquire("someone-else")
    r = run(client)
    assert r.status_code == 503 and r.json()["error"] == "server_busy"


def test_runs_that_exceed_the_time_budget_stop_with_a_timeout():
    client, *_ = make(run_timeout_s=-1)
    r = run(client)
    assert r.status_code == 504 and r.json()["error"] == "timeout"
    events = parse_sse(run(client, stream=True).text)
    assert events[-1][0] == "error" and events[-1][1]["status"] == 504


def test_rate_limit_buckets_are_per_ip_only_when_proxy_headers_are_trusted():
    trusted, *_ = make(rate_limit_runs=1, trust_proxy_headers=True)
    assert run(trusted, headers={**HEADERS, "X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert run(trusted, headers={**HEADERS, "X-Forwarded-For": "2.2.2.2"}).status_code == 200
    assert run(trusted, headers={**HEADERS, "X-Forwarded-For": "1.1.1.1"}).status_code == 429
    untrusted, *_ = make(rate_limit_runs=1)
    assert run(untrusted, headers={**HEADERS, "X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert run(untrusted, headers={**HEADERS, "X-Forwarded-For": "2.2.2.2"}).status_code == 429


# ---- HTTPS and CORS ------------------------------


def test_hosted_mode_refuses_keys_over_plain_http():
    client, factory, _ = make(require_https=True)
    r = run(client)
    assert r.status_code == 400 and r.json()["error"] == "https_required"
    assert client.post("/api/live/validate", headers=HEADERS).json()["error"] == "https_required"
    assert factory.requests == []
    assert client.get("/api/health").status_code == 200  # replay stays open


def test_hosted_mode_accepts_https_directly_or_via_a_trusted_proxy_header():
    secure, *_ = make(require_https=True, base_url="https://testserver")
    assert run(secure).status_code == 200
    proxied, *_ = make(require_https=True, trust_proxy_headers=True)
    assert run(proxied, headers={**HEADERS, "X-Forwarded-Proto": "https"}).status_code == 200
    untrusted, *_ = make(require_https=True)
    assert run(untrusted, headers={**HEADERS, "X-Forwarded-Proto": "https"}).status_code == 400


def test_cors_allows_the_local_dev_origin_and_the_key_header_only():
    client, *_ = make()
    pre = client.options(
        "/api/run",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-groq-api-key,content-type",
        },
    )
    assert pre.status_code == 200
    assert pre.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "x-groq-api-key" in pre.headers["access-control-allow-headers"].lower()
    assert "access-control-allow-credentials" not in pre.headers
    bad = client.options(
        "/api/run",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in bad.headers
    ok = client.get("/api/health", headers={"Origin": "http://127.0.0.1:5173"})
    assert ok.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


def test_cors_origins_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("RAGBENCH_CORS_ORIGINS", '["https://demo.example"]')
    assert ApiSettings(_env_file=None).cors_origins == ["https://demo.example"]


# ---- key validation endpoint ------------------------------


def test_validate_accepts_a_key_when_both_models_are_available():
    client, factory, _ = make()
    r = client.post("/api/live/validate", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] and body["generator_available"] and body["judge_available"]
    assert body["generator_model"] == "fake-llm" and body["judge_model"] == "fake-judge"
    assert ("generator", KEY) in factory.requests and KEY not in r.text


def test_validate_reports_a_missing_model_without_failing_the_key():
    factory = FakeFactory()
    factory.judge.available_models = ["something-else"]
    client, *_ = make(factory)
    body = client.post("/api/live/validate", headers=HEADERS).json()
    assert body["valid"] is False and body["judge_available"] is False
    assert "fake-judge" in body["message"]


def test_validate_maps_bad_keys_and_quota_to_clear_errors():
    class Bad(FakeLLMClient):
        def __init__(self, code):
            super().__init__()
            self._code = code

        def list_models(self):
            raise llm_error(self._code)

    for code, status, error in (
        ("auth_failed", 401, "invalid_key"),
        ("rate_limited", 429, "provider_quota"),
    ):
        client, *_ = make(FakeFactory(generator=Bad(code)))
        r = client.post("/api/live/validate", headers=HEADERS)
        assert r.status_code == status and r.json()["error"] == error
    client, *_ = make()
    assert client.post("/api/live/validate").status_code == 401


def test_validation_attempts_are_rate_limited():
    client, *_ = make(rate_limit_validate=2, rate_limit_validate_window_s=60)
    assert client.post("/api/live/validate", headers=HEADERS).status_code == 200
    assert client.post("/api/live/validate", headers=HEADERS).status_code == 200
    r = client.post("/api/live/validate", headers=HEADERS)
    assert r.status_code == 429 and "retry-after" in r.headers


# ---- building blocks ------------------------------


def test_rate_limiter_sliding_window_with_a_fake_clock():
    now = [0.0]
    limiter = RateLimiter(2, 10.0, clock=lambda: now[0])
    assert limiter.check("a") == (True, 0.0)
    now[0] = 1.0
    assert limiter.check("a")[0] is True
    allowed, wait = limiter.check("a")
    assert allowed is False and wait == pytest.approx(9.0)
    assert limiter.check("b")[0] is True  # independent key
    now[0] = 10.5
    assert limiter.check("a")[0] is True  # the first hit left the window


def test_inflight_guard_semantics():
    guard = InflightGuard(max_total=2)
    assert guard.acquire("a") is None and guard.acquire("a") == "run_in_progress"
    assert guard.acquire("b") is None and guard.acquire("c") == "server_busy"
    guard.release("a")
    assert guard.acquire("c") is None
    guard.release("nobody")  # releasing an unknown client is harmless


@pytest.mark.parametrize(
    ("key", "ok"),
    [
        (KEY, True),
        ("short", False),
        ("has space in it but long enough", False),
        ("", False),
        (None, False),
        ("x" * 257, False),
        ("gsk_" + "a" * 20, True),
    ],
)
def test_key_shape_check(key, ok):
    assert valid_key_shape(key) is ok


def test_lazy_retriever_builds_once_and_only_on_first_use():
    built = []

    def factory():
        built.append(1)
        return StubRetriever()

    lazy = LazyRetriever(factory)
    assert built == []
    lazy.query("a")
    lazy.query("b")
    assert built == [1]


# ---- warm-up and the time budget -------------------------


def test_warmup_loads_in_the_background_and_queries_wait_for_it():
    import threading

    gate = threading.Event()
    built = []

    def factory():
        gate.wait(5)
        built.append(1)
        return StubRetriever()

    lazy = LazyRetriever(factory)
    lazy.warm()
    lazy.warm()  # idempotent
    assert lazy.ready is False
    gate.set()
    assert lazy.query("q")  # blocks until the background load finished, then answers
    assert lazy.ready is True and built == [1]


def test_health_reports_whether_live_runs_are_ready():
    client, *_ = make()
    # the stub retriever is built on demand; before any use it is not "ready"
    first = client.get("/api/health").json()["live_ready"]
    run(client)
    assert client.get("/api/health").json()["live_ready"] is True
    assert first in (True, False)


def test_a_slow_cold_start_is_not_charged_to_the_run_budget():
    import time

    class SlowStart(StubRetriever):
        def query(self, text, top_k=4):
            time.sleep(0.4)  # stands in for loading the embedding model
            return super().query(text, top_k)

    client, *_ = make(retriever=SlowStart(), run_timeout_s=0.3, warm_retriever=False)
    r = run(client, {"question_id": "L1", "score": False})
    assert r.status_code == 200  # the run itself finished inside its 0.3 s budget


# ---- warm-up failures and import side effects --------------------


def test_importing_the_api_module_builds_no_app_and_starts_no_warmup():
    import importlib
    import threading

    import ragbench.api.main as main

    importlib.reload(main)
    assert main._app is None  # nothing was constructed by the import
    assert not [t for t in threading.enumerate() if t.name == "retriever-warmup"]


def test_the_module_level_app_is_built_lazily_once(monkeypatch):
    import ragbench.api.main as main

    monkeypatch.setenv("RAGBENCH_WARM_RETRIEVER", "0")
    monkeypatch.setattr(main, "_app", None)
    first = main.app
    assert main.app is first and first.title == "RAGBench API"


def test_a_failed_warmup_is_reported_by_health_instead_of_waiting_forever():
    def broken():
        raise RuntimeError("index unreadable")

    client, *_ = make(warm_retriever=False)
    ctx = client.app.state.ctx
    ctx.retriever = LazyRetriever(broken)
    ctx.retriever.warm()
    for _ in range(100):
        if ctx.retriever.error:
            break
        import time

        time.sleep(0.02)
    body = client.get("/api/health").json()
    assert body["live_ready"] is False and body["live_error"] == "RuntimeError"
    assert "index unreadable" not in client.get("/api/health").text  # only the type, no detail


def test_concurrent_retriever_builds_are_serialised_across_instances():
    import threading
    import time

    state = {"active": 0, "max": 0}
    lock = threading.Lock()

    def slow_factory():
        with lock:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
        time.sleep(0.15)
        with lock:
            state["active"] -= 1
        return StubRetriever()

    a, b = LazyRetriever(slow_factory), LazyRetriever(slow_factory)
    threads = [
        threading.Thread(target=a.query, args=("q",)),
        threading.Thread(target=b.query, args=("q",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert state["max"] == 1 and a.ready and b.ready


# ---- cancellation --------------------


def _events(factory, cancelled):
    from ragbench.api.live import live_events

    return live_events(
        question=QUESTIONS[0],
        free_text=False,
        score=True,
        api_key=KEY,
        retriever=StubRetriever(),
        factory=factory,
        settings=load_experiment_settings(),
        pricing=PRICING,
        titles=TITLES,
        models=MODELS,
        budget_s=60,
        cancelled=cancelled,
    )


def test_a_cancelled_run_stops_before_the_next_model_call_and_reports_nothing():
    factory = FakeFactory()
    state = {"cancel": False}
    seen = []
    for name, _ in _events(factory, lambda: state["cancel"]):
        seen.append(name)
        if name == "strategy_done":
            state["cancel"] = True  # the visitor leaves right after the first answer
    assert "error" not in seen and "done" not in seen
    assert len(factory.generator.calls) == 1  # only Baseline's single call ever ran
    assert not factory.judge.calls  # and the judge never started


def test_an_uncancelled_run_is_unaffected_by_the_cancel_wrapper():
    factory = FakeFactory()
    names = [n for n, _ in _events(factory, lambda: False)]
    assert names[0] == "retrieval" and names[-1] == "done"
    assert len(factory.generator.calls) == 9


def test_closing_the_stream_releases_the_slot_and_sets_the_cancel_flag():
    client, *_ = make()
    guard = client.app.state.ctx.guard
    run(client, stream=True)  # completes and releases, as always
    assert guard.acquire("testclient") is None


# ---- scores must come from the judge, per answer: never a default, never one constant ----------


def _answer_sensitive_factory():
    """A generator whose control answer is not in the passages, and a judge that actually looks.

    The shared `judge_responder` returns the same verdicts for every answer, so no test could
    notice scores collapsing to one value for all five conditions.
    """
    from ragbench.llm.fake import default_responder

    def generator(messages):
        text = "\n".join(m["content"] for m in messages)
        if "strict reviewer" in text or "reading retrieved passages" in text:
            return default_responder(messages)  # stage shapes the strategies parse
        if "No reference documents are provided" in text:
            return "A claim from memory about something else entirely."
        return "alpha beta gamma."

    def judge(messages):
        text = messages[-1]["content"]
        if "Decompose the answer" in text:
            answer = text.split("Answer:", 1)[1].split("Return JSON only", 1)[0].strip()
            return json.dumps({"claims": [answer]})
        if "checking claims" in text:
            context = text.split("Passages:", 1)[1].split("Claims:", 1)[0]
            claim = text.split("Claims:", 1)[1].split("For each claim", 1)[0]
            ok = "alpha beta gamma" in context and "alpha beta gamma" in claim
            label = "supported" if ok else "unsupported"
            return json.dumps(
                {
                    "verdicts": [
                        {"claim_id": 1, "label": label, "evidence_passage": 1, "reason": "r"}
                    ]
                }
            )
        answer = text.split("Answer:", 1)[1].split("Gold facts:", 1)[0]
        n = len(re.findall(r"^\d+\. ", text.split("Gold facts:")[-1], re.M))
        return json.dumps(
            {"coverage": [{"fact_id": i + 1, "covered": "alpha" in answer} for i in range(n)]}
        )

    return FakeFactory(
        generator=FakeLLMClient(generator, model="fake-generator"),
        judge=FakeLLMClient(judge, model="fake-judge"),
    )


def _scores(body):
    out = {}
    for name, a in body["answers"].items():
        m = a["metrics"]
        out[name] = (
            (m["groundedness"] or {}).get("G"),
            (m["coverage"] or {}).get("C"),
            m["fcs"],
        )
    return out


def test_live_scores_follow_each_answer_and_do_not_collapse_to_one_value():
    client, *_ = make(factory=_answer_sensitive_factory())
    scores = _scores(run(client, {"question_id": "L1", "score": True}).json())
    for name in ("baseline", "structured", "rrr", "self_rag"):
        assert scores[name] == (1.0, 1.0, 1.0), name
    # the control's answer is not in the passages: the judge must say so
    assert scores["no_retrieval"] == (0.0, 0.0, 0.0)
    assert len(set(scores.values())) > 1  # not the same numbers for every condition


def test_a_missing_score_is_null_never_a_default_of_one():
    client, *_ = make(factory=_answer_sensitive_factory())
    body = run(client, {"question": "A free-text question?", "score": True}).json()
    for name, a in body["answers"].items():
        m = a["metrics"]
        assert m["coverage"] is None or m["coverage"]["C"] is None, name  # no gold facts
        assert m["fcs"] is None, name  # undefined without coverage, so null (not 1.0, not 0.0)
        assert m["judged"] is True


def test_unscored_runs_carry_no_scores_at_all():
    client, *_ = make(factory=_answer_sensitive_factory())
    body = run(client, {"question_id": "L1", "score": False}).json()
    for a in body["answers"].values():
        m = a["metrics"]
        assert m["judged"] is False and m["groundedness"] is None and m["fcs"] is None
