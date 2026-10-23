"""Judge, groundedness, coverage and scoring, using a scripted fake model (no network)."""

from __future__ import annotations

import json

import pytest

from ragbench.llm.fake import FakeLLMClient
from ragbench.metrics.cost import ModelPrice
from ragbench.metrics.coverage import (
    abstention_correct,
    faithful_coverage,
    score_coverage,
)
from ragbench.metrics.groundedness import score_groundedness
from ragbench.metrics.judge import Judge, JudgeError
from ragbench.metrics.scoring import evaluate_answer
from ragbench.prompts import REFUSAL
from ragbench.strategies.base import CallRecord, StrategyResult
from ragbench.types import Chunk, RetrievedChunk

QUESTION = "What differs between Assert and Suggest?"
PASSAGES = [
    "Assert halts execution.",
    "Suggest only logs.",
    "Both retry.",
    "Handlers are composable.",
]
CHUNKS = [
    RetrievedChunk(i, 0.8, Chunk(f"d:p1:c{i}", "d", "T", "core", 1, t))
    for i, t in enumerate(PASSAGES, start=1)
]


def judge_for(claims, verdicts=None, coverage=None, model="openai/gpt-oss-20b"):
    """A fake judge model that answers each of the three judge prompts from fixed data."""

    def responder(messages):
        text = messages[-1]["content"]
        if "Decompose the answer" in text:
            return json.dumps({"claims": claims})
        if "checking claims" in text:
            return json.dumps({"verdicts": verdicts})
        if "Decide whether the answer states each gold fact" in text:
            return json.dumps({"coverage": coverage})
        raise AssertionError("unexpected prompt")

    client = FakeLLMClient(responder, model=model)
    return Judge(client), client


def v(i, label, passage=1):
    return {"claim_id": i, "label": label, "evidence_passage": passage, "reason": "r"}


class Scripted:
    """Responder that replays a list of replies (str, or (str, finish_reason))."""

    def __init__(self, replies):
        self.replies = list(replies)

    def __call__(self, messages):
        return self.replies.pop(0)


# ---- Judge ----------------------------------------------------------------------------------


def test_judge_parses_plain_fenced_and_prose_wrapped_json():
    from pydantic import BaseModel

    class M(BaseModel):
        x: int

    for reply in ['{"x": 1}', '```json\n{"x": 1}\n```', 'Sure! Here it is: {"x": 1} Done.']:
        judge = Judge(FakeLLMClient(Scripted([reply])))
        value, calls = judge.ask("t", "p", M)
        assert value.x == 1 and len(calls) == 1


def test_judge_requests_json_mode_at_temperature_zero():
    judge, client = judge_for(["c"], [v(1, "supported")])
    score_groundedness(QUESTION, "An answer.", CHUNKS, judge)
    assert all(c["json_mode"] and c["temperature"] == 0.0 for c in client.calls)


def test_truncated_reasoning_reply_is_retried_with_a_doubled_budget():
    from pydantic import BaseModel

    class M(BaseModel):
        x: int

    client = FakeLLMClient(Scripted([("", "length"), ('{"x": 2}', "stop")]))
    judge = Judge(client, max_tokens=1500)
    value, calls = judge.ask("t", "p", M)
    assert value.x == 2 and len(calls) == 2
    assert [c["max_tokens"] for c in client.calls] == [1500, 3000]
    assert client.calls[0]["cache_salt"] != client.calls[1]["cache_salt"]


def test_budget_doubling_stops_at_the_cap():
    from pydantic import BaseModel

    class M(BaseModel):
        x: int

    client = FakeLLMClient(Scripted([("", "length")] * 3))
    judge = Judge(client, max_tokens=1500, max_tokens_cap=2000, max_retries=2)
    with pytest.raises(JudgeError, match="budget exhausted"):
        judge.ask("t", "p", M)
    assert [c["max_tokens"] for c in client.calls] == [1500, 2000, 2000]


def test_invalid_json_is_retried_then_succeeds():
    from pydantic import BaseModel

    class M(BaseModel):
        x: int

    client = FakeLLMClient(Scripted(["not json", '{"x": 5}']))
    value, calls = Judge(client).ask("t", "p", M)
    assert value.x == 5 and len(calls) == 2
    assert client.calls[0]["max_tokens"] == client.calls[1]["max_tokens"]


def test_schema_violations_exhaust_retries_with_a_clear_error():
    from pydantic import BaseModel

    class M(BaseModel):
        x: int

    client = FakeLLMClient(Scripted(['{"y": 1}'] * 3))
    with pytest.raises(JudgeError, match="invalid JSON"):
        Judge(client).ask("t", "p", M)


# ---- Groundedness ---------------------------------------------------------------------------


def test_groundedness_is_supported_over_claims():
    judge, _ = judge_for(
        ["A", "B", "C"], [v(1, "supported"), v(2, "supported"), v(3, "unsupported", None)]
    )
    r = score_groundedness(QUESTION, "some answer", CHUNKS, judge)
    assert (r.n_claims, r.n_supported, r.n_unsupported, r.n_contradicted) == (3, 2, 1, 0)
    assert r.groundedness == pytest.approx(2 / 3)
    assert r.hallucination_rate == pytest.approx(1 / 3)
    assert r.contradiction_rate == 0.0 and not r.abstained and len(r.judge_calls) == 2


def test_contradicted_claims_count_toward_hallucination_and_contradiction_rate():
    judge, _ = judge_for(["A", "B"], [v(1, "supported"), v(2, "contradicted")])
    r = score_groundedness(QUESTION, "some answer", CHUNKS, judge)
    assert r.groundedness == 0.5 and r.hallucination_rate == 0.5 and r.contradiction_rate == 0.5


def test_labels_are_normalised_and_unknown_labels_rejected():
    judge, _ = judge_for(["A"], [v(1, " Supported ")])
    assert score_groundedness(QUESTION, "x", CHUNKS, judge).n_supported == 1
    bad, _ = judge_for(["A"], [v(1, "maybe")])
    with pytest.raises(JudgeError):
        score_groundedness(QUESTION, "x", CHUNKS, bad)


def test_missing_verdicts_count_as_unsupported_and_are_flagged():
    judge, _ = judge_for(["A", "B", "C"], [v(1, "supported"), v(3, "supported")])
    r = score_groundedness(QUESTION, "x", CHUNKS, judge)
    assert r.n_missing_verdicts == 1 and r.n_unsupported == 1
    assert r.groundedness == pytest.approx(2 / 3)
    assert [x.claim_id for x in r.verdicts] == [1, 2, 3]


def test_no_claims_means_abstained_and_undefined_not_zero():
    judge, client = judge_for([], None)
    r = score_groundedness(QUESTION, "I would rather not say.", CHUNKS, judge)
    assert r.abstained and r.groundedness is None and r.hallucination_rate is None
    assert len(client.calls) == 1  # extraction only; verification is skipped


@pytest.mark.parametrize(
    "answer",
    [
        REFUSAL,
        f"## Summary\n{REFUSAL}\n\n## Key Points\nN/A\n\n## Elaboration\nN/A",
    ],
)
def test_refusals_short_circuit_without_calling_the_judge(answer):
    judge, client = judge_for(["never used"], None)
    r = score_groundedness(QUESTION, answer, CHUNKS, judge)
    assert r.abstained and not client.calls and r.groundedness is None


def test_a_long_answer_that_merely_mentions_the_refusal_sentence_is_still_scored():
    judge, client = judge_for(["A"], [v(1, "supported")])
    answer = "Some real content. " * 30 + REFUSAL
    r = score_groundedness(QUESTION, answer, CHUNKS, judge)
    assert not r.abstained and len(client.calls) == 2


def test_verification_prompt_carries_all_passages_and_numbered_claims():
    judge, client = judge_for(
        ["First claim", "Second claim"], [v(1, "supported"), v(2, "supported")]
    )
    score_groundedness(QUESTION, "x", CHUNKS, judge)
    prompt = client.calls[1]["messages"][-1]["content"]
    assert all(p in prompt for p in PASSAGES)
    assert "1. First claim" in prompt and "2. Second claim" in prompt


# ---- Coverage -------------------------------------------------------------------------------


def test_coverage_is_covered_over_gold_facts():
    judge, _ = judge_for(
        [],
        None,
        coverage=[
            {"fact_id": 1, "covered": True},
            {"fact_id": 2, "covered": False},
            {"fact_id": 3, "covered": True},
        ],
    )
    r = score_coverage(QUESTION, "an answer", ["f1", "f2", "f3"], judge)
    assert r.covered == [True, False, True] and r.coverage == pytest.approx(2 / 3)


def test_facts_the_judge_skips_are_not_covered():
    judge, _ = judge_for([], None, coverage=[{"fact_id": 2, "covered": True}])
    assert score_coverage(QUESTION, "x", ["f1", "f2"], judge).covered == [False, True]


def test_no_gold_facts_means_undefined_coverage_and_no_call():
    judge, client = judge_for([], None)
    r = score_coverage(QUESTION, "x", [], judge)
    assert r.coverage is None and not client.calls


def test_a_refusal_covers_nothing_and_costs_no_calls():
    judge, client = judge_for([], None)
    r = score_coverage(QUESTION, REFUSAL, ["f1", "f2"], judge)
    assert r.coverage == 0.0 and not client.calls


@pytest.mark.parametrize(
    ("g", "c", "expected"),
    [
        (1.0, 1.0, 1.0),
        (0.5, 1.0, 2 / 3),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (None, 1.0, None),
        (1.0, None, None),
    ],
)
def test_faithful_coverage_is_the_harmonic_mean(g, c, expected):
    assert (
        faithful_coverage(g, c) == pytest.approx(expected)
        if expected is not None
        else faithful_coverage(g, c) is None
    )


def test_a_short_grounded_but_incomplete_answer_scores_low_fcs():
    assert faithful_coverage(1.0, 0.2) == pytest.approx(2 * 0.2 / 1.2)


def test_abstention_correctness():
    assert abstention_correct(True, True) and abstention_correct(False, False)
    assert not abstention_correct(True, False) and not abstention_correct(False, True)


# ---- Scoring --------------------------------------------------------------------------------


PRICING = {
    "qwen/qwen3.8-27b": ModelPrice(1.0, 4.0),
    "openai/gpt-oss-20b": ModelPrice(0.1, 0.4),
}


def make_result(answer, critique=None):
    calls = [CallRecord("answer", "qwen/qwen3.8-27b", 0.5, 1000, 200)]
    return StrategyResult("baseline", answer, calls=calls, critique=critique)


def test_evaluate_answer_combines_all_metrics():
    judge, client = judge_for(
        ["A", "B"],
        [v(1, "supported"), v(2, "unsupported", None)],
        coverage=[{"fact_id": 1, "covered": True}, {"fact_id": 2, "covered": True}],
    )
    result = make_result("Assert halts and Suggest logs. " * 3)
    m = evaluate_answer(
        QUESTION, result, CHUNKS, judge=judge, gold_facts=["g1", "g2"], pricing=PRICING
    )
    assert m.groundedness.groundedness == 0.5 and m.coverage.coverage == 1.0
    assert m.fcs == pytest.approx(2 * 0.5 * 1.0 / 1.5)
    assert m.gen_cost_usd == pytest.approx((1000 * 1.0 + 200 * 4.0) / 1e6)
    assert m.gen_latency_s == 0.5 and m.gen_prompt_tokens == 1000
    assert m.cost_per_supported_claim == pytest.approx(m.gen_cost_usd / 1)
    assert m.judge_cost_usd is not None and m.judge_cost_usd > 0
    assert m.judge_prompt_tokens > 0 and not m.abstained
    assert m.lexical.words > 0


def test_judge_cost_is_separate_from_generation_cost():
    judge, _ = judge_for(["A"], [v(1, "supported")], coverage=[{"fact_id": 1, "covered": True}])
    m = evaluate_answer(
        QUESTION, make_result("x " * 20), CHUNKS, judge=judge, gold_facts=["g"], pricing=PRICING
    )
    assert m.gen_cost_usd == pytest.approx(0.0018)  # judge calls are not included


def test_self_rag_critique_is_never_shown_to_the_judge():
    judge, client = judge_for(
        ["A"], [v(1, "supported")], coverage=[{"fact_id": 1, "covered": True}]
    )
    result = make_result("The refined answer.", critique="SECRET-CRITIQUE-TEXT")
    evaluate_answer(QUESTION, result, CHUNKS, judge=judge, gold_facts=["g"], pricing=PRICING)
    assert not any("SECRET-CRITIQUE-TEXT" in c["messages"][-1]["content"] for c in client.calls)


def test_unknown_model_price_gives_none_costs_not_a_crash():
    judge, _ = judge_for(
        ["A"], [v(1, "supported")], coverage=[{"fact_id": 1, "covered": True}], model="mystery"
    )
    m = evaluate_answer(
        QUESTION, make_result("x " * 20), CHUNKS, judge=judge, gold_facts=["g"], pricing=PRICING
    )
    assert m.judge_cost_usd is None and m.gen_cost_usd is not None


def test_a_refusal_is_scored_as_abstention_with_no_judge_spend():
    judge, client = judge_for([], None)
    m = evaluate_answer(
        QUESTION, make_result(REFUSAL), CHUNKS, judge=judge, gold_facts=["g"], pricing=PRICING
    )
    assert m.abstained and m.fcs is None and m.coverage.coverage == 0.0
    assert not client.calls and m.judge_prompt_tokens == 0 and m.judge_cost_usd == 0.0


# ---- Provider JSON rejection (HTTP 400 json_validate_failed) ---------------------------------


class RaisingThenOk:
    """Fake client whose first calls raise LLMError, then answers normally."""

    model = "fake"

    def __init__(self, errors, reply='{"x": 7}'):
        self.errors = list(errors)
        self.reply = reply
        self.budgets: list[int] = []

    def chat(self, messages, *, temperature, max_tokens, json_mode=False, seed=None, cache_salt=""):
        from ragbench.llm.client import LLMResponse

        self.budgets.append(max_tokens)
        if self.errors:
            raise self.errors.pop(0)
        return LLMResponse(self.reply, "fake", 0.0, 10, 5, finish_reason="stop")


def _json_error(failed=None):
    from ragbench.llm.client import LLMError

    return LLMError(
        "400 json_validate_failed", code="json_validate_failed", failed_generation=failed
    )


def test_provider_json_rejection_is_retried_with_a_larger_budget():
    from pydantic import BaseModel

    class M(BaseModel):
        x: int

    client = RaisingThenOk([_json_error()])
    value, calls = Judge(client, max_tokens=1500).ask("t", "p", M)
    assert value.x == 7 and client.budgets == [1500, 3000]
    assert len(calls) == 1  # only the successful call produced a record


def test_provider_json_rejection_salvages_parseable_failed_generation():
    from pydantic import BaseModel

    class M(BaseModel):
        x: int

    client = RaisingThenOk([_json_error('```json\n{"x": 3}\n```')])
    value, _ = Judge(client).ask("t", "p", M)
    assert value.x == 3 and len(client.budgets) == 1


def test_other_llm_errors_are_not_swallowed():
    from pydantic import BaseModel

    from ragbench.llm.client import LLMError

    class M(BaseModel):
        x: int

    client = RaisingThenOk([LLMError("auth failed", code="invalid_api_key")])
    with pytest.raises(LLMError, match="auth failed"):
        Judge(client).ask("t", "p", M)


def test_repeated_provider_rejections_raise_a_judge_error():
    from pydantic import BaseModel

    class M(BaseModel):
        x: int

    client = RaisingThenOk([_json_error()] * 3)
    with pytest.raises(JudgeError, match="rejected the JSON"):
        Judge(client, max_retries=2).ask("t", "p", M)


# ---- Context support (does the retrieved context contain each gold fact?) --------------------


def test_context_support_recall_is_supported_over_gold_facts():
    from ragbench.metrics.coverage import score_context_support

    def responder(messages):
        return json.dumps(
            {"support": [{"fact_id": 1, "supported": True}, {"fact_id": 2, "supported": False}]}
        )

    judge = Judge(FakeLLMClient(responder))
    r = score_context_support("q", "[1] passage", ["f1", "f2"], judge)
    assert r.supported == [True, False] and r.recall == 0.5 and len(r.judge_calls) == 1


def test_context_support_skips_the_judge_without_gold_facts_and_treats_missing_as_absent():
    from ragbench.metrics.coverage import score_context_support

    client = FakeLLMClient(lambda m: json.dumps({"support": [{"fact_id": 2, "supported": True}]}))
    judge = Judge(client)
    assert score_context_support("q", "ctx", [], judge).recall is None and not client.calls
    assert score_context_support("q", "ctx", ["a", "b"], judge).supported == [False, True]


def test_context_support_prompt_contains_passages_and_numbered_facts():
    from ragbench.metrics.coverage import score_context_support

    client = FakeLLMClient(lambda m: json.dumps({"support": []}))
    score_context_support(
        "my question", "[1] PASSAGE-TEXT", ["first fact", "second fact"], Judge(client)
    )
    prompt = client.calls[0]["messages"][-1]["content"]
    assert "PASSAGE-TEXT" in prompt and "1. first fact" in prompt and "2. second fact" in prompt
    assert "my question" in prompt
