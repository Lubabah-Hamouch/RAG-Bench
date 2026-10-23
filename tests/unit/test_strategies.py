from __future__ import annotations

import pytest

from ragbench.llm.fake import FakeLLMClient
from ragbench.pipeline import run_question
from ragbench.prompts import REFUSAL
from ragbench.strategies.base import format_context
from ragbench.strategies.registry import ALL_NAMES, REGISTRY, get_strategy
from ragbench.strategies.self_rag import parse_critique
from ragbench.types import Chunk, RetrievedChunk

QUESTION = "What is the difference between RAG-Sequence and RAG-Token?"
TEXTS = [
    "RAG-Sequence uses the same retrieved document for every token of the output.",
    "RAG-Token can draw on a different document for each generated token.",
    "Both variants marginalise over the top-k retrieved documents.",
    "The retriever is a bi-encoder initialised from DPR.",
]


def make_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(i, 0.9 - 0.05 * i, Chunk(f"d:p1:c{i}", "d", "T", "core", 1, t))
        for i, t in enumerate(TEXTS, start=1)
    ]


class StubRetriever:
    def __init__(self):
        self.queries: list[tuple[str, int]] = []
        self.chunks = make_chunks()

    def query(self, text, top_k=4):
        self.queries.append((text, top_k))
        return self.chunks


def run(name, client=None, **kw):
    client = client or FakeLLMClient()
    result = get_strategy(name).run(
        QUESTION, make_chunks(), client=client, temperature=0.5, max_tokens=700, **kw
    )
    return result, client


def prompt_text(call):
    return "\n".join(m["content"] for m in call["messages"])


def test_registry_order_and_names():
    assert ALL_NAMES == ["baseline", "structured", "rrr", "self_rag", "no_retrieval"]
    assert all(REGISTRY[n].name == n for n in ALL_NAMES)
    with pytest.raises(KeyError):
        get_strategy("nope")


@pytest.mark.parametrize(
    ("name", "n_calls"),
    [("baseline", 1), ("structured", 1), ("rrr", 3), ("self_rag", 3), ("no_retrieval", 1)],
)
def test_call_counts_and_aggregates(name, n_calls):
    result, client = run(name)
    assert result.n_calls == len(client.calls) == n_calls
    assert len(result.trace) == n_calls
    assert result.prompt_tokens == sum(c.prompt_tokens for c in result.calls) > 0
    assert result.answer == result.trace[-1].text


def test_retrieval_strategies_put_all_four_passages_in_every_prompt():
    for name in ["baseline", "structured", "rrr", "self_rag"]:
        _, client = run(name)
        for call in client.calls:
            for text in TEXTS:
                assert text in prompt_text(call), (name, text)


def test_control_prompt_contains_no_retrieved_text():
    _, client = run("no_retrieval")
    text = prompt_text(client.calls[0])
    assert QUESTION in text and not any(t in text for t in TEXTS)


def test_baseline_and_structured_use_their_own_system_prompts():
    _, c = run("baseline")
    assert c.calls[0]["messages"][0]["role"] == "system"
    assert "one or two sentences" in c.calls[0]["messages"][0]["content"]
    _, c = run("structured")
    system = c.calls[0]["messages"][0]["content"]
    assert all(h in system for h in ["## Summary", "## Key Points", "## Elaboration"])


def test_rrr_threads_each_stage_into_the_next():
    result, client = run("rrr")
    stages = [s.name for s in result.trace]
    assert stages == ["identify", "synthesize", "refine"]
    assert result.trace[0].text in prompt_text(client.calls[1])
    assert result.trace[1].text in prompt_text(client.calls[2])
    assert result.answer == result.trace[2].text


def test_self_rag_critiques_the_draft_then_refines_with_both():
    result, client = run("self_rag")
    assert [s.name for s in result.trace] == ["draft", "critique", "refine"]
    draft, critique = result.trace[0].text, result.trace[1].text
    assert draft in prompt_text(client.calls[1])
    refine_prompt = prompt_text(client.calls[2])
    assert draft in refine_prompt and critique in refine_prompt
    assert result.critique == critique and result.answer == result.trace[2].text


def test_self_rag_display_shows_answer_then_critique_but_answer_excludes_it():
    result, _ = run("self_rag")
    shown = result.display()
    assert shown.index("**Answer**") < shown.index("**Self-critique**")
    assert "Passage relevance" in shown and "Passage relevance" not in result.answer


def test_other_strategies_display_just_the_answer():
    result, _ = run("baseline")
    assert result.display() == result.answer


def test_settings_seed_and_trial_salt_are_forwarded():
    _, client = run("rrr", seed=11, trial=3)
    assert all(c["temperature"] == 0.5 and c["max_tokens"] == 700 for c in client.calls)
    assert all(c["seed"] == 11 for c in client.calls)
    salts = [c["cache_salt"] for c in client.calls]
    assert salts == ["rrr:trial3:identify", "rrr:trial3:synthesize", "rrr:trial3:refine"]


def test_different_trials_get_different_cache_salts():
    _, a = run("baseline", trial=0)
    _, b = run("baseline", trial=1)
    assert a.calls[0]["cache_salt"] != b.calls[0]["cache_salt"]


def refusing(messages):
    return REFUSAL


@pytest.mark.parametrize("name", ["baseline", "structured", "rrr", "self_rag"])
def test_abstained_flag_follows_the_refusal_sentence(name):
    result, _ = run(name, client=FakeLLMClient(refusing))
    assert result.abstained
    assert not run(name)[0].abstained


def test_no_retrieval_control_is_not_flagged_as_context_abstention_by_default():
    assert not run("no_retrieval")[0].abstained


def test_format_context_numbers_passages():
    out = format_context(make_chunks())
    assert out.startswith("[1] RAG-Sequence") and "\n\n[4] The retriever" in out


def test_pipeline_retrieves_once_and_shares_identical_chunks():
    retriever = StubRetriever()
    client = FakeLLMClient()
    result = run_question(QUESTION, retriever, client, top_k=4)
    assert len(retriever.queries) == 1 and retriever.queries[0] == (QUESTION, 4)
    assert list(result.results) == ALL_NAMES
    assert len(client.calls) == 1 + 1 + 3 + 3 + 1
    for call in client.calls:
        text = prompt_text(call)
        has_ctx = TEXTS[0] in text
        assert has_ctx or "No reference documents" in text


def test_pipeline_subset_of_strategies():
    result = run_question(
        QUESTION, StubRetriever(), FakeLLMClient(), strategies=["baseline", "rrr"]
    )
    assert list(result.results) == ["baseline", "rrr"]


def test_parse_critique_extracts_fields():
    result, _ = run("self_rag")
    parsed = parse_critique(result.critique or "")
    assert parsed["answerable"] == "Yes" and parsed["utility"] == "4"
    assert parsed["passage_relevance"] == ["Relevant", "Relevant", "Irrelevant", "Irrelevant"]
    assert parse_critique("garbage")["passage_relevance"] == []
