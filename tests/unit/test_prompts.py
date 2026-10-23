import pytest

from ragbench.prompts import REFUSAL, is_refusal, load_template, placeholders, render

ALL_TEMPLATES = [
    "baseline.system",
    "baseline.user",
    "structured.system",
    "structured.user",
    "rrr.identify",
    "rrr.synthesize",
    "rrr.refine",
    "selfrag.draft",
    "selfrag.critique",
    "selfrag.refine",
    "control.system",
    "control.user",
]


@pytest.mark.parametrize("name", ALL_TEMPLATES)
def test_template_exists_and_is_nonempty(name):
    assert load_template(name).strip()


def test_render_fills_every_placeholder():
    for name in ALL_TEMPLATES:
        out = render(name, **{p: f"<{p}>" for p in placeholders(name)})
        for p in placeholders(name):
            assert f"<{p}>" in out and "{" + p + "}" not in out


def test_render_is_strict_about_missing_and_extra_values():
    with pytest.raises(KeyError):
        render("baseline.user", question="q")
    with pytest.raises(KeyError):
        render("baseline.user", question="q", context="c", extra="x")


def test_inserted_text_is_not_rescanned():
    out = render("baseline.user", question="{context}", context="{question}")
    assert "Question: {context}" in out and "Context:\n{question}" in out


@pytest.mark.parametrize(
    "name",
    ["baseline.system", "structured.system", "rrr.synthesize", "selfrag.draft", "selfrag.refine"],
)
def test_every_retrieval_strategy_carries_the_exact_refusal_sentence(name):
    assert REFUSAL in load_template(name)


def test_partial_answer_instruction_is_present_for_answering_stages():
    for name in ["baseline.system", "structured.system", "rrr.synthesize", "selfrag.draft"]:
        assert "only part" in load_template(name)


def test_rrr_refine_preserves_a_refusal_and_selfrag_critique_checks_answerability():
    assert REFUSAL in load_template("rrr.refine")
    assert "Answerable from context" in load_template("selfrag.critique")


def test_control_prompt_has_no_context_placeholder_and_admits_uncertainty():
    assert "context" not in placeholders("control.user")
    assert "not sure" in load_template("control.system")


def test_is_refusal():
    assert is_refusal(REFUSAL)
    assert is_refusal("## Summary\n" + REFUSAL + "\n\n## Key Points\nN/A")
    assert is_refusal("the provided context does not contain this information")
    assert not is_refusal("RAG-Token uses a different document per token.")
