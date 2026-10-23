"""`ragbench ask` end to end in its default fake mode, against the real index (no network)."""

from __future__ import annotations

import pytest

from ragbench.cli import main

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module", autouse=True)
def built_index():
    # Ensure the index exists (cheap if it already does).
    from ragbench.cli import _cmd_ingest

    _cmd_ingest(None, None)


def test_ask_defaults_to_the_fake_model_and_prints_every_strategy(capsys):
    assert main(["ask", "What is Weaviate?"]) == 0
    out = capsys.readouterr()
    assert "fake LLM" in out.err
    for label in ["Baseline", "Structured", "RRR", "Self-RAG (prompted)", "No retrieval (control)"]:
        assert f"=== {label}" in out.out
    assert "weaviate_docs" in out.out


def test_ask_strategy_subset_and_trace(capsys):
    main(["ask", "What is Weaviate?", "--strategies", "rrr", "--trace"])
    out = capsys.readouterr().out
    assert "=== RRR" in out and "=== Baseline" not in out
    assert "--- stage: identify" in out and "--- stage: refine" in out
