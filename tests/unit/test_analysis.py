"""Analysis, report and judge-validation code on synthetic benchmark output (fakes, no network)."""

from __future__ import annotations

import csv
import hashlib
import json
import re

import numpy as np
import pytest

from ragbench.analysis import (
    context_recall,
    fact_in_chunks,
    sample_claims_for_labeling,
    summarize,
)
from ragbench.benchmark import run_benchmark
from ragbench.config import load_experiment_settings
from ragbench.llm.fake import FakeLLMClient
from ragbench.metrics.cost import ModelPrice
from ragbench.metrics.judge import Judge
from ragbench.metrics.validation import agreement, load_labelled
from ragbench.prompts import REFUSAL
from ragbench.questions import GoldFact, Question
from ragbench.report import make_figures, render_markdown, write_outputs
from ragbench.types import Chunk, RetrievedChunk

SETTINGS = load_experiment_settings()
PRICING = {"fake-llm": ModelPrice(1.0, 2.0), "fake-judge": ModelPrice(0.1, 0.2)}


class HashEmbedder:
    name = "hash"

    def encode(self, texts):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


def gold(text, quote):
    return GoldFact(text=text, doc_id="d", page=1, quote=quote)


QUESTIONS = [
    Question(
        id="L1", type="lookup", question="Lookup one?", gold_facts=[gold("f1", "alpha beta gamma")]
    ),
    Question(
        id="L2", type="lookup", question="Lookup two?", gold_facts=[gold("f2", "delta epsilon")]
    ),
    Question(
        id="M1",
        type="multi_hop",
        question="Multi hop?",
        gold_facts=[gold("f3", "zeta eta theta"), gold("f4", "iota kappa")],
    ),
    Question(id="O1", type="out_of_corpus", question="Unknowable?", expected_abstain=True),
]


class Retriever:
    def query(self, text, top_k=4):
        texts = ["alpha beta gamma delta", "unrelated text here", "zeta eta theta words", "more"]
        role = ["core", "core", "core", "distractor"]
        return [
            RetrievedChunk(i + 1, 0.8 - 0.1 * i, Chunk(f"d:p1:c{i}", "d", "T", role[i], 1, t))
            for i, t in enumerate(texts)
        ]


def judge_responder(messages):
    text = messages[-1]["content"]
    if "Decompose the answer" in text:
        return json.dumps({"claims": ["c1", "c2"]})
    if "checking claims" in text:
        labels = ["supported", "unsupported"]
        return json.dumps(
            {
                "verdicts": [
                    {"claim_id": i + 1, "label": labels[i], "evidence_passage": 1, "reason": "why"}
                    for i in range(2)
                ]
            }
        )
    n = len(re.findall(r"^\d+\. ", text.split("Gold facts:")[-1], re.M))
    return json.dumps({"coverage": [{"fact_id": i + 1, "covered": i == 0} for i in range(n)]})


@pytest.fixture
def results(tmp_path):
    def build(gen=None):
        run_benchmark(
            QUESTIONS,
            Retriever(),
            gen or FakeLLMClient(),
            Judge(FakeLLMClient(judge_responder, model="fake-judge")),
            PRICING,
            SETTINGS,
            tmp_path,
            trials=2,
            judge_trials=1,
            log=lambda _: None,
        )
        return tmp_path

    return build


def summary_for(path, **kw):
    return summarize(QUESTIONS, path, HashEmbedder(), PRICING, n_resamples=200, seed=1, **kw)


def test_summary_has_every_strategy_and_metric_with_intervals(results):
    s = summary_for(results())
    assert list(s["strategies"]) == ["baseline", "structured", "rrr", "self_rag", "no_retrieval"]
    for row in s["strategies"].values():
        for m in ("G", "C", "FCS", "words", "latency_s", "consistency_semantic"):
            assert row[m] is not None and row[m]["lo"] <= row[m]["mean"] <= row[m]["hi"]


def test_quality_averages_only_answerable_questions(results):
    s = summary_for(results())
    assert s["strategies"]["baseline"]["FCS"]["n"] == 3  # L1, L2, M1; O1 is excluded
    assert set(s["by_type"]) == {"lookup", "multi_hop"}
    assert s["by_type"]["lookup"]["baseline"]["FCS"]["n"] == 2


def test_known_metric_values_from_the_scripted_judge(results):
    s = summary_for(results())
    # judge: 2 claims, 1 supported -> G = 0.5; coverage: first fact only
    assert s["strategies"]["baseline"]["G"]["mean"] == pytest.approx(0.5)
    pq = s["per_question"]
    assert pq["L1"]["strategies"]["baseline"]["C"] == pytest.approx(1.0)
    assert pq["M1"]["strategies"]["baseline"]["C"] == pytest.approx(0.5)


def test_paired_tables_exclude_the_reference_itself(results):
    s = summary_for(results())
    assert "baseline" not in s["paired"]["vs_baseline"]
    assert "no_retrieval" not in s["paired"]["vs_control"]
    assert s["paired"]["vs_control"]["rrr"]["FCS"]["diff"]["n"] == 3


def test_abstention_is_counted_separately(results):
    class Refuser(FakeLLMClient):
        def __init__(self):
            super().__init__(lambda m: REFUSAL)

    s = summary_for(results(Refuser()))
    a = s["abstention"]["baseline"]
    assert a["out_of_corpus_abstain_rate"] == 1.0 and a["out_of_corpus_n"] == 2
    assert a["false_abstain_rate_answerable"] == 1.0


def test_a_refusal_on_an_answerable_question_scores_fcs_zero_and_undefined_g(results):
    class Refuser(FakeLLMClient):
        def __init__(self):
            super().__init__(lambda m: REFUSAL)

    s = summary_for(results(Refuser()))
    row = s["strategies"]["baseline"]
    assert row["FCS"]["mean"] == 0.0 and row["C"]["mean"] == 0.0 and row["G"] is None


def test_out_of_corpus_question_never_enters_quality_means(results):
    s = summary_for(results())
    assert s["per_question"]["O1"]["type"] == "out_of_corpus"
    assert s["control_out_of_corpus"] and s["control_out_of_corpus"][0]["qid"] == "O1"


def test_retrieval_diagnostics_and_context_recall(results):
    s = summary_for(results())
    r = s["retrieval"]
    assert r["distractor_chunks_in_top4"] == len(QUESTIONS)  # one distractor chunk per question
    assert r["mean_similarity_by_rank"]["1"] == pytest.approx(0.8)
    assert r["context_recall_by_question"]["L1"] == 1.0
    # L2: only "delta" of "delta epsilon" is in a chunk (50% < the 80% threshold), so not recalled
    assert r["context_recall_by_question"]["L2"] == 0.0
    assert r["context_recall_by_question"]["O1"] is None


def test_totals_count_calls_tokens_and_judged_answers(results):
    s = summary_for(results())
    t = s["totals"]
    assert t["answers"] == 4 * 5 * 2 and t["judged_answers"] == 4 * 5
    assert t["generator_tokens"] > 0 and t["judge_tokens"] > 0


def test_fact_in_chunks_uses_a_word_overlap_threshold():
    chunks = ["the quick brown fox jumps", "other words entirely"]
    assert fact_in_chunks("quick brown fox", chunks)
    assert not fact_in_chunks("quick purple elephant", chunks)  # only 1/3 of the words
    assert not fact_in_chunks("", chunks)


def test_context_recall_counts_gold_facts_present():
    q = QUESTIONS[2]  # f3 present, f4 absent
    assert context_recall(q, ["zeta eta theta words", "nothing"]) == 0.5
    assert context_recall(QUESTIONS[3], ["anything"]) is None


def test_markdown_states_the_reduction_and_the_consistency_limitation(results):
    path = results()
    s = summary_for(path)
    meta = {
        "n_questions": 4,
        "n_answerable": 3,
        "composition": "2 lookup, 1 multi-hop, 1 out-of-corpus",
        "trials": 2,
        "judge_trials": 1,
        "generator": "gen",
        "judge": "jdg",
        "log": None,
    }
    md = render_markdown(s, meta)
    assert "only trial 0 is judged" in md
    assert "Consistency rests on 2 samples per question" in md
    assert "Quality (answerable questions" in md and "Abstention" in md


def test_outputs_and_figures_are_written(results, tmp_path):
    path = results()
    s = summary_for(path)
    meta = {
        "n_questions": 4,
        "n_answerable": 3,
        "composition": "x",
        "trials": 2,
        "judge_trials": 1,
        "generator": "g",
        "judge": "j",
        "log": None,
    }
    write_outputs(s, meta, path)
    assert json.loads((path / "summary.json").read_text(encoding="utf-8"))["meta"]["trials"] == 2
    assert (path / "summary.md").read_text(encoding="utf-8").startswith("# Benchmark results")
    figs = make_figures(s, meta, path / "figures")
    assert len(figs) == 7 and all(p.exists() and p.stat().st_size > 1000 for p in figs)


def test_claim_sample_leaves_human_labels_blank_and_is_stratified(results, tmp_path):
    path = results()
    out = path / "claims.csv"
    n = sample_claims_for_labeling(path, out, n=10, seed=3)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert n == len(rows) <= 10 and rows
    assert all(r["human_label"] == "" for r in rows)
    assert {"supported", "unsupported"} <= {r["judge_label"] for r in rows}
    assert all("[1]" in r["passages"] for r in rows)


# ---- judge validation ------------------------------------------------------------------------


def row(judge, human, i=1):
    return {"sample_id": str(i), "judge_label": judge, "human_label": human}


def test_perfect_agreement():
    out = agreement([row("supported", "supported"), row("unsupported", "unsupported", 2)])
    assert out["accuracy_3way"] == 1.0 and out["kappa_3way"] == pytest.approx(1.0)
    assert out["disagreements"] == []


def test_disagreements_are_listed_and_binary_accuracy_differs():
    rows = [
        row("supported", "supported", 1),
        row("unsupported", "contradicted", 2),
        row("supported", "unsupported", 3),
        row("unsupported", "unsupported", 4),
    ]
    out = agreement(rows)
    assert out["accuracy_3way"] == 0.5 and out["accuracy_supported_vs_not"] == 0.75
    assert [d["sample_id"] for d in out["disagreements"]] == ["2", "3"]


def test_invalid_human_labels_are_rejected_and_empty_input_is_safe():
    with pytest.raises(ValueError, match="human_label"):
        agreement([row("supported", "maybe")])
    assert agreement([]) == {"n": 0}


def test_only_filled_rows_are_loaded(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("sample_id,judge_label,human_label\n1,supported,supported\n2,supported,\n")
    assert [r["sample_id"] for r in load_labelled(p)] == ["1"]


# ---- Sensitivity analysis (excluding questions) -----------------------------------------------


def test_summarize_on_a_question_subset_ignores_the_excluded_runs(results):
    path = results()
    subset = [q for q in QUESTIONS if q.id != "L2"]
    full = summary_for(path)
    part = summarize(subset, path, HashEmbedder(), PRICING, n_resamples=200, seed=1)
    assert full["strategies"]["baseline"]["FCS"]["n"] == 3
    assert part["strategies"]["baseline"]["FCS"]["n"] == 2
    assert "L2" not in part["per_question"] and part["retrieval"]["questions"] == 3


def test_markdown_includes_the_sensitivity_section_with_both_versions_and_the_note(results):
    path = results()
    full = summary_for(path)
    part = summarize(
        [q for q in QUESTIONS if q.id != "L2"],
        path,
        HashEmbedder(),
        PRICING,
        n_resamples=200,
        seed=1,
    )
    meta = {
        "n_questions": 4,
        "n_answerable": 3,
        "composition": "x",
        "trials": 2,
        "judge_trials": 1,
        "generator": "g",
        "judge": "j",
        "log": None,
    }
    md = render_markdown(
        full, meta, {"excluded": ["L2"], "summary": part, "note": "NOTE-TEXT-MARKER"}
    )
    assert "## Sensitivity: excluding L2" in md
    assert "G (all) | G (excl.)" in md and "dFCS (all) | dFCS (excl.)" in md
    assert "NOTE-TEXT-MARKER" in md and "## Run totals" in md
    assert md.index("## Sensitivity") < md.index("## Run totals")


def test_markdown_without_sensitivity_has_no_such_section(results):
    path = results()
    meta = {
        "n_questions": 4,
        "n_answerable": 3,
        "composition": "x",
        "trials": 2,
        "judge_trials": 1,
        "generator": "g",
        "judge": "j",
        "log": None,
    }
    assert "Sensitivity" not in render_markdown(summary_for(path), meta)


def test_write_outputs_stores_the_sensitivity_summary_in_json(results):
    path = results()
    full = summary_for(path)
    part = summarize(
        [q for q in QUESTIONS if q.id != "L2"],
        path,
        HashEmbedder(),
        PRICING,
        n_resamples=200,
        seed=1,
    )
    meta = {
        "n_questions": 4,
        "n_answerable": 3,
        "composition": "x",
        "trials": 2,
        "judge_trials": 1,
        "generator": "g",
        "judge": "j",
        "log": None,
    }
    write_outputs(full, meta, path, {"excluded": ["L2"], "summary": part, "note": None})
    data = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    assert data["sensitivity"]["excluded"] == ["L2"]
    assert data["sensitivity"]["strategies"]["baseline"]["FCS"]["n"] == 2


def test_coverage_rows_stay_out_of_the_claim_agreement_and_are_reported_separately(tmp_path):
    import csv

    from ragbench.metrics.validation import coverage_agreement, load_coverage_labelled

    path = tmp_path / "c.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "judge_label", "human_label"])
        w.writerow([1, "supported", "supported"])
        w.writerow([2, "covered", "not covered"])  # a coverage verdict, not a claim verdict
        w.writerow([3, "covered", ""])  # unlabelled: ignored
    claims = load_labelled(path)
    assert [r["sample_id"] for r in claims] == ["1"]
    assert agreement(claims)["accuracy_3way"] == 1.0  # the coverage row cannot raise or skew this
    cov = coverage_agreement(load_coverage_labelled(path))
    assert cov["n"] == 1 and cov["agree"] == 0
    assert cov["cases"] == [{"sample_id": "2", "judge": "covered", "human": "not covered"}]


def test_markdown_report_maps_app_numbers_to_research_ids():
    from ragbench.report import question_numbering

    out = "\n".join(
        question_numbering(
            {"Q02": {"type": "lookup"}, "Q03": {"type": "lookup"}, "Q13": {"type": "out_of_corpus"}}
        )
    )
    assert "| Q1 | Q02 | lookup |" in out and "| Q2 | Q03 | lookup |" in out
    assert "| Q3 | Q13 | out-of-corpus |" in out
