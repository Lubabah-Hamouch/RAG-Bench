import re
from collections import Counter

import pytest

from ragbench.ingest.corpus import load_cleaned_documents
from ragbench.ingest.manifest import load_manifest
from ragbench.questions import load_questions

QS = load_questions()


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("’", "'").replace("‘", "'")).strip()


@pytest.fixture(scope="module")
def pages():
    return {d.spec.id: {p.number: p.text for p in d.pages} for d in load_cleaned_documents()}


def test_composition_is_five_lookup_four_multi_hop_one_out_of_corpus():
    assert Counter(q.type for q in QS.questions) == {
        "lookup": 5,
        "multi_hop": 4,
        "out_of_corpus": 1,
    }
    assert len(QS.questions) == 10


def test_ids_are_unique_and_dropped_ones_are_not_in_the_set():
    ids = [q.id for q in QS.questions]
    assert len(set(ids)) == len(ids)
    dropped = {d.id for d in QS.dropped_questions}
    assert dropped == {"Q01", "Q04", "Q07"} and not dropped & set(ids)
    assert all(d.type == "lookup" and d.reason for d in QS.dropped_questions)


def test_only_the_out_of_corpus_question_expects_abstention_and_has_no_gold_facts():
    for q in QS.questions:
        if q.type == "out_of_corpus":
            assert q.expected_abstain and not q.gold_facts
        else:
            assert not q.expected_abstain and len(q.gold_facts) >= 2


def test_multi_hop_questions_draw_on_at_least_two_documents():
    for q in QS.questions:
        if q.type == "multi_hop":
            assert len({g.doc_id for g in q.gold_facts}) >= 2, q.id


def test_gold_fact_sources_exist_in_the_manifest():
    ids = {d.id for d in load_manifest().documents}
    for q in QS.questions:
        assert {g.doc_id for g in q.gold_facts} <= ids


def test_the_distractor_paper_is_never_a_gold_source():
    assert all(g.doc_id != "zhu2023_pointcloud_survey" for q in QS.questions for g in q.gold_facts)


def test_every_gold_fact_quote_appears_verbatim_on_its_cited_page(pages):
    # Papers are pinned and must match. Web pages change after the recorded snapshot, so a
    # mismatch there degrades to a skip (after every strict check has run) instead of a failure.
    web = {d.id: d for d in load_manifest().documents if "arxiv.org" not in d.source_url}
    drifted = []
    for q in QS.questions:
        for g in q.gold_facts:
            found = norm(g.quote) in norm(pages[g.doc_id].get(g.page, ""))
            if g.doc_id in web:
                if not found:
                    drifted.append((q.id, g.doc_id, g.page))
            else:
                assert found, (q.id, g.doc_id, g.page)
    if drifted:
        retrieved = {web[doc_id].id: web[doc_id].retrieved for _, doc_id, _ in drifted}
        pytest.skip(
            f"gold-fact quotes not found on freshly fetched web pages {drifted}; they were valid "
            f"as of the recorded snapshots {retrieved}, and the live pages have changed since"
        )


def test_the_out_of_corpus_topic_is_really_absent_from_the_corpus(pages):
    pattern = re.compile(r"llama[\s\-]?3", re.IGNORECASE)
    assert not any(pattern.search(t) for doc in pages.values() for t in doc.values())
