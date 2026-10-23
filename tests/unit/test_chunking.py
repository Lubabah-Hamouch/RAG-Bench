from ragbench.ingest.chunking import chunk_corpus, chunk_document, length_stats
from ragbench.ingest.cleaning import CleanPage
from ragbench.ingest.corpus import CleanedDocument
from ragbench.ingest.manifest import DocumentSpec

SENTENCE = "Retrieval grounds the generator in documents that can be checked. "


def make_doc(pages: dict[int, str], doc_id: str = "doc_a", role: str = "core") -> CleanedDocument:
    spec = DocumentSpec(id=doc_id, file=f"{doc_id}.pdf", title="Title A", role=role)
    return CleanedDocument(spec, "pymupdf", [CleanPage(n, t) for n, t in pages.items()])


def test_chunks_respect_size_limit_and_have_metadata():
    doc = make_doc({3: SENTENCE * 40, 4: SENTENCE * 20}, role="distractor")
    chunks = chunk_document(doc, chunk_chars=550, overlap=50)
    assert len(chunks) > 3
    assert all(len(c.text) <= 550 for c in chunks)
    assert {c.page for c in chunks} == {3, 4}
    assert all(c.doc_id == "doc_a" and c.role == "distractor" for c in chunks)


def test_chunk_ids_are_deterministic_and_unique():
    doc = make_doc({1: SENTENCE * 40})
    a, b = chunk_document(doc), chunk_document(doc)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)
    assert a[0].chunk_id == "doc_a:p1:c0"


def test_chunk_is_attributed_to_the_page_it_starts_on():
    doc = make_doc({1: "alpha beta. " * 30, 2: "omega zeta. " * 30})
    chunks = chunk_document(doc, chunk_chars=200, overlap=0)
    assert {c.page for c in chunks} == {1, 2}
    assert all("alpha" in c.text for c in chunks if c.page == 1 and "omega" not in c.text)
    assert all(c.text.startswith("omega") for c in chunks if c.page == 2 and "alpha" not in c.text)


def test_chunks_start_on_sentence_text_not_a_stray_period():
    chunks = chunk_document(make_doc({1: SENTENCE * 40}))
    assert all(c.text[0].isalpha() for c in chunks)


def test_overlap_repeats_text_between_neighbours():
    doc = make_doc({1: " ".join(f"word{i}" for i in range(300))})
    chunks = chunk_document(doc, chunk_chars=200, overlap=40)
    first_tail = set(chunks[0].text.split()[-3:])
    assert first_tail & set(chunks[1].text.split())


def test_documents_shorter_than_min_chunk_yield_nothing():
    assert chunk_document(make_doc({1: "Fig. 3"})) == []


def test_min_chars_filter_applies():
    doc = make_doc({1: SENTENCE * 3})
    assert chunk_document(doc, min_chars=10_000) == []


def test_text_is_not_lost_across_page_breaks():
    doc = make_doc({1: "first page ends mid sentence and", 2: "continues on the second page here."})
    text = " ".join(c.text for c in chunk_document(doc, min_chars=1))
    assert "mid sentence and continues on the second page" in text


def test_chunk_corpus_and_length_stats():
    docs = [make_doc({1: SENTENCE * 30}, "doc_a"), make_doc({1: SENTENCE * 30}, "doc_b")]
    chunks = chunk_corpus(docs, 550, 50)
    assert {c.doc_id for c in chunks} == {"doc_a", "doc_b"}
    stats = length_stats(chunks)
    assert stats["n"] == len(chunks)
    assert stats["max"] <= 550
    assert stats["min"] <= stats["median"] <= stats["max"]
