from ragbench.ingest.cleaning import (
    build_vocab,
    clean_pages,
    dehyphenate,
    find_repeated_lines,
    normalize_unicode,
    strip_boilerplate,
    strip_references,
    unwrap_lines,
)
from ragbench.ingest.loaders import RawPage
from ragbench.ingest.manifest import ReferenceSpan


def test_normalize_expands_ligatures_and_drops_private_use():
    assert normalize_unicode("ﬁne-tuned Weaviate Academy") == "fine-tuned Weaviate Academy"


def test_normalize_drops_soft_hyphen_and_zero_width():
    assert normalize_unicode("para­metric​") == "parametric"


def test_repeated_lines_detects_running_header_with_changing_page_number():
    pages = [[f"Header {n}", f"distinct body text number {chr(96 + n)}"] for n in range(1, 7)]
    assert "Header #" in find_repeated_lines(pages)
    cleaned = strip_boilerplate(pages)
    assert [ln[0] for ln in cleaned] == [
        f"distinct body text number {chr(96 + n)}" for n in range(1, 7)
    ]


def test_repeated_lines_skipped_for_tiny_documents():
    assert find_repeated_lines([["a"], ["a"]]) == set()


def test_page_numbers_removed_only_at_page_edges():
    pages = [["intro", "42", "more", "7"], ["x", "y"], ["z"]]
    assert strip_boilerplate(pages)[0] == ["intro", "42", "more"]


def test_strip_references_cuts_span_and_keeps_appendix():
    pages = [
        ["Conclusion text", "References", "[1] A. Author. Title."],
        ["[2] B. Author. Title.", "APPENDIX", "A Details"],
    ]
    out = strip_references(pages, ReferenceSpan(start=r"^References$", end=r"^APPENDIX$"))
    assert out == [["Conclusion text"], ["APPENDIX", "A Details"]]


def test_strip_references_without_end_drops_to_document_end():
    out = strip_references([["a", "References", "b"], ["c"]], ReferenceSpan(start="^References$"))
    assert out == [["a"], []]


def test_strip_references_noop_without_span():
    pages = [["a", "References"]]
    assert strip_references(pages, None) == pages


def test_dehyphenate_removes_typographic_hyphen():
    pages = [["the paramet-", "ric memory is large"]]
    assert dehyphenate(pages, build_vocab(pages)) == [["the parametric", "memory is large"]]


def test_dehyphenate_keeps_existing_compound():
    pages = [["an end-to-", "end model"]]
    assert dehyphenate(pages, set()) == [["an end-to-end", "model"]]


def test_dehyphenate_keeps_hyphen_when_document_uses_hyphenated_form():
    pages = [["we use fine-", "tuned models", "fine-tuned models work"]]
    out = dehyphenate(pages, build_vocab(pages))
    assert out[0][0] == "we use fine-tuned"


def test_dehyphenate_keeps_trailing_punctuation():
    pages = [["a retrieval-augmented sys-", "tem, which helps"]]
    assert dehyphenate(pages, set())[0][0] == "a retrieval-augmented system,"


def test_unwrap_joins_wrapped_lines_but_keeps_paragraph_breaks():
    lines = ["Retrieval helps grounding", "of answers in sources.", "Next paragraph begins here."]
    assert unwrap_lines(lines) == (
        "Retrieval helps grounding of answers in sources.\nNext paragraph begins here."
    )


def test_unwrap_breaks_before_list_items_and_numbered_refs():
    out = unwrap_lines(["Steps:", "- first thing", "- second thing", "[3] ref entry"])
    assert out.split("\n") == ["Steps:", "- first thing", "- second thing", "[3] ref entry"]


def test_clean_pages_end_to_end():
    raw = [
        RawPage(
            i,
            f"Running header {i}\nPage {chr(96 + i)} has some ﬁne para-\n"
            f"meter text {chr(96 + i)}.\n{i}",
        )
        for i in range(1, 5)
    ]
    cleaned = clean_pages(raw)
    assert [p.text for p in cleaned] == [
        f"Page {chr(96 + i)} has some fine parameter text {chr(96 + i)}." for i in range(1, 5)
    ]
