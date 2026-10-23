"""Acceptance checks on the real corpus: every file is accounted for and extracts cleanly."""

from __future__ import annotations

import re

import pytest

from ragbench.config import repo_root
from ragbench.ingest.cleaning import clean_pages
from ragbench.ingest.loaders import extract_pdf, has_text_layer
from ragbench.ingest.manifest import load_manifest

ROOT = repo_root()
MANIFEST = load_manifest()
CORPUS = ROOT / "data" / "corpus"

# Phrases that must survive cleaning (they anchor the benchmark's gold facts).
MUST_CONTAIN = {
    "lewis2020_rag": ["RAG-Token", "RAG-Sequence", "non-parametric", "BART"],
    "asai2023_selfrag": ["ISREL", "ISSUP", "ISUSE", "reflection tokens"],
    "singhvi2023_dspy_assertions": ["Assert", "Suggest", "backtrack"],
    "pinecone_vector_db": ["vector database"],
    "weaviate_docs": ["open-source"],
    "langchain_overview": ["agent"],
}


def test_every_pdf_in_corpus_dir_is_in_manifest():
    listed = {d.file for d in MANIFEST.documents}
    on_disk = {p.name for p in CORPUS.glob("*.pdf")}
    assert on_disk == listed


def test_distractor_is_declared():
    assert [d.id for d in MANIFEST.documents if d.role == "distractor"] == [
        "zhu2023_pointcloud_survey"
    ]


@pytest.mark.parametrize("spec", MANIFEST.documents, ids=lambda s: s.id)
def test_document_extracts_clean_text(spec):
    pages, _ = extract_pdf(CORPUS / spec.file)
    assert has_text_layer(pages)
    text = "\n".join(p.text for p in clean_pages(pages, spec.references))
    assert len(text) > 2000
    assert not re.search(r"[-�]", text), "private-use or replacement chars left"
    assert (
        not re.search(r"(?:\b[A-Za-z] ){6,}", text.replace("\n", " ")) or spec.role == "distractor"
    )
    for phrase in MUST_CONTAIN.get(spec.id, []):
        assert phrase in text, f"{phrase!r} missing from cleaned {spec.id}"


@pytest.mark.parametrize(
    "spec", [d for d in MANIFEST.documents if d.references], ids=lambda s: s.id
)
def test_bibliography_removed(spec):
    pages, _ = extract_pdf(CORPUS / spec.file)
    text = "\n".join(p.text for p in clean_pages(pages, spec.references))
    assert len(re.findall(r"arXiv preprint|URL https?://", text)) <= 2


def test_pinecone_cookie_banner_removed():
    spec = next(d for d in MANIFEST.documents if d.id == "pinecone_vector_db")
    pages, _ = extract_pdf(CORPUS / spec.file)
    text = "\n".join(p.text for p in clean_pages(pages))
    assert "Cookie Policy" not in text
