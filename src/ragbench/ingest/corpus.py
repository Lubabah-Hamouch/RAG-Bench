"""Load the whole corpus as cleaned, page-tagged documents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ragbench.config import repo_root
from ragbench.ingest.cleaning import CleanPage, clean_pages
from ragbench.ingest.loaders import extract_pdf
from ragbench.ingest.manifest import DocumentSpec, load_manifest


@dataclass(frozen=True)
class CleanedDocument:
    spec: DocumentSpec
    extractor: str
    pages: list[CleanPage]


def load_cleaned_documents(root: Path | None = None) -> list[CleanedDocument]:
    root = root or repo_root()
    corpus_dir = root / "data" / "corpus"
    manifest = load_manifest(corpus_dir / "manifest.yaml")
    docs = []
    for spec in manifest.documents:
        path = corpus_dir / spec.file
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Run `python scripts/fetch_corpus.py` to download the corpus."
            )
        raw_pages, extractor = extract_pdf(path)
        docs.append(CleanedDocument(spec, extractor, clean_pages(raw_pages, spec.references)))
    return docs
