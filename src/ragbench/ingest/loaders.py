"""PDF text extraction. PyMuPDF is primary; pypdf is the fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pymupdf
import pypdf

# pypdf logs a warning per font it cannot fully decode; the audit reports quality instead.
logging.getLogger("pypdf").setLevel(logging.ERROR)

MIN_CHARS_PER_PAGE = 200  # below this on average we treat the PDF as having no usable text layer


@dataclass(frozen=True)
class RawPage:
    number: int  # 1-based
    text: str


def extract_pymupdf(path: Path) -> list[RawPage]:
    with pymupdf.open(path) as doc:  # type: ignore[no-untyped-call]
        return [RawPage(i + 1, page.get_text()) for i, page in enumerate(doc)]


def extract_pypdf(path: Path) -> list[RawPage]:
    reader = pypdf.PdfReader(str(path))
    return [RawPage(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


EXTRACTORS = {"pymupdf": extract_pymupdf, "pypdf": extract_pypdf}


def has_text_layer(pages: list[RawPage]) -> bool:
    if not pages:
        return False
    return sum(len(p.text.strip()) for p in pages) / len(pages) >= MIN_CHARS_PER_PAGE


def extract_pdf(path: Path, extractor: str = "pymupdf") -> tuple[list[RawPage], str]:
    """Extract with the preferred extractor, falling back to the other if it yields no text."""
    order = [extractor] + [name for name in EXTRACTORS if name != extractor]
    last: tuple[list[RawPage], str] = ([], extractor)
    for name in order:
        pages = EXTRACTORS[name](path)
        last = (pages, name)
        if has_text_layer(pages):
            return pages, name
    return last
