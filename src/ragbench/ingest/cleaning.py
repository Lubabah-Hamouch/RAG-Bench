"""Text cleaning for extracted PDF pages.

Pipeline (see `clean_pages`): normalise unicode -> drop repeated header/footer/banner lines ->
drop the bibliography span -> de-hyphenate line breaks -> unwrap lines into paragraphs.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from ragbench.ingest.loaders import RawPage
from ragbench.ingest.manifest import ReferenceSpan

_PUA = re.compile(r"[-]")  # private-use glyphs: icon fonts, unmapped ligatures
_INVISIBLE = re.compile(r"[­​‌‍﻿]")
_DASHES = str.maketrans({"‐": "-", "‑": "-"})
_PAGE_NUMBER = re.compile(r"^\d{1,3}$")
_SENTENCE_END = re.compile(r"[.!?:]['\")\]]?$")
_BLOCK_START = re.compile(r"^(?:[-*••]\s|\[\d+\]|\(\d+\)|\d+[.)]\s|#+\s)")
_WORD = re.compile(r"[a-z]+(?:-[a-z]+)*")


def normalize_unicode(text: str) -> str:
    """NFKC (expands ligatures like 'ﬁ'), drops private-use and invisible characters."""
    text = unicodedata.normalize("NFKC", text).translate(_DASHES)
    text = _PUA.sub("", text)
    text = _INVISIBLE.sub("", text)
    return text.replace(" ", " ")


def _norm_line(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip())


def find_repeated_lines(
    pages: list[list[str]], min_pages: int = 3, min_fraction: float = 0.3, max_len: int = 120
) -> set[str]:
    """Normalised short lines that recur on many pages: running headers, cookie banners, nav."""
    if len(pages) < min_pages:
        return set()
    counts: Counter[str] = Counter()
    for lines in pages:
        counts.update({_norm_line(ln) for ln in lines if 0 < len(ln.strip()) <= max_len})
    threshold = max(min_pages, min_fraction * len(pages))
    return {line for line, n in counts.items() if n >= threshold}


def strip_boilerplate(pages: list[list[str]]) -> list[list[str]]:
    repeated = find_repeated_lines(pages)
    out: list[list[str]] = []
    for lines in pages:
        kept = [ln for ln in lines if ln.strip() and _norm_line(ln) not in repeated]
        # bare page numbers only count at the edges of a page, never inside tables
        while kept and _PAGE_NUMBER.match(kept[-1].strip()):
            kept.pop()
        while kept and _PAGE_NUMBER.match(kept[0].strip()):
            kept.pop(0)
        out.append(kept)
    return out


def strip_references(pages: list[list[str]], span: ReferenceSpan | None) -> list[list[str]]:
    """Drop lines from the first `start` match up to (not including) the next `end` match."""
    if span is None:
        return pages
    start = re.compile(span.start)
    end = re.compile(span.end) if span.end else None
    in_refs = False
    out: list[list[str]] = []
    for lines in pages:
        kept: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not in_refs and start.search(stripped):
                in_refs = True
            elif in_refs and end is not None and end.search(stripped):
                in_refs = False
            if not in_refs:
                kept.append(line)
        out.append(kept)
    return out


def build_vocab(pages: list[list[str]]) -> set[str]:
    """Words seen in the document away from line breaks; used to decide hyphen handling."""
    text = "\n".join("\n".join(lines) for lines in pages).lower()
    text = re.sub(r"[a-z]+-\n[a-z]+", " ", text)
    return set(_WORD.findall(text))


def dehyphenate(pages: list[list[str]], vocab: set[str]) -> list[list[str]]:
    """Rejoin words split across lines ('paramet-' + 'ric').

    The hyphen is kept when the document itself uses the hyphenated form or the left part is
    already a compound ('end-to-' + 'end'); otherwise it is typographic and removed.
    """
    flat = [(i, ln) for i, lines in enumerate(pages) for ln in lines]
    result: list[list[str]] = [[] for _ in pages]
    j = 0
    while j < len(flat):
        page_idx, line = flat[j]
        nxt = flat[j + 1][1] if j + 1 < len(flat) else ""
        m_left = re.search(r"([A-Za-z]+(?:-[A-Za-z]+)*)-$", line.rstrip())
        m_right = re.match(r"([a-z]+)([.,;:)\]]*)(.*)$", nxt.lstrip())
        if m_left and m_right:
            left, right, trailing = m_left.group(1), m_right.group(1), m_right.group(2)
            hyphenated = f"{left}-{right}".lower()
            joined = f"{left}{right}".lower()
            keep_hyphen = "-" in left or (hyphenated in vocab and joined not in vocab)
            sep = "-" if keep_hyphen else ""
            merged = line.rstrip()[: -len(left) - 1] + left + sep + right + trailing
            flat[j + 1] = (flat[j + 1][0], m_right.group(3).lstrip())
            result[page_idx].append(merged)
            # drop an emptied continuation line; otherwise the remainder becomes the next line
            if not flat[j + 1][1].strip():
                j += 1
        else:
            result[page_idx].append(line)
        j += 1
    return result


def unwrap_lines(lines: list[str]) -> str:
    """Join hard-wrapped lines into paragraphs, keeping breaks at likely block boundaries."""
    paragraphs: list[str] = []
    current = ""
    for raw in lines:
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        if not current:
            current = line
            continue
        ends_block = bool(_SENTENCE_END.search(current)) and (
            line[0].isupper() or line[0].isdigit() or bool(_BLOCK_START.match(line))
        )
        if ends_block or _BLOCK_START.match(line):
            paragraphs.append(current)
            current = line
        else:
            current = f"{current} {line}"
    if current:
        paragraphs.append(current)
    return "\n".join(paragraphs)


@dataclass(frozen=True)
class CleanPage:
    number: int
    text: str


def clean_pages(pages: list[RawPage], references: ReferenceSpan | None = None) -> list[CleanPage]:
    line_pages = [normalize_unicode(p.text).split("\n") for p in pages]
    line_pages = strip_boilerplate(line_pages)
    line_pages = strip_references(line_pages, references)
    line_pages = dehyphenate(line_pages, build_vocab(line_pages))
    return [
        CleanPage(p.number, unwrap_lines(lines))
        for p, lines in zip(pages, line_pages, strict=True)
        if any(ln.strip() for ln in lines)
    ]


def quality_report(text: str) -> dict[str, float | int]:
    """Cheap extraction-quality signals, used by the corpus audit."""
    words = re.findall(r"\S+", text)
    letters = sum(c.isalpha() for c in text)
    return {
        "chars": len(text),
        "words": len(words),
        "alpha_ratio": round(letters / len(text), 3) if text else 0.0,
        "pua_chars": len(_PUA.findall(text)),
        "replacement_chars": text.count("�"),
        "spaced_letter_runs": len(re.findall(r"(?:\b[A-Za-z] ){5,}[A-Za-z]\b", text)),
        "line_end_hyphens": len(re.findall(r"[A-Za-z]-\n", text)),
    }
