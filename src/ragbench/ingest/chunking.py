"""Split cleaned documents into chunks of at most `chunk_chars` characters (default 600)."""

from __future__ import annotations

import bisect
import json
import statistics
from collections.abc import Sequence
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragbench.ingest.corpus import CleanedDocument
from ragbench.types import Chunk

# Sentence-level separators: packing whole sentences fills chunks to ~85-90% of the cap, whereas
# packing paragraphs left them only ~70% full.
SEPARATORS = [". ", "? ", "! ", "; ", ", ", " ", ""]
MIN_CHUNK_CHARS = 60  # drops stray fragments (captions, table debris) that retrieve as noise


def make_splitter(chunk_chars: int, overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_chars,
        chunk_overlap=overlap,
        separators=SEPARATORS,
        keep_separator="end",  # the period stays with the sentence it ends
        length_function=len,
        add_start_index=True,
    )


def chunk_document(
    doc: CleanedDocument,
    chunk_chars: int = 600,
    overlap: int = 50,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[Chunk]:
    """Chunk the whole document, then tag each chunk with the page it *starts* on.

    Chunking across page breaks avoids a short tail chunk on every page. A chunk that straddles a
    page boundary is attributed to its first page.
    """
    flat_pages = [(p.number, p.text.replace("\n", " ")) for p in doc.pages]
    starts: list[int] = []
    parts: list[str] = []
    offset = 0
    for _, text in flat_pages:
        starts.append(offset)
        parts.append(text)
        offset += len(text) + 1  # one joining space
    full_text = " ".join(parts)

    splitter = make_splitter(chunk_chars, overlap)
    per_page: dict[int, int] = {}
    chunks: list[Chunk] = []
    for piece in splitter.create_documents([full_text]):
        text = piece.page_content.strip()
        if len(text) < min_chars:
            continue
        page = flat_pages[bisect.bisect_right(starts, piece.metadata["start_index"]) - 1][0]
        idx = per_page.get(page, 0)
        per_page[page] = idx + 1
        chunks.append(
            Chunk(
                chunk_id=f"{doc.spec.id}:p{page}:c{idx}",
                doc_id=doc.spec.id,
                doc_title=doc.spec.title,
                role=doc.spec.role,
                page=page,
                text=text,
            )
        )
    return chunks


def chunk_corpus(
    docs: Sequence[CleanedDocument], chunk_chars: int = 600, overlap: int = 50
) -> list[Chunk]:
    return [c for doc in docs for c in chunk_document(doc, chunk_chars, overlap)]


def length_stats(chunks: Sequence[Chunk]) -> dict[str, float]:
    lengths = sorted(len(c.text) for c in chunks)
    if not lengths:
        return {}
    q = statistics.quantiles(lengths, n=10) if len(lengths) >= 2 else [lengths[0]] * 9
    return {
        "n": len(lengths),
        "min": lengths[0],
        "p10": q[0],
        "median": statistics.median(lengths),
        "mean": round(statistics.fmean(lengths), 1),
        "p90": q[-1],
        "max": lengths[-1],
    }


def write_chunks_jsonl(chunks: Sequence[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
