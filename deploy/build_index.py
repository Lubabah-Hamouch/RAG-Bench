"""Build the index shipped in the Docker image (deploy/chroma/) from redistributable sources only.

    python deploy/build_index.py

Chroma stores the full text of every chunk, so the deploy index redistributes whatever it holds.
It is built only from manifest entries whose licence is in REDISTRIBUTABLE (today: the CC BY
Self-RAG paper, the one PDF that also ships in data/corpus/), with the same cleaning, chunking and
embedding model as `ragbench ingest`. The local research index (data/chroma/) is untouched.

Live mode on the hosted demo therefore retrieves from fewer documents than the recorded benchmark.
"""

from __future__ import annotations

import shutil
import sqlite3
from collections import Counter
from pathlib import Path

from ragbench.config import load_experiment_settings, repo_root
from ragbench.ingest.chunking import chunk_corpus
from ragbench.ingest.cleaning import clean_pages
from ragbench.ingest.corpus import CleanedDocument
from ragbench.ingest.loaders import extract_pdf
from ragbench.ingest.manifest import load_manifest
from ragbench.retrieval.embedder import Embedder
from ragbench.retrieval.store import VectorStore

REDISTRIBUTABLE = {"CC BY 4.0"}
OUT = repo_root() / "deploy" / "chroma"


def drop_orphan_segments(chroma_dir: Path) -> int:
    """Delete segment folders the SQLite catalogue no longer references (left by rebuilds)."""
    with sqlite3.connect(chroma_dir / "chroma.sqlite3") as db:
        live = {row[0] for row in db.execute("select id from segments")}
    orphans = [p for p in chroma_dir.iterdir() if p.is_dir() and p.name not in live]
    for p in orphans:
        shutil.rmtree(p)
    return len(orphans)


def main() -> None:
    s = load_experiment_settings()
    corpus_dir = repo_root() / s.paths.corpus_dir
    specs = [d for d in load_manifest().documents if d.license in REDISTRIBUTABLE]
    docs = []
    for spec in specs:
        raw_pages, extractor = extract_pdf(corpus_dir / spec.file)
        docs.append(CleanedDocument(spec, extractor, clean_pages(raw_pages, spec.references)))
    chunks = chunk_corpus(docs, s.retrieval.chunk_chars, s.retrieval.chunk_overlap)

    shutil.rmtree(OUT, ignore_errors=True)  # start clean: no stale collection or segments
    n = VectorStore(OUT, Embedder(s.retrieval.embedding_model)).rebuild(chunks)
    removed = drop_orphan_segments(OUT)

    print(f"Indexed {n} chunks into {OUT} ({removed} orphan segment folders removed)")
    for doc_id, count in Counter(c.doc_id for c in chunks).items():
        print(f"  {doc_id:32s} {count:4d}")


if __name__ == "__main__":
    main()
