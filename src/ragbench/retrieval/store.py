"""ChromaDB vector store with cosine similarity. One retrieval call feeds every strategy."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from pathlib import Path

import chromadb
from chromadb.config import Settings

from ragbench.retrieval.embedder import EmbedderLike
from ragbench.types import Chunk, RetrievedChunk

COLLECTION = "ragbench_chunks"
ADD_BATCH = 256


class VectorStore:
    def __init__(
        self, persist_dir: Path, embedder: EmbedderLike, collection: str = COLLECTION
    ) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir), settings=Settings(anonymized_telemetry=False)
        )
        self._embedder = embedder
        self._name = collection

    def _collection(self) -> chromadb.Collection:
        return self._client.get_collection(self._name)

    def rebuild(self, chunks: Sequence[Chunk]) -> int:
        """Drop and recreate the collection from `chunks`. Returns the number indexed."""
        # the error raised for a missing collection differs across chroma versions
        with contextlib.suppress(Exception):
            self._client.delete_collection(self._name)
        col = self._client.create_collection(
            self._name,
            metadata={"hnsw:space": "cosine", "embedding_model": self._embedder.name},
        )
        for start in range(0, len(chunks), ADD_BATCH):
            batch = chunks[start : start + ADD_BATCH]
            col.add(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                embeddings=self._embedder.encode([c.text for c in batch]).tolist(),
                metadatas=[
                    {
                        "doc_id": c.doc_id,
                        "doc_title": c.doc_title,
                        "role": c.role,
                        "page": c.page,
                    }
                    for c in batch
                ],
            )
        return col.count()

    def count(self) -> int:
        return self._collection().count()

    def query(self, text: str, top_k: int = 4) -> list[RetrievedChunk]:
        """Top-k chunks by cosine similarity (similarity = 1 - Chroma's cosine distance)."""
        vector = self._embedder.encode([text]).tolist()
        res = self._collection().query(
            query_embeddings=vector,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = res["ids"][0]
        docs = res["documents"][0]  # type: ignore[index]
        metas = res["metadatas"][0]  # type: ignore[index]
        dists = res["distances"][0]  # type: ignore[index]
        out: list[RetrievedChunk] = []
        for rank, (cid, doc, meta, dist) in enumerate(
            zip(ids, docs, metas, dists, strict=True), start=1
        ):
            chunk = Chunk(
                chunk_id=cid,
                doc_id=str(meta["doc_id"]),
                doc_title=str(meta["doc_title"]),
                role=str(meta["role"]),
                page=int(meta["page"]),
                text=doc,
            )
            out.append(
                RetrievedChunk(rank=rank, similarity=round(1.0 - float(dist), 6), chunk=chunk)
            )
        return out
