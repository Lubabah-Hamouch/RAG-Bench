"""Sentence-transformer embedder (all-MiniLM-L6-v2), loaded lazily and shared."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbedderLike(Protocol):
    """What the vector store needs. Lets tests substitute a fast fake."""

    name: str

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        self.name = model_name
        self._device = device
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            try:
                # Cached files first: no network round trip, so a slow or absent connection
                # cannot stall the load once the model has been downloaded.
                self._model = SentenceTransformer(
                    self.name, device=self._device, local_files_only=True
                )
            except Exception:  # noqa: BLE001 - not cached yet: download it
                self._model = SentenceTransformer(self.name, device=self._device)
        return self._model

    def warm(self) -> None:
        """Import the library, load the weights and run one encode.

        The import alone can take a minute on a cold start, and it is lazy, so without this the
        first real query pays for it.
        """
        self.encode(["warm up"])

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> NDArray[np.float32]:
        """Return L2-normalised float32 vectors, shape (len(texts), dim)."""
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
