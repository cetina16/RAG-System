"""Embedding model wrapper using sentence-transformers."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_model_cache: dict[str, "Embedder"] = {}


class Embedder:
    """Thin wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        logger.info("loading_embedding_model", model=model_name)
        self._model = SentenceTransformer(model_name)
        self.dimension: int = self._model.get_sentence_embedding_dimension()
        logger.info("embedding_model_ready", model=model_name, dim=self.dimension)

    def embed(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """
        Encode texts into vectors.

        Args:
            texts: list of strings to encode
            normalize: L2-normalize for cosine similarity via inner product

        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return vectors.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.embed([query])[0]


def get_embedder() -> Embedder:
    """Return a cached Embedder instance."""
    settings = get_settings()
    key = settings.embedding_model
    if key not in _model_cache:
        _model_cache[key] = Embedder(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
        )
    return _model_cache[key]
