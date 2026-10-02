"""Thin wrapper around the multilingual sentence encoder.

The model is loaded lazily (only when first needed) and once per process.
All vectors are L2-normalised, so cosine similarity == dot product.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from src import config


@lru_cache(maxsize=2)
def get_model(model_path: str = config.MODEL_PATH):
    from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

    return SentenceTransformer(model_path, device="cpu")


def encode(texts: list[str], model_path: str = config.MODEL_PATH, show_progress: bool = False) -> np.ndarray:
    model = get_model(model_path)
    vecs = model.encode(
        list(texts),
        batch_size=config.EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
    return vecs.astype(np.float32)
