"""V3 lite retriever: same scoring code as V2, no transformer at runtime.

It subclasses the V2 ``Retriever`` so ranking, max-pooling, title coverage, lexicon expansion
and evidence selection are *the same code*. Only the "dense" signal changes:

* ``static``  -- a precomputed static embedding table (``data/processed/static_*``): a query
  vector is the weighted mean of its words' vectors, a numpy lookup. Built offline from the V2
  model (``python -m src.lite.build_static``), so no model is loaded in production.
* ``None``    -- no dense signal (pure lexical: char n-grams + BM25 + title coverage).

Which signals and weights are used is chosen by ``src/evaluation/evaluate_v3.py`` on dev+val.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from src import config
from src.indexing.chunking import chunks_fingerprint
from src.retrieval.bm25 import BM25Index
from src.retrieval.lexical import LexicalIndex
from src.retrieval.retriever import Retriever, StaleIndexError
from src.utils.arabic import NORMALIZATION_VERSION

LITE_CACHE_PKL = config.PROCESSED_DIR / "lexical_index.pkl"     # shared with V2 (same fitted objects)


class LiteRetriever(Retriever):
    def __init__(self, chunks: pd.DataFrame, static=None, lexical: LexicalIndex | None = None,
                 bm25: BM25Index | None = None):
        self.static = static
        emb = static.chunk_matrix if static is not None else np.zeros((len(chunks), 1), dtype=np.float32)
        super().__init__(chunks, emb, model_path="(none: lite runtime)", lexical=lexical, bm25=bm25)

    # the only V2 method that touches the transformer
    def _encode_uncached(self, text: str) -> np.ndarray:
        if self.static is None:
            return np.zeros(self.embeddings.shape[1], dtype=np.float32)
        return self.static.encode(text)

    @classmethod
    def from_disk(cls, use_static: bool = True, **_) -> "LiteRetriever":
        chunks = pd.read_csv(config.CHUNKS_CSV, encoding="utf-8", dtype=str, keep_default_na=False)
        meta = json.loads(config.INDEX_META_JSON.read_text(encoding="utf-8"))
        fp = chunks_fingerprint(chunks)
        if meta["fingerprint"] != fp:
            raise StaleIndexError("chunks.csv changed since the index was built -- rebuild the index.")
        lexical = bm25 = None
        if LITE_CACHE_PKL.exists():
            try:
                with open(LITE_CACHE_PKL, "rb") as f:
                    blob = pickle.load(f)
                if blob.get("fingerprint") == fp and blob.get("norm") == NORMALIZATION_VERSION:
                    lexical, bm25 = blob["lexical"], blob["bm25"]
            except Exception:
                lexical = bm25 = None
        static = None
        if use_static:
            from src.lite.static_embeddings import StaticEncoder
            static = StaticEncoder.load_if_present(fp)
        r = cls(chunks, static, lexical, bm25)
        if lexical is None:
            try:
                with open(LITE_CACHE_PKL, "wb") as f:
                    pickle.dump({"fingerprint": fp, "norm": NORMALIZATION_VERSION,
                                 "lexical": r.lexical, "bm25": r.bm25}, f, protocol=pickle.HIGHEST_PROTOCOL)
            except OSError:
                pass
        return r
