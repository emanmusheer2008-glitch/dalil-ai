"""Service retrieval over the cached chunk index.

Three scoring methods are implemented so they can be compared on the
benchmark (see ``src/evaluation``):

* ``lexical`` -- TF-IDF character n-grams (keyword baseline)
* ``dense``   -- multilingual sentence embeddings, cosine similarity
* ``hybrid``  -- ``alpha * dense + (1 - alpha) * lexical``

Chunk scores are aggregated to *services* by taking each service's best chunk
(max-pooling). This also removes duplicates: a service appears once in the
results however many of its chunks match.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config
from src.indexing.chunking import chunks_fingerprint
from src.retrieval import embedder
from src.retrieval.lexical import LexicalIndex

METHODS = ("lexical", "dense", "hybrid")


@dataclass
class Evidence:
    chunk_id: str
    section: str
    lang: str
    text: str
    score: float


@dataclass
class ServiceHit:
    service_id: str
    score: float
    dense: float
    lexical: float
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class SearchResult:
    query: str
    method: str
    hits: list[ServiceHit]
    latency_ms: float


class StaleIndexError(RuntimeError):
    pass


def load_retrieval_config() -> dict:
    """Calibrated settings written by the evaluation (alpha, threshold...)."""
    if config.RETRIEVAL_CONFIG_JSON.exists():
        return json.loads(config.RETRIEVAL_CONFIG_JSON.read_text(encoding="utf-8"))
    return {"method": "dense", "alpha": 1.0, "threshold": None, "calibrated": False}


class Retriever:
    def __init__(self, chunks: pd.DataFrame, embeddings: np.ndarray, model_path: str = config.MODEL_PATH):
        if len(chunks) != len(embeddings):
            raise StaleIndexError("chunks and embeddings have different lengths")
        self.chunks = chunks.reset_index(drop=True)
        self.embeddings = embeddings
        self.model_path = model_path
        self.lexical = LexicalIndex(self.chunks["embed_text"].tolist())
        # service_id -> row indices, for fast max-pooling
        self._service_codes, self._service_ids = pd.factorize(self.chunks["service_id"])

    # ------------------------------------------------------------------ load
    @classmethod
    def from_disk(cls, model_path: str = config.MODEL_PATH) -> "Retriever":
        chunks = pd.read_csv(config.CHUNKS_CSV, encoding="utf-8", dtype=str, keep_default_na=False)
        embeddings = np.load(config.EMBEDDINGS_NPY)
        meta = json.loads(config.INDEX_META_JSON.read_text(encoding="utf-8"))
        if meta["fingerprint"] != chunks_fingerprint(chunks):
            raise StaleIndexError(
                "Embedding cache does not match chunks.csv -- run `python -m src.indexing.build_index`."
            )
        if meta["model_name"] != config.MODEL_NAME:
            raise StaleIndexError("Index was built with a different model.")
        return cls(chunks, embeddings, model_path)

    # --------------------------------------------------------------- scoring
    def chunk_scores(self, query: str, langs: tuple[str, ...] | None = None) -> tuple[np.ndarray, np.ndarray]:
        q = embedder.encode([query], self.model_path)[0]
        dense = self.embeddings @ q
        lex = self.lexical.scores(query)
        if langs:
            mask = ~self.chunks["lang"].isin(langs).to_numpy()
            dense = dense.copy()
            lex = lex.copy()
            dense[mask] = -np.inf
            lex[mask] = -np.inf
        return dense, lex

    def search(
        self,
        query: str,
        method: str = "hybrid",
        alpha: float = 0.7,
        top_k: int = 5,
        langs: tuple[str, ...] | None = None,
        evidence_per_service: int = 3,
    ) -> SearchResult:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        t0 = time.perf_counter()
        dense, lex = self.chunk_scores(query, langs)
        if method == "dense":
            combined = dense
        elif method == "lexical":
            combined = lex
        else:
            combined = alpha * dense + (1 - alpha) * lex

        n_services = len(self._service_ids)
        best = np.full(n_services, -np.inf, dtype=np.float32)
        np.maximum.at(best, self._service_codes, combined)
        order = np.argsort(-best)[:top_k]

        hits = []
        for s in order:
            if not np.isfinite(best[s]):
                continue
            rows = np.flatnonzero(self._service_codes == s)
            rows = rows[np.argsort(-combined[rows])]
            ev = [
                Evidence(
                    chunk_id=self.chunks.at[r, "chunk_id"],
                    section=self.chunks.at[r, "section"],
                    lang=self.chunks.at[r, "lang"],
                    text=self.chunks.at[r, "text"],
                    score=float(combined[r]),
                )
                for r in rows[:evidence_per_service]
                if np.isfinite(combined[r])
            ]
            top_row = rows[0]
            hits.append(
                ServiceHit(
                    service_id=self._service_ids[s],
                    score=float(best[s]),
                    dense=float(dense[top_row]),
                    lexical=float(lex[top_row]),
                    evidence=ev,
                )
            )
        return SearchResult(query, method, hits, (time.perf_counter() - t0) * 1000)
