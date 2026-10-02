"""Service retrieval over the cached chunk index.

Signals per chunk (all computed at query time against precomputed indexes):

* ``dense`` -- cosine similarity of multilingual sentence embeddings
* ``char``  -- TF-IDF over character 3-5-grams (Arabic-normalised)
* ``bm25``  -- word-level BM25 with light bilingual stemming, squashed to
  [0, 1) with ``s / (s + k)`` so it keeps an absolute meaning

Methods
-------
* V1 (kept for the baseline): ``lexical`` (char), ``dense``, ``hybrid``
  (``alpha*dense + (1-alpha)*char``) on the unexpanded query.
* V2: ``v2`` -- ``w_dense*dense + w_char*char + w_bm25*bm25`` with optional
  lexicon query expansion (see ``lexicon.py``). Weights are chosen on the
  development split by ``src/evaluation/evaluate_v2.py``.

Chunk scores are max-pooled per service (a service appears once), then the
top services carry their best evidence chunks, which the answer composer uses.
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from src import config
from src.indexing.chunking import chunks_fingerprint
from src.retrieval import embedder
from src.retrieval.bm25 import BM25Index, tokenize
from src.retrieval.lexical import LexicalIndex
from src.retrieval.lexicon import expand_query
from src.utils.arabic import NORMALIZATION_VERSION

METHODS = ("lexical", "dense", "hybrid", "v2")

DEFAULT_V2 = {"w_dense": 0.4, "w_char": 0.3, "w_bm25": 0.3, "bm25_k": 6.0, "expand": True, "expand_dense": False,
              "w_tcov": 0.0}


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
    expanded_query: str | None = None


class StaleIndexError(RuntimeError):
    pass


def load_retrieval_config() -> dict:
    """Calibrated settings written by the evaluation."""
    for path in (config.RETRIEVAL_CONFIG_V2_JSON, config.RETRIEVAL_CONFIG_JSON):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {"method": "dense", "alpha": 1.0, "threshold": None, "calibrated": False}


class Retriever:
    def __init__(self, chunks: pd.DataFrame, embeddings: np.ndarray, model_path: str = config.MODEL_PATH,
                 lexical: LexicalIndex | None = None, bm25: BM25Index | None = None):
        if len(chunks) != len(embeddings):
            raise StaleIndexError("chunks and embeddings have different lengths")
        self.chunks = chunks.reset_index(drop=True)
        self.embeddings = embeddings
        self.model_path = model_path
        texts = self.chunks["embed_text"].tolist()
        self.lexical = lexical or LexicalIndex(texts)
        self.bm25 = bm25 or BM25Index(texts)
        self._service_codes, self._service_ids = pd.factorize(self.chunks["service_id"])
        self._lang = self.chunks["lang"].to_numpy()
        self._section = self.chunks["section"].to_numpy()
        self._service_rows = [np.flatnonzero(self._service_codes == s) for s in range(len(self._service_ids))]
        self._encode = lru_cache(maxsize=2048)(self._encode_uncached)
        # title token sets per service (both languages) for the title-coverage feature
        titles = self.chunks[self.chunks["section"] == "title"] if (self.chunks["section"] == "title").any() \
            else self.chunks.drop_duplicates(["service_id", "lang"])
        code_of = {sid: i for i, sid in enumerate(self._service_ids)}
        self._title_tokens: list[list[set]] = [[] for _ in self._service_ids]
        for sid, title in zip(titles["service_id"], titles["title"]):
            toks = set(tokenize(title))
            if toks:
                self._title_tokens[code_of[sid]].append(toks)

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
        # Fitting the char-n-gram TF-IDF takes ~3 s; reuse a fitted copy when it matches these chunks.
        # The pickle is a local cache this project writes itself (never downloaded or shared).
        lexical = bm25 = None
        cache = config.LEXICAL_CACHE_PKL
        if cache.exists():
            try:
                with open(cache, "rb") as f:
                    blob = pickle.load(f)
                if blob.get("fingerprint") == meta["fingerprint"] and blob.get("norm") == NORMALIZATION_VERSION:
                    lexical, bm25 = blob["lexical"], blob["bm25"]
            except Exception:
                lexical = bm25 = None
        r = cls(chunks, embeddings, model_path, lexical, bm25)
        if lexical is None:
            try:
                with open(cache, "wb") as f:
                    pickle.dump({"fingerprint": meta["fingerprint"], "norm": NORMALIZATION_VERSION,
                                 "lexical": r.lexical, "bm25": r.bm25}, f,
                                protocol=pickle.HIGHEST_PROTOCOL)
            except OSError:
                pass  # read-only filesystem: just fit at start-up
        return r

    @property
    def service_ids(self):
        return self._service_ids

    # --------------------------------------------------------------- scoring
    def _encode_uncached(self, text: str) -> np.ndarray:
        return embedder.encode([text], self.model_path)[0]

    def signals(self, query: str, expand: bool = False, expand_dense: bool = False,
                langs: tuple[str, ...] | None = None, bm25_k: float = 6.0) -> dict[str, np.ndarray]:
        """Per-chunk scores for every signal. Cached query embeddings."""
        q_lex = expand_query(query) if expand else query
        q_dense = q_lex if (expand and expand_dense) else query
        dense = self.embeddings @ self._encode(q_dense)
        char = self.lexical.scores(q_lex)
        raw = self.bm25.scores(q_lex)
        bm25 = raw / (raw + bm25_k)
        if langs:
            mask = ~np.isin(self._lang, langs)
            dense, char, bm25 = dense.copy(), char.copy(), bm25.copy()
            dense[mask] = char[mask] = bm25[mask] = -np.inf
        return {"dense": dense, "char": char, "bm25": bm25, "expanded": q_lex, "qtokens": set(tokenize(q_lex))}

    def title_coverage(self, qtokens: set) -> np.ndarray:
        """Per service: best share of its title's words that appear in the query (0..1).

        Penalises sibling services whose titles add words the user never used
        (e.g. "Extension of ... reservation" when the user just asked to reserve).
        """
        cov = np.zeros(len(self._service_ids), dtype=np.float32)
        if not qtokens:
            return cov
        for i, sets in enumerate(self._title_tokens):
            if sets:
                cov[i] = max(len(t & qtokens) / len(t) for t in sets)
        return cov

    # backwards-compatible (V1)
    def chunk_scores(self, query: str, langs: tuple[str, ...] | None = None) -> tuple[np.ndarray, np.ndarray]:
        s = self.signals(query, langs=langs)
        return s["dense"], s["char"]

    @staticmethod
    def combine(sig: dict, method: str, alpha: float = 0.7, params: dict | None = None) -> np.ndarray:
        if method == "dense":
            return sig["dense"]
        if method == "lexical":
            return sig["char"]
        if method == "hybrid":
            return alpha * sig["dense"] + (1 - alpha) * sig["char"]
        p = {**DEFAULT_V2, **(params or {})}
        return p["w_dense"] * sig["dense"] + p["w_char"] * sig["char"] + p["w_bm25"] * sig["bm25"]

    def pool(self, combined: np.ndarray) -> np.ndarray:
        best = np.full(len(self._service_ids), -np.inf, dtype=np.float32)
        np.maximum.at(best, self._service_codes, combined)
        return best

    def search(
        self,
        query: str,
        method: str = "hybrid",
        alpha: float = 0.7,
        top_k: int = 5,
        langs: tuple[str, ...] | None = None,
        evidence_per_service: int = 3,
        params: dict | None = None,
    ) -> SearchResult:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        t0 = time.perf_counter()
        p = {**DEFAULT_V2, **(params or {})}
        v2 = method == "v2"
        sig = self.signals(query, expand=v2 and p["expand"], expand_dense=v2 and p["expand_dense"],
                           langs=langs, bm25_k=p["bm25_k"])
        combined = self.combine(sig, method, alpha, p)
        best = self.pool(combined)
        if v2 and p.get("w_tcov"):
            best = best + p["w_tcov"] * self.title_coverage(sig["qtokens"])
        order = np.argsort(-best)[:top_k]

        hits = []
        for s in order:
            if not np.isfinite(best[s]):
                continue
            rows = self._service_rows[s]
            rows = rows[np.argsort(-combined[rows])]
            ev = [
                Evidence(
                    chunk_id=self.chunks.at[r, "chunk_id"],
                    section=self._section[r],
                    lang=self._lang[r],
                    text=self.chunks.at[r, "text"],
                    score=float(combined[r]),
                )
                for r in rows[:evidence_per_service]
                if np.isfinite(combined[r])
            ]
            top_row = rows[0]
            hits.append(ServiceHit(self._service_ids[s], float(best[s]), float(sig["dense"][top_row]),
                                   float(sig["char"][top_row]), ev))
        return SearchResult(query, method, hits, (time.perf_counter() - t0) * 1000,
                            expanded_query=sig["expanded"] if v2 else None)
