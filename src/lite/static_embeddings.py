"""Static word embeddings distilled from the V2 multilingual model (runtime: numpy only).

Idea (as in Model2Vec / SIF): embed each *word* of the corpus once with the transformer,
store the table, and represent any text as the IDF-weighted mean of its words' vectors.
Words are normalised exactly like the lexical index (Arabic orthography, lower-case).
Out-of-vocabulary words fall back to a light prefix/suffix-stripped form, else are skipped
(the character-n-gram signal still sees them).

Artifacts (written by ``python -m src.lite.build_static``):
    data/processed/static_vocab.json     {"words": [...], "idf": [...], "fingerprint": ..., "doc_side": ...}
    data/processed/static_vectors.npy    float16 [n_words, dim], L2-normalised rows
    data/processed/static_chunks.npy     float16 [n_chunks, dim], L2-normalised (document side)
"""
from __future__ import annotations

import json
import re

import numpy as np

from src import config
from src.utils.arabic import normalize_for_matching

VOCAB_JSON = config.PROCESSED_DIR / "static_vocab.json"
VECTORS_NPY = config.PROCESSED_DIR / "static_vectors.npy"
CHUNKS_NPY = config.PROCESSED_DIR / "static_chunks.npy"

_TOKEN = re.compile(r"[a-z0-9]+|[ء-ي]+")
_AR_PREFIX = re.compile(r"^(?:وال|فال|بال|كال|لل|ال|و|ف|ب|ل|ك)(?=[ء-ي]{3,})")
_AR_SUFFIX = re.compile(r"(?:ات|ين|ون|ها|هم|ه|ي)$")


def words(text: str) -> list[str]:
    return _TOKEN.findall(normalize_for_matching(text))


def light_stem(w: str) -> str:
    if w[0] >= "ء":
        s = _AR_PREFIX.sub("", w)
        s2 = _AR_SUFFIX.sub("", s)
        return s2 if len(s2) >= 3 else s
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


class StaticEncoder:
    def __init__(self, vocab: list[str], idf: np.ndarray, vectors: np.ndarray, chunk_matrix: np.ndarray):
        self.index = {w: i for i, w in enumerate(vocab)}
        self.idf = idf.astype(np.float32)
        self.vectors = vectors                    # float16 to halve memory; upcast per query
        self.chunk_matrix = chunk_matrix.astype(np.float32)
        self.dim = vectors.shape[1]

    @classmethod
    def load_if_present(cls, fingerprint: str | None = None) -> "StaticEncoder | None":
        if not (VOCAB_JSON.exists() and VECTORS_NPY.exists() and CHUNKS_NPY.exists()):
            return None
        meta = json.loads(VOCAB_JSON.read_text(encoding="utf-8"))
        if fingerprint and meta.get("fingerprint") != fingerprint:
            return None                            # stale: built for other chunks
        return cls(meta["words"], np.asarray(meta["idf"], dtype=np.float32), np.load(VECTORS_NPY),
                   np.load(CHUNKS_NPY))

    def _ids(self, text: str) -> list[int]:
        out = []
        for w in words(text):
            i = self.index.get(w)
            if i is None:
                i = self.index.get(light_stem(w))
            if i is not None:
                out.append(i)
        return out

    def encode(self, text: str) -> np.ndarray:
        ids = self._ids(text)
        if not ids:
            return np.zeros(self.dim, dtype=np.float32)
        ids = np.asarray(ids)
        v = (self.vectors[ids].astype(np.float32) * self.idf[ids, None]).sum(axis=0)
        n = np.linalg.norm(v)
        return v / n if n else v
