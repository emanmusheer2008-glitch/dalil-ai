"""Build the static embedding table for the V3 lite runtime (OFFLINE; needs the V2 model/torch).

    python -m src.lite.build_static [--doc-side static|model]

Vocabulary = every distinct normalised word in the indexed chunks (both languages) plus the
words of the bilingual lexicon, and their light-stemmed forms. Benchmark questions are NOT
used (no evaluation leakage). Each word is embedded alone with the V2 model; IDF weights come
from the chunks.

``--doc-side`` chooses the document vectors used at query time:
    static : chunks re-encoded with the static table (query and documents in the same space)
    model  : the V2 transformer chunk embeddings (``embeddings.npy``), static queries only
The choice is made by the evaluation, not here; both can be built and compared.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter

import numpy as np
import pandas as pd

from src import config
from src.indexing.chunking import chunks_fingerprint
from src.lite.static_embeddings import CHUNKS_NPY, VECTORS_NPY, VOCAB_JSON, StaticEncoder, light_stem, words
from src.retrieval import lexicon


def build(doc_side: str = "static", remove_pc: int = 0) -> dict:
    chunks = pd.read_csv(config.CHUNKS_CSV, encoding="utf-8", dtype=str, keep_default_na=False)
    fp = chunks_fingerprint(chunks)
    docs = [set(words(t)) for t in chunks["embed_text"]]
    df = Counter(w for d in docs for w in d)
    lex_words = set()
    for pats, add in lexicon._EN + lexicon._AR:
        for t in list(pats) + [add]:
            lex_words.update(words(t))
    vocab = set(df) | lex_words
    vocab |= {light_stem(w) for w in list(vocab)}
    vocab = sorted(w for w in vocab if len(w) >= 2)
    n = len(docs)
    idf = np.array([math.log((n + 1) / (df.get(w, 0) + 1)) + 1.0 for w in vocab], dtype=np.float32)

    from src.retrieval import embedder               # torch only here, offline
    vecs = embedder.encode(vocab, show_progress=True)
    if remove_pc:
        # SIF-style: remove the common direction shared by all isolated-word embeddings, renormalise
        v = vecs - vecs.mean(axis=0, keepdims=True)
        u = np.linalg.svd(v[: min(len(v), 20000)], full_matrices=False)[2][:remove_pc]
        v = v - (v @ u.T) @ u
        vecs = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
    vecs = vecs.astype(np.float16)

    enc = StaticEncoder(vocab, idf, vecs, np.zeros((1, vecs.shape[1]), np.float32))
    if doc_side == "model":
        chunk_m = np.load(config.EMBEDDINGS_NPY).astype(np.float16)
    else:
        chunk_m = np.stack([enc.encode(t) for t in chunks["embed_text"]]).astype(np.float16)
    np.save(VECTORS_NPY, vecs)
    np.save(CHUNKS_NPY, chunk_m)
    VOCAB_JSON.write_text(json.dumps({"words": vocab, "idf": [round(float(x), 5) for x in idf], "fingerprint": fp,
                                      "doc_side": doc_side, "remove_pc": remove_pc, "model": config.MODEL_NAME}, ensure_ascii=False),
                          encoding="utf-8")
    return {"words": len(vocab), "dim": int(vecs.shape[1]), "doc_side": doc_side,
            "vectors_mb": round(vecs.nbytes / 1e6, 1), "chunks_mb": round(chunk_m.nbytes / 1e6, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-side", choices=["static", "model"], default="static")
    ap.add_argument("--remove-pc", type=int, default=0, help="remove this many common components (SIF)")
    a = ap.parse_args()
    print(build(a.doc_side, a.remove_pc))
