"""Build (or reuse) the cached vector index.

    python -m src.indexing.build_index          # rebuild knowledge base + index
    python -m src.indexing.build_index --skip-ingest

Outputs in ``data/processed/``:
    chunks.csv        section-level evidence chunks with metadata
    embeddings.npy    float32 [n_chunks, 384], L2-normalised
    index_meta.json   model name, fingerprint of chunks, build time

If ``chunks.csv`` has not changed since the last build (same fingerprint and
model), the existing embeddings are reused instead of recomputed.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import config
from src.indexing.chunking import build_chunks, chunks_fingerprint
from src.ingestion import pipeline


def build(skip_ingest: bool = False, force: bool = False) -> dict:
    if not skip_ingest:
        pipeline.run()
    records = list(pipeline.load_services().values())
    chunks = build_chunks(records)
    fp = chunks_fingerprint(chunks)

    if (not force and config.INDEX_META_JSON.exists() and config.EMBEDDINGS_NPY.exists()):
        meta = json.loads(config.INDEX_META_JSON.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fp and meta.get("model_name") == config.MODEL_NAME:
            chunks.to_csv(config.CHUNKS_CSV, index=False, encoding="utf-8")
            print(f"Index up to date ({meta['n_chunks']} chunks) -- reusing cached embeddings.")
            return meta

    from src.retrieval import embedder

    # Incremental: reuse the vector of any chunk whose embed_text was already embedded with the
    # same model (a parser fix touching 2 services should not re-embed 9,000 chunks).
    reuse: dict[str, np.ndarray] = {}
    if not force and config.INDEX_META_JSON.exists() and config.EMBEDDINGS_NPY.exists() and config.CHUNKS_CSV.exists():
        meta_old = json.loads(config.INDEX_META_JSON.read_text(encoding="utf-8"))
        old = pd.read_csv(config.CHUNKS_CSV, dtype=str, keep_default_na=False)
        old_emb = np.load(config.EMBEDDINGS_NPY)
        if meta_old.get("model_name") == config.MODEL_NAME and len(old) == len(old_emb):
            reuse = dict(zip(old["embed_text"], old_emb))
    texts = chunks["embed_text"].tolist()
    todo = [t for t in dict.fromkeys(texts) if t not in reuse]
    t0 = time.perf_counter()
    if todo:
        new_vecs = embedder.encode(todo, show_progress=True)
        reuse.update(zip(todo, new_vecs))
    emb = np.stack([reuse[t] for t in texts]).astype(np.float32)
    elapsed = time.perf_counter() - t0
    print(f"Embedded {len(todo)} new chunk texts, reused {len(texts) - len(todo)}.")
    chunks.to_csv(config.CHUNKS_CSV, index=False, encoding="utf-8")
    np.save(config.EMBEDDINGS_NPY, emb)
    meta = {
        "model_name": config.MODEL_NAME,
        "dim": int(emb.shape[1]),
        "n_chunks": int(len(chunks)),
        "n_services": int(chunks["service_id"].nunique()),
        "chunks_by_lang": chunks["lang"].value_counts().to_dict(),
        "chunks_by_section": chunks["section"].value_counts().to_dict(),
        "fingerprint": fp,
        "normalized": True,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_seconds": round(elapsed, 2),
        "newly_embedded_chunks": len(todo),
    }
    config.INDEX_META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Index: {len(chunks)} chunks ({len(todo)} newly embedded in {elapsed:.1f}s) -> {config.EMBEDDINGS_NPY}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ingest", action="store_true", help="use existing services.jsonl")
    ap.add_argument("--force", action="store_true", help="recompute embeddings even if cached")
    args = ap.parse_args()
    build(skip_ingest=args.skip_ingest, force=args.force)
