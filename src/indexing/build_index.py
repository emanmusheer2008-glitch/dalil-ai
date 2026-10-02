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

    t0 = time.perf_counter()
    emb = embedder.encode(chunks["embed_text"].tolist(), show_progress=True)
    elapsed = time.perf_counter() - t0
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
    }
    config.INDEX_META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Embedded {len(chunks)} chunks in {elapsed:.1f}s -> {config.EMBEDDINGS_NPY}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ingest", action="store_true", help="use existing services.jsonl")
    ap.add_argument("--force", action="store_true", help="recompute embeddings even if cached")
    args = ap.parse_args()
    build(skip_ingest=args.skip_ingest, force=args.force)
