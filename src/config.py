"""Central configuration: paths and model settings.

Everything is relative to the repository root so the project runs the same way
on Windows, Linux and a hosting platform.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "evaluation"

SERVICES_JSONL = PROCESSED_DIR / "services.jsonl"
SERVICES_CSV = PROCESSED_DIR / "services.csv"
QUARANTINE_JSONL = PROCESSED_DIR / "quarantine.jsonl"
CHUNKS_CSV = PROCESSED_DIR / "chunks.csv"
EMBEDDINGS_NPY = PROCESSED_DIR / "embeddings.npy"
INDEX_META_JSON = PROCESSED_DIR / "index_meta.json"
BUILD_REPORT_JSON = PROCESSED_DIR / "build_report.json"

BENCHMARK_CSV = EVAL_DIR / "benchmark.csv"
EVAL_RESULTS_JSON = EVAL_DIR / "results.json"
EVAL_PER_QUERY_CSV = EVAL_DIR / "per_query_results.csv"
RETRIEVAL_CONFIG_JSON = PROCESSED_DIR / "retrieval_config.json"

# Free, open-source multilingual sentence encoder (Apache-2.0).
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Optional: point to a local copy of the model (offline use / restricted
# networks). If unset, sentence-transformers downloads the model from the
# Hugging Face Hub on first use and caches it.
MODEL_PATH = os.environ.get("DALIL_MODEL_PATH") or MODEL_NAME

EMBED_BATCH_SIZE = 32
