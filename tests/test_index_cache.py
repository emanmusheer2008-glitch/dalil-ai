"""The vector index must be cached and reused, and a stale cache detected."""
import json

import pytest

from src import config
from src.indexing import build_index
from src.retrieval.retriever import Retriever, StaleIndexError


@pytest.fixture
def tmp_processed(tmp_path, monkeypatch, records):
    for name, fname in [("SERVICES_JSONL", "services.jsonl"), ("CHUNKS_CSV", "chunks.csv"),
                        ("EMBEDDINGS_NPY", "embeddings.npy"), ("INDEX_META_JSON", "index_meta.json")]:
        monkeypatch.setattr(config, name, tmp_path / fname)
    with open(tmp_path / "services.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    return tmp_path


def test_embeddings_are_cached_and_reused(tmp_processed, fake_embedder):
    meta1 = build_index.build(skip_ingest=True)
    assert fake_embedder["n"] == 1
    meta2 = build_index.build(skip_ingest=True)
    assert fake_embedder["n"] == 1                      # no re-embedding
    assert meta1["fingerprint"] == meta2["fingerprint"]
    r = Retriever.from_disk()
    assert len(r.chunks) == meta1["n_chunks"]


def test_force_rebuild(tmp_processed, fake_embedder):
    build_index.build(skip_ingest=True)
    build_index.build(skip_ingest=True, force=True)
    assert fake_embedder["n"] == 2


def test_stale_cache_detected(tmp_processed, fake_embedder):
    build_index.build(skip_ingest=True)
    text = config.CHUNKS_CSV.read_text(encoding="utf-8").replace("Trade Name Reservation", "Edited title")
    config.CHUNKS_CSV.write_text(text, encoding="utf-8")
    with pytest.raises(StaleIndexError):
        Retriever.from_disk()


def test_deterministic_loading(tmp_processed, fake_embedder):
    build_index.build(skip_ingest=True)
    a, b = Retriever.from_disk(), Retriever.from_disk()
    assert a.chunks.equals(b.chunks)
    assert (a.embeddings == b.embeddings).all()
