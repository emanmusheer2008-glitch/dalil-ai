# Data

| Folder | Contents | Versioned? |
|---|---|---|
| `raw/official/mc/` | Captured Ministry of Commerce service pages (EN + AR), verbatim `<article>` HTML + metadata | yes (see redistribution note in docs/data_provenance.md) |
| `raw/manual/` | Official pages saved by hand from a browser, each with a `.meta.json` (empty for now) | yes |
| `raw/seed/` | Original 18-row prototype seed — **quarantined**, never indexed | yes |
| `processed/` | Knowledge base (`services.jsonl/.csv`), `chunks.csv`, cached `embeddings.npy`, `index_meta.json`, `build_report.json`, `quarantine.jsonl`, calibrated `retrieval_config.json` | yes (small; lets the app start without rebuilding) |
| `evaluation/` | `make_benchmark.py` (source of truth), `benchmark.csv`, `results.json`, `per_query_results.csv` | yes |

Rebuild: `python -m src.indexing.build_index` then `python -m src.evaluation.evaluate`.
