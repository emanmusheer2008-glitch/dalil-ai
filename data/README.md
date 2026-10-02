# Data

| Folder | Contents | In git? |
|---|---|---|
| `raw/official/mc/` | V1 Ministry of Commerce capture (EN + AR `<article>` HTML + metadata) | no (from V2 on; it remains in the V1 commit) |
| `raw/official/v2/` | `dalil_v2_<agency>.json` captures (+ `_arfix` supplements): verbatim page HTML, URL, HTTP status, time | **no** (≈200 MB; redistribution terms unclear) |
| `raw/manual/` | Official pages saved by hand, each with a `.meta.json` (empty) | yes |
| `raw/seed/` | The 18 V0 prototype records: **quarantined**, never indexed | yes |
| `processed/` | `services.jsonl/.csv` (knowledge base, 799), `chunks.csv` (8,188), `embeddings.npy`, `index_meta.json`, `build_report.json` (per-agency page stats), `quarantine.jsonl`, `retrieval_config_v2.json` (calibrated), `retrieval_config.json` (V1) | yes: lets the app start without the raw files (see the licence note in `docs/DEPLOYMENT.md`) |
| `processed/lexical_index.pkl` | fitted TF-IDF + BM25 cache | no (rebuilt automatically on first start) |
| `evaluation/` | `make_benchmark_v2.py` (source of truth), `benchmark_v2.csv`, `results_v2.json`, `per_query_results_v2.csv`; V1 files and `v1_baseline/` | yes |

Rebuild from raw captures: `python -m src.indexing.build_index`, then `python data/evaluation/make_benchmark_v2.py` and `python -m src.evaluation.evaluate_v2`.
