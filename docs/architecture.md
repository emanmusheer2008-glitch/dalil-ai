# Architecture (V2)

```mermaid
flowchart TD
    A[Official Saudi service pages<br/>10 agencies · EN + AR] --> CAP[Browser capture scripts<br/>sequential, ~2 s, no bypass]
    CAP --> RAW[(data/raw/official/…<br/>raw captures, unchanged)]
    RAW --> B[Loaders<br/>mc_catalog · generic_service_page + supplements · open data · seed CSV]
    B --> C[Normalisation<br/>NFC, zero-width, whitespace]
    C --> D[Validation<br/>gov.sa domain · HTTPS · provenance · language check · substantive content]
    D -->|fail| Q[(quarantine.jsonl + reason)]
    D --> E[De-duplication<br/>canonical URL · agency + title]
    E --> F[(services.jsonl · 799 services)]
    F --> G[Chunking<br/>title chunk + sections × language · 8,188 chunks]
    G --> H[Multilingual embeddings<br/>MiniLM-L12 · incremental, fingerprinted]
    G --> L[Char-n-gram TF-IDF + BM25<br/>cached lexical_index.pkl]
    U[Question AR / EN / mixed] --> X[Lexicon expansion<br/>lexical signals only]
    X --> R[Retriever<br/>0.3·dense + 0.3·char + 0.1·title coverage<br/>max-pool per service]
    H --> R
    L --> R
    R --> T{Calibrated decision}
    T -- "< 0.285" --> N[Decline: not enough verified information]
    T -- "0.285 – 0.391" --> P[Possible match + warning]
    T -- "≥ 0.391" --> S[Answer]
    P --> SY[Synthesizer<br/>verbatim official points · sections · intent order<br/>conflict + language notes · citations]
    S --> SY
    SY --> UI[Streamlit UI · RTL Arabic · mobile]
    BM[Benchmark 250 Qs · dev / val / test] --> EV[evaluate_v2.py<br/>grid on dev · select on dev+val · thresholds on dev+val<br/>report on test · leakage checks · latency]
    EV --> CFG[(retrieval_config_v2.json)]
    CFG --> R
```

`src/engine.py` (`Dalil.ask`) is the single path from question to answer. The app, the integration tests, the latency measurement and the CLI all call it.

## Components

| Module | Responsibility |
|---|---|
| `src/schema.py` | Canonical `ServiceRecord`: `_en`/`_ar` official-text fields, provenance fields |
| `src/ingestion/generic_service_page.py` | One bilingual parser for the V2 sites (SharePoint, Drupal, custom): tabs (ARIA, Bootstrap, in-page anchors), headings, bold/field labels, label/value pairs; drops related-services cards, widgets, placeholders; loader with supplement merging and a language check |
| `src/ingestion/mc_catalog.py` | Ministry of Commerce parser (V1) |
| `src/ingestion/validation.py`, `pipeline.py` | Validation rules, de-duplication, quarantine, build report with per-agency page statistics |
| `src/indexing/` | Chunking with citation metadata; incremental embedding build |
| `src/retrieval/` | Lazy model, char-n-gram TF-IDF, BM25, bilingual lexicon, hybrid `Retriever` with title coverage and a cached lexical index |
| `src/answering/synthesizer.py` | Three-way decision; sectioned answer from verbatim official text; intent ordering; fee-conflict and language notes |
| `src/engine.py` | `Dalil` engine + CLI |
| `src/evaluation/evaluate_v2.py` | Grid search, selection, thresholds, decision metrics, per-language/style breakdowns, paraphrase robustness, leakage checks, latency |
| `app.py`, `src/ui.py` | Streamlit pages, bilingual strings, escaped HTML cards |

## Key decisions

- **No FAISS:** exact NumPy search over 8k vectors takes milliseconds; FAISS failed to install on Windows and adds nothing at this scale.
- **No generative model:** zero cost, no hallucinated facts, reproducible. The synthesizer only arranges official text.
- **Hybrid retrieval:** dense similarity bridges languages and paraphrases; character n-grams catch exact service names, Arabic morphology and misspellings. Each alone is worse on test (dense 71.7 %, char 68.3 % Top-1 vs 81.7 % hybrid).
- **A middle "possible match" state** instead of one cliff threshold. This was the root cause of the V1 trade-name refusal.
- **Calibrate on dev+val, report on test**, with leakage checks.
- **Caching** of the model, embeddings and lexical index: no per-query reloading or refitting.
- **Streamlit:** Python end to end, free hosting. A separate API/microservice layer would add moving parts without improving anything measured.
