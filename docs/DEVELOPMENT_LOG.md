# Development log

An honest record of what was tried, what failed, and why each decision was made.

## Phase 0 — Prototype (late September 2026)

- **Streamlit shell.** Created `app.py` with four placeholder pages (Ask, Explore, Knowledge Base, Evaluation). Streamlit 1.64.0 installed and ran locally. Metrics on the pages were hard-coded placeholders ("0", "—").
- **FAISS installation failed** in the Windows virtual environment. Rather than fight the install, retrieval was implemented with **L2-normalised embeddings + cosine similarity** (scikit-learn / NumPy). At a few hundred vectors, exact search is sub-millisecond, so FAISS was never needed.
- **Scraping attempt.** `src/collect_services.py` requested `https://my.gov.sa/en/services` with Python `requests` and received **HTTP 403 Forbidden**. Decision: **do not bypass** (no proxy rotation, header spoofing or CAPTCHA solving). The script also had a bug — it would have overwritten `data/services.csv` with an empty file. It is kept in `docs/archive/` for the record and is no longer used.
- **Seed corpus.** 18 records were written by hand (`src/build_knowledge_base.py`) as a smoke-test corpus: titles and `my.gov.sa/en/services/<id>` URLs, with **paraphrased** descriptions.
- **First semantic retrieval test.** With `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, the query *"How can I get my foreign university degree recognized in Saudi Arabia?"* returned *"Equivalency of Qualifications Issued Outside the Kingdom of Saudi Arabia"* with cosine similarity **0.675** — confirming the embedding pipeline worked. Arabic was hard to test because of PowerShell's console encoding.

## Phase 1 — Audit (30 Sep 2026)

Findings: retrieval core worked but loaded the model at import and re-embedded on every run; the app was not connected to retrieval; no tests, README, requirements or `.gitignore`; seed descriptions were summaries with no capture trail; the scraper would overwrite the seed data.

The prototype's retrieval test was reproduced exactly (0.675) with the same model weights, and the Arabic version of the question (*"كيف أعادل شهادتي الجامعية من خارج السعودية؟"*) scored 0.634 against the same service — the first proper UTF-8 Arabic test.

## Phase 2 — Finding legitimate data

- `my.gov.sa` also returned 403 to a hosted fetch tool, and a Cloudflare "you have been blocked" page to an automated browser session. **Stopped contacting my.gov.sa entirely.**
- `open.data.gov.sa` (national open-data portal) could not be reached from the development environment. An adapter for official open-data files was built anyway (`src/ingestion/official_open_data.py`) so a dataset can be added later without new code.
- **Ministry of Commerce (`mc.gov.sa`)** publishes a structured e-services catalogue — 75 services, each with a service-detail page in **English and Arabic** listing description, steps, conditions, required documents, beneficiaries, duration, fees and contact channels. These pages are openly served to browsers. They were captured from a normal browser session, one request every ~2 seconds (the server's response time set the real pace), storing each page's `<article>` HTML verbatim with URL, timestamp and HTTP status (`scripts/browser_capture_mc.js`). Progress was kept in the browser's IndexedDB because tabs were closed twice during the first attempts.
- Parsing is done in Python (`src/ingestion/mc_catalog.py`) so it is deterministic and unit-tested.
- Decision: the 18 seed records are **quarantined** (loaded, validated, rejected with a reason, shown on the Knowledge Base page) — they are not traceable to captured official text.

## Phase 3 — Architecture

- Canonical bilingual schema (`src/schema.py`): `_en`/`_ar` fields hold **official** text in that language only; missing fields stay empty.
- Modular loaders (MC catalogue, saved official pages, open-data files, legacy seed CSV) → normalisation → validation (official domain, HTTPS, provenance, language sanity) → duplicate detection → `services.jsonl` + quarantine + build report.
- Section-level chunks per language; embeddings cached in `embeddings.npy` with a content fingerprint; stale caches are detected.
- Three retrieval methods (lexical TF-IDF char n-grams, dense, hybrid) to compare experimentally.
- Evidence-grounded answer composer instead of an LLM (zero cost, no hallucinated facts).

## Phase 4 — Evaluation

- Benchmark written **before** looking at retrieval results (see `data/evaluation/make_benchmark.py` for authorship notes), split into calibration and test halves by service.
- Method, hybrid weight and refusal threshold chosen on calibration only; results reported on test.

## Phase 5 — Results and fixes (30 Sep 2026)

- Capture: 150/150 pages HTTP 200. Parsing produced 75 services; **2 had no description on the official page** (sID 53, 56) and were quarantined. One parser bug was found and fixed (a page without a "Service level agreement" link lost its description; the header now also ends at the first tab heading). Their benchmark questions were moved to the "not indexed" category, since Dalil cannot answer them.
- Page labels (e.g. "Merchant · Commercial Register · Business sector") turned out to appear in inconsistent order, so they are stored verbatim instead of guessing a single "category".
- Index: 730 chunks (365 EN / 365 AR), embedded in ~113 s on CPU, then cached.
- Method comparison: hybrid clearly best on the calibration half (MRR 0.834 vs lexical 0.725, dense 0.673). The first α grid (0.5–0.9) selected its edge value, so the grid was widened to 0.2–0.9 — still selected on calibration only (α=0.4). On the test half α=0.4 scored Top-1 69.6%, MRR 0.780; α=0.5 would have scored higher on test, but choosing by test results would be cheating, so 0.4 is reported.
- Surprising finding: dense-only retrieval was *weaker* than lexical for Arabic questions against Arabic text (47.7% vs 75.0% Top-1); the multilingual model's value is in crossing languages (Arabic → English-only text: 59.1% vs 2.3%).
- Refusal: the threshold calibrated for balanced accuracy (0.424) declined every unanswerable test question but also about a third of answerable ones. Kept as calibrated and documented, rather than hand-tuned.
- A pre-registered integration test (*"كيف أحجز اسم تجاري لمنشأتي؟"* → Trade Name Reservation) failed: the "extend" and "cancel" reservation services share the phrase and outrank it. Recorded as an expected failure (xfail) and a limitation instead of being hidden. Demo example questions were taken from benchmark questions the evaluation answered correctly.
- App verified in a headless browser: English, Arabic (RTL) and refusal flows, mobile width, all five pages without exceptions.

