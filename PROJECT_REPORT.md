# Dalil AI — Final Project Report (V2, 1 October 2026)

## A. Status
**Almost finished.** The code, data pipeline, index, evaluation, tests, app and documentation are complete and verified. Two items remain open: the everyday Interior/Absher topics can't be captured legitimately from the available connection, and deployment needs the owner's approval plus a data-licence decision.

## B. Project location
`<project folder>` (git repository; V1 commit tagged `v1`, V2 committed on top). The `OneDrive\Desktop\dalil-ai` folder was empty and is not used.

## C. Data
| Agency | Raw pages | HTTP 200 | Services indexed | Arabic / English | Exclusions & failures |
|---|---|---|---|---|---|
| ZATCA | 322 | 322 | 161 | 161 / 161 | — |
| Municipalities & Housing | 302 | 302 | 150 | 148 / 150 | 2 duplicates; 2 services without an official Arabic link |
| Justice | 302 | 302 | 148 | 148 / 148 | 3 duplicates, 1 error page |
| HRSD | 252 | 250 | 125 | 125 / 125 | 2 pages 404 |
| Commerce (V1 capture) | 150 | 150 | 73 | 73 / 73 | 2 placeholder-only pages |
| Education | 98 | 97 | 49 | 47 / 49 | 1 fetch failure, 1 wrong-language page |
| Foreign Affairs | 84 | 84 | 42 | 42 / 42 | — |
| SFDA | 66 + 26 re-fetched | 40 + 25 | 25 | 25 / 25 | 1 page 403; 8 pages without real content |
| Council of Health Insurance | 30 | 15 | 15 | 0 / 15 | Arabic pages re-captured but blocked from export |
| Hajj & Umrah | 22 | 22 | 11 | 11 / 11 | — |
| **Total** | | | **799** | **780 / 799** | 33 quarantined (incl. 18 V0 seed records), 5 duplicates |

8,188 evidence chunks (4,126 EN / 4,062 AR). Not available (never bypassed): my.gov.sa (Cloudflare block), Absher (DNS), MOI (error page), MOH (bot check), GOSI (firewall), Transport General Authority, MISA.

## D. Final architecture
**Simple:** question → add official synonyms → search every section of every service by meaning and by spelling → rank services → decide (answer / possible match / related links / decline) → show only official text with sources.
**Technical:** raw captures → source-aware bilingual parser → validation and quarantine → `services.jsonl` → title + section chunks → multilingual MiniLM embeddings (incremental, cached) + char-n-gram TF-IDF (cached) → hybrid scoring with max-pooling per service → calibrated thresholds → grounded synthesizer. A single engine (`src/engine.py`) serves the app, the tests and the CLI. See `docs/architecture.md`.

## E. Retrieval (exact)
`score(service) = max over its chunks of (0.3·cosine(MiniLM) + 0.3·TF-IDF char 3–5-grams)` + `0.1·title coverage`; lexicon expansion applied to the lexical signal only; title chunks on; BM25 weight 0 (tested, not selected). Thresholds: answer ≥ 0.391, possible match ≥ 0.285, related links ≥ 0.200, otherwise decline. All were selected on dev+val.

## F. Known trade-name test
"How can I reserve a trade name?" → **Trade Name Reservation (mc-1) ranked first**, score 0.382, shown as a **possible match** with full official steps, conditions, documents, fees ("200 for Arabic Name, 500 for English Name", verbatim) and the source link. Not refused. "I want to book a business name" → confident answer, mc-1. The Arabic V1 xfail «كيف أحجز اسم تجاري لمنشأتي؟» → answered, mc-1.

## G. Language quality (test)
English Top-1 79.4 % / Top-3 90.5 % (n=63) · Arabic 86.8 % / 98.1 % (n=53) · mixed 50 % / 75 % (n=4, too small to conclude). Colloquial 75 %, conversational 73 %, formal 87 %, short 94 % Top-1.

## H. Evaluation (held-out test, 120 answerable + 21 unanswerable)
Top-1 **81.7 %**, Top-3 **93.3 %**, MRR **0.884** (V1 setting on the same data: 70.0 % / 84.2 % / 0.780). Answerable success 74.2 %; right service shown incl. related links 83.3 %; false refusals 17.5 %; confident-answer precision 91.9 %; unanswerable declined 61.9 %; unanswerable answered confidently 4.8 %; refusal precision / recall / F1 0.38 / 0.62 / 0.47. Paraphrase robustness: 85 % of families have every phrasing in the top 3. Leakage checks: fresh families 81.8 %, lexicon not fired 80.3 %, expansion off 79.2 % Top-1. Selection-rule change disclosed (DEVELOPMENT_LOG Issue 9).

## I. Speed (2-core CPU)
Cold start 8.7 s → **4.7 s** (cached lexical index). Retrieval 79 ms median; full answer **92 ms median**, p95 0.4 s; repeated question 14 ms. Rebuild after a data fix: 0 of 8,188 chunks re-embedded (incremental).

## J. Tests
`python -m pytest` → **90 passed, 0 failed, 0 xfailed, 0 skipped** (16 s). The V1 xfail now passes and is a normal test.

## K. UI
Five pages: Ask Dalil (examples from 5 agencies plus an out-of-scope example, loading state, answer card with sections and [n] citations, possible-match badge, related-services card, collapsible evidence, latency), Explore (filter by agency), Knowledge base (computed coverage, field coverage, coverage gaps, quarantine table), Evaluation (V1 vs V2, by language and style, leakage checks, every test question), About (methodology + diagram). Bilingual with RTL Arabic (including the input box), mobile-friendly, disclaimer on every page.

## L. Errors & development history
13 documented issues in `docs/DEVELOPMENT_LOG.md`, including: the V1 threshold false refusal; missing captures across two machines; downloads blocked by site scripts; SharePoint `<form>` deleting whole pages; another service's description taken from "related services" cards; guessed Arabic URLs (404 / wrong content); a description-required validation rule; slow cold start; a selection rule that couldn't separate configurations; leakage checks; the tanween normalisation fix; stale integration tests; and the too-strict refusal (related links).

## M. GitHub
Public-ready contents: code, tests, docs, scripts, benchmark, results, and `data/processed/` (prebuilt index). Excluded: raw captures (`data/raw/official/*`), `lexical_index.pkl`, `.venv`, caches, zips, chat-context files.

## N. Licensing / data
Raw HTML captures are not in git (redistribution terms unchecked, ~200 MB). The processed knowledge base contains verbatim official text: keep the repo **private**, or check each agency's terms before going public. The V1 commit contains the MC raw capture in history (see `docs/DEPLOYMENT.md`).

## O. Security
Audit of code, docs and data for API keys, tokens, passwords, `.env` files and private Windows paths: none found in tracked files. No secrets are needed to run.

## P. Deployment
Not deployed (localhost, as requested). Ready for Streamlit Community Cloud; steps in `docs/DEPLOYMENT.md`.

## Q. Manual tests
10 questions with recorded results in `docs/MANUAL_ACCEPTANCE_TESTS.md`: 6 pass, 2 partial, 2 fail (verb-form Arabic false refusal; driving licence shown as a lookalike possible match).

## R. Documentation
`README.md`, `PROJECT_REPORT.md`, `LEARNING_GUIDE.md` (concepts + 30 interview Q&A), `docs/DEVELOPMENT_LOG.md`, `docs/methodology.md`, `docs/architecture.md`, `docs/data_provenance.md`, `docs/MANUAL_ACCEPTANCE_TESTS.md`, `docs/DEPLOYMENT.md`, `docs/CV_BULLETS.md`, `docs/PORTFOLIO.md`, `data/README.md`, `docs/archive/HANDOVER_V2_2026-10-01.md`.

## S. CV
**Dalil AI — Bilingual Saudi Public-Service Retrieval Assistant.** An Arabic–English assistant that answers natural questions about Saudi government services using only verbatim official text from 799 services across 10 agencies, with citations and calibrated refusal. Bullets: see `docs/CV_BULLETS.md`.

## T. First 10 concepts to learn
Pipeline · embeddings & cosine similarity · hybrid retrieval · Top-1/Top-3/MRR · thresholds & false refusals · dev/val/test & leakage · raw vs processed data & provenance · why no LLM · the parser bugs · limitations (`LEARNING_GUIDE.md` Part 4).

## U. Remaining work
| Item | Priority | Human time |
|---|---|---|
| Decide private vs public repo (data licence), then deploy to Streamlit Community Cloud | CRITICAL (before sharing a link) | 15–20 min |
| Export the captured CHI Arabic pages: allow automatic downloads for chi.gov.sa in Chrome, then re-run the export snippet | RECOMMENDED | 5 min |
| Add Interior/Absher everyday topics from a connection where moi.gov.sa / absher.sa load normally (no bypassing) | RECOMMENDED | 30–60 min plus a rebuild |
| Ask 2–3 native speakers to write 30 new test questions (independent benchmark) | RECOMMENDED | 1 hour |
| Cross-encoder re-ranker for sibling services; clarifying questions for ambiguous queries | OPTIONAL | — |
