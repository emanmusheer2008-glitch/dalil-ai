# Portfolio entry: Dalil AI

**Title:** Dalil AI · دليل — Bilingual Saudi Public-Service Retrieval Assistant
**Live demo:** _[add the Streamlit URL after deployment]_
**Code:** _[add the GitHub URL]_
**Status:** working locally, deployment-ready, independent educational project

## Short summary
Ask about a Saudi government service in Arabic or English, in your own words. Dalil finds the official service across 10 agencies and shows its steps, documents and fees using only official text with sources, or tells you it doesn't know.

## Problem
Service information is split across many ministry websites and written in official terms people don't use ("book a business name" vs *Trade Name Reservation*). Generic chatbots can invent fees or documents, which is unacceptable for government procedures.

## Solution
- Captured 799 official service pages (EN + AR) from 10 agencies politely and with provenance (URL, date, SHA-256).
- One bilingual parser for SharePoint, Drupal and custom layouts, plus validation and quarantine.
- Hybrid retrieval: multilingual embeddings + character n-grams + title coverage, with a small everyday-to-official vocabulary.
- A calibrated decision: answer, "possible match" with a warning, or decline.
- Answers assembled from verbatim official sentences with [n] citations (no language model writes text).

## Architecture
Official pages → capture → parse → validate → knowledge base → chunks → embeddings / TF-IDF → hybrid retrieval → calibrated decision → grounded answer with sources. (Diagram: README, *How it works*.)

## Tech stack
Python · sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) · scikit-learn · NumPy/SciPy · BeautifulSoup · pandas · Streamlit · pytest · Playwright (UI checks). $0 running cost.

## Verified metrics (held-out test set)
| | V1 method | Dalil V2 |
|---|---|---|
| Top-1 / Top-3 | 70.0 % / 84.2 % | **81.7 % / 93.3 %** |
| MRR | 0.780 | **0.884** |
| False-refusal rate | 26.7 % | **17.5 %** |
| Confident answers correct | — | **91.9 %** |
| Arabic / English Top-1 | — | 86.8 % / 79.4 % |
Speed: ~90 ms per answer, 4.7 s cold start (2-core CPU). Tests: 90 passing.

## Screenshots needed
1. Ask, English answer with sections and citations (trade-name question).
2. Ask, Arabic answer, right-to-left.
3. Polite refusal for an out-of-scope question.
4. Mobile view.
5. Knowledge base page (agencies, coverage, quarantine).
6. Evaluation page (V1 vs V2 table, leakage checks).
(Current captures are in `assets/screenshots/`.)

## Key learning
The hardest bugs weren't crashes but *plausible wrong text*: a parser that took a description from a neighbouring "related services" card. Checking data across records, calibrating thresholds on held-out data, and reporting leakage checks mattered more than adding model complexity.

## Limitations
10 agencies (no Absher/MOI, Health or GOSI); questions about nearby missing services can get a lookalike "possible match"; sibling services are the main error; the benchmark was self-written; CHI is English-only.
