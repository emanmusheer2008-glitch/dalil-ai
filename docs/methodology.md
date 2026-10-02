## The problem

Information about Saudi public services is spread across many official websites, in Arabic and English, and written in administrative language. People often don't know which agency handles their need, or the official name of the service ("I want to book a name for my shop" → *Trade Name Reservation*, Ministry of Commerce). Generic chatbots answer fluently but can invent fees, documents or steps, which is unacceptable for government procedures.

## What Dalil does

Dalil is a **bilingual retrieval-grounded assistant**. It answers Arabic, English or mixed questions over its **indexed official corpus**: **799 services from 10 Saudi government agencies**, captured from their official websites on 30 Sep – 1 Oct 2026. For each question it:

1. adds official synonyms for everyday words (a small, general bilingual vocabulary),
2. searches every section of every service by **meaning** (multilingual embeddings) and by **spelling** (character n-grams), plus a title-coverage signal,
3. decides how sure it is: **answer**, **possible match** (with a warning), or **decline**,
4. builds a structured answer (direct answer, what you need, documents, steps, fees & processing, who it is for, notes) **only from verbatim official sentences**, each with a citation `[n]`, the official link and the capture date.

Dalil does **not** generate text with a language model, so it is not "generative RAG". It implements the retrieval and grounding half of RAG, which keeps it free to run and unable to invent official facts.

## Data

- **Sources:** official e-service pages of ZATCA, Municipalities & Housing, Justice, HRSD, Commerce, Education, Foreign Affairs, SFDA, Council of Health Insurance, and Hajj & Umrah.
- **Capture:** a normal browser session on each agency's own site, one page every ~2 seconds, no logins and no bypassing of any protection. Sites that blocked access (my.gov.sa, Absher/MOI, MOH, GOSI) are listed as coverage gaps.
- **Stored per service:** official EN/AR URLs, capture time, SHA-256 of each captured page, verification status. Raw captures are kept unchanged, and processing is reproducible.
- **Quality control:** failed or 404 pages, error pages, wrong-language pages, duplicates, placeholders, and records without real official content are **quarantined with a reason** (Knowledge base page).

## Retrieval and decision

- **Chunks:** per service and language, a title chunk plus sections (overview, conditions, documents, steps, fees & details, notes). 8,188 chunks in total.
- **Signals:** cosine similarity of `paraphrase-multilingual-MiniLM-L12-v2` embeddings (cached), TF-IDF over Arabic-normalised character 3–5-grams, and title coverage (penalises sibling services whose titles add words you didn't use). BM25 was also built and tested, but the selection gave it weight 0.
- **Final score per service:** 0.3·dense + 0.3·char + 0.1·title-coverage (best chunk per service). Chosen by grid search on the *dev+val* splits of the benchmark, never on test.
- **Related services:** between 0.200 and 0.285, Dalil declines but lists up to three closest official services as links (the 0.200 floor sits just above the highest out-of-domain dev+val score).
- **Thresholds:** decline below 0.285 (keeps ≥80 % of unanswerable dev+val questions declined); confident answer at ≥0.391 (≥90 % of confident dev+val answers correct). In between: "possible match".

## Evaluation (held-out test split)

250 questions in 121 families (each need asked formally, conversationally, short, misspelled, in MSA and Saudi colloquial Arabic, and mixed), split by family into dev / val / test. Test: 120 answerable + 21 unanswerable.

- **Top-1 81.7 %, Top-3 93.3 %, MRR 0.884** (V1 method on the same data: 70.0 % / 84.2 % / 0.780).
- Arabic Top-1 86.8 %, English 79.4 %.
- 74.2 % of answerable questions are shown with the right service first; 17.5 % are falsely declined; 91.9 % of confident answers are correct.
- 61.9 % of unanswerable questions are declined; 4.8 % (1 of 21) get a confident answer.
- Counting "related official services" links, the right service is shown for 83.3 % of answerable questions.
- Leakage checks: questions written after the vocabulary was frozen give 81.8 % Top-1; with expansion switched off, 79.2 %.

## Responsible AI

- Official text only. No machine translation is presented as official; if a service has no official text in your language, Dalil says so and shows the original.
- Every answer carries source links and capture dates. A test checks that every displayed point occurs verbatim in its source.
- Calibrated uncertainty instead of always answering, and its error rates are reported, not hidden.
- No user accounts, no query logging, no paid APIs, no secrets.

## Limitations

- 10 agencies, **not all Saudi government services**: no Interior/Absher (iqama, ID, traffic), Health or GOSI.
- Questions about nearby services that aren't indexed often get a "possible match" from a lookalike service rather than a refusal.
- Sibling services ("register for VAT" vs "VAT registration verification") are the most common error.
- The benchmark was written by the project author with AI assistance; the unanswerable test set is small.
- Official pages change; the data must be re-captured periodically.

## Future work

A cross-encoder re-ranker for sibling services; clarifying questions for ambiguous queries ("I want to file a complaint"); independent native-speaker benchmark writers; more agencies through official open-data feeds; an optional generator constrained to cite retrieved chunks, evaluated for faithfulness before use.
