## The problem

Information about Saudi public services is spread across many official websites, in Arabic and English, and written in administrative language. People often don't know the official name of the service they need ("I want to reserve a name for my shop" → *Trade Name Reservation*). Generic chatbots can answer fluently but may invent fees, documents or steps — unacceptable for government procedures.

## What Dalil does

Dalil is a **multilingual retrieval-grounded assistant**. It answers Arabic or English questions over the **currently indexed official corpus** — 73 Ministry of Commerce e-services (of 75 in its catalogue), each with official English and Arabic text — by:

1. finding the service whose official description best matches the question,
2. showing that service's **official text** (only fields the source states),
3. highlighting the passages that matched, and
4. linking to the official page, with the capture date.

When the evidence is weak, Dalil says it doesn't have enough verified information instead of guessing.

Dalil does **not** generate text with a language model, so it is not "generative RAG"; it implements the retrieval and grounding half of RAG.

## Data provenance

- **Source**: the Ministry of Commerce e-services catalogue on `mc.gov.sa` (service-detail pages in English and Arabic).
- **Capture**: done from a normal browser session at a low request rate (roughly one page every few seconds). No access control was bypassed. `my.gov.sa` blocks automated access (HTTP 403 / Cloudflare), so it was *not* used.
- **Stored per service**: official URLs (EN/AR), capture timestamp, SHA-256 of each captured page's HTML, the page's own "last modified" date, and a verification status.
- **Quarantine**: the 18 hand-written seed records from the first prototype could not be traced to captured official text, so they are kept on disk but excluded from answers.

## Retrieval

- **Chunks**: each service is split per language into sections — overview, conditions, required documents, steps, fees & service details.
- **Dense retrieval**: `paraphrase-multilingual-MiniLM-L12-v2` (free, CPU) embeds chunks into 384-d normalised vectors, cached on disk. Cosine similarity = dot product. No FAISS needed at this scale.
- **Lexical retrieval**: TF-IDF over character 3–5-grams with Arabic orthographic normalisation.
- **Hybrid**: `α·dense + (1−α)·lexical`, with chunk scores max-pooled per service.
- The method, α and the refusal threshold are chosen on a **calibration** half of the benchmark and reported on a held-out **test** half (see the Evaluation page).

## Responsible AI

- Official text only; no machine translation presented as official. If a service has no official text in the user's language, Dalil says so and shows the original.
- Every answer carries a source link and capture date.
- Calibrated refusal for unsupported questions; limitations reported openly.
- No user accounts, no query logging, no paid APIs, no secrets.

## Limitations

- One ministry, 73 services — not all Saudi government services.
- The benchmark is small and was written by the project author with AI assistance.
- A single similarity threshold cannot perfectly separate "government-like but not indexed" questions from answerable ones.
- Official pages change; the data must be re-captured periodically.

## Future work

Official open-data feeds of services (adapter already built), more agencies via saved official pages, independent Arabic-speaking benchmark writers, a cross-encoder re-ranker, and an optional generator constrained to cite retrieved chunks.

*Dalil AI is an independent educational project and is not an official Saudi government service.*
