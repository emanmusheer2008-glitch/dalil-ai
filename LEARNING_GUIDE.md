# Dalil AI — Learning Guide

A plain-language explanation of every idea in this project, so you can explain it confidently in an interview. Numbers quoted here come from `data/evaluation/results.json`; re-run the evaluation and they may change slightly.

---

## 1. What is an embedding?

An **embedding** is a list of numbers (a vector) that represents the *meaning* of a piece of text. Dalil's model turns any sentence into 384 numbers. Sentences with similar meaning end up with similar vectors, even when they use different words:

- "How do I reserve a name for my business?"
- "Trade Name Reservation"

share almost no words, but their vectors point in nearly the same direction.

The model is a neural network trained on millions of sentence pairs that mean the same thing, so it learned to place paraphrases close together.

## 2. What is semantic search?

**Keyword search** finds documents that contain the same words as the question. **Semantic search** finds documents with the same *meaning*:

1. Embed every document once (offline).
2. Embed the question (at query time).
3. Return the documents whose vectors are closest to the question's vector.

## 3. What is cosine similarity?

A way to measure how close two vectors are by the **angle** between them:

- 1.0 → same direction (same meaning)
- 0 → unrelated
- negative → opposite

Formula: `cos(a, b) = (a · b) / (|a| × |b|)`. Dalil **normalises** every vector to length 1 when it is created, so cosine similarity becomes a plain dot product: `scores = embeddings @ query_vector`. For 730 chunks this takes well under a millisecond with NumPy.

## 4. Why multilingual embeddings?

`paraphrase-multilingual-MiniLM-L12-v2` was trained so that a sentence and its **translation** get similar vectors. So an Arabic question can match English text, and the reverse, without translating anything. Dalil measures this directly in its *cross-lingual* experiment (Arabic questions against English text only).

Why this particular model: free (Apache-2.0), small (≈470 MB), runs on a CPU, supports 50+ languages including Arabic, and it already worked in the first prototype.

## 5. What is RAG?

**Retrieval-Augmented Generation**: (1) *retrieve* relevant documents for a question, then (2) give them to a *generative* language model (like GPT or Claude) that writes an answer based on them.

## 6. Is Dalil technically RAG?

**Partly — and it's important to say this honestly.** Dalil implements the **retrieval** half fully (chunking, embeddings, ranking, evidence, citations). It does **not** use a generative model to write answers. Instead an **answer composer** shows the official text of the best-matching service, field by field, plus the matching passages and the link.

The right description is a **"multilingual retrieval-grounded assistant."** Reasons for this design:

- **Zero cost**: no paid LLM API.
- **No hallucination risk**: every sentence shown is official text, never generated.
- **Deployability**: a local LLM good at Arabic would need far more memory than free hosting provides.

Adding a generator later would turn it into full RAG; the retrieval and evidence pipeline would stay the same.

## 7. What is chunking?

Splitting a document into smaller pieces before embedding. Dalil splits each service by **section** and **language**: overview, conditions, required documents, steps, and fees/service details — in English and Arabic.

Why: one vector for a whole page blurs everything together. With sections, a question about *fees* matches the fees chunk strongly, and Dalil can show exactly that passage as evidence. Each chunk keeps its metadata (service id, title, agency, section, language, official URL), so it can always be cited.

## 8. Why cache embeddings?

Embedding all chunks takes a noticeable amount of time on a CPU; recomputing on every app start would be slow and wasteful. `python -m src.indexing.build_index` saves them to `data/processed/embeddings.npy`, with a **fingerprint** (SHA-256 of every chunk's text) in `index_meta.json`. On the next build, if the fingerprint and model are unchanged, the cache is reused. If `chunks.csv` changes but the embeddings don't, the app refuses to load a stale index (`StaleIndexError`).

## 9. Top-1 / Top-3 accuracy

For each benchmark question we know the correct service.

- **Top-1 accuracy** = share of questions where the correct service is ranked **first**.
- **Top-3 accuracy** = share where it is somewhere in the **first three**.

Some questions have more than one acceptable answer (e.g. "annual confirmation" exists for an establishment *and* for a company); those alternatives are listed in the benchmark.

## 10. What is MRR?

**Mean Reciprocal Rank.** For each question, score `1 / rank` of the first correct result (rank 1 → 1.0, rank 2 → 0.5, rank 3 → 0.33, not found → 0), then average. It rewards putting the right answer near the top, not just anywhere.

## 11. How unsupported-question detection works

Every search returns a **top score** (how similar the best service is). Unrelated questions ("best pizza in Riyadh") produce low top scores. Dalil declines to answer when the top score is below a **threshold**.

The threshold is **not guessed**. The benchmark is split in two halves:

- **Calibration half** — used to choose the retrieval method *and* the threshold that best balances "answer answerable questions correctly" against "decline unanswerable ones" (balanced accuracy).
- **Test half** — never used for choosing anything; used only to report results.

The hardest cases are **real government services that are not in Dalil's corpus** (e.g. passport renewal, VAT registration). These look "government-like", so their scores are higher. The evaluation reports them separately (`not_indexed` vs `out_of_domain`). This is an honest limitation of a single-score threshold.

## 12. Why hallucination is dangerous here

A made-up fee, document or deadline for a government procedure can cost someone money, time, or a rejected application. That's why Dalil never generates facts: it displays official text only, shows the source link, and declines when evidence is weak.

## 13. Why provenance matters

**Provenance** = where a piece of information came from and when. Every Dalil record stores the official URL, capture timestamp, a SHA-256 hash of the captured HTML, the source's own "last modified" date, and a verification status. This lets anyone trace an answer back to its source and see how fresh it is. The original 18 hand-written seed records had no such trail, so they were **quarantined** (kept for the record, excluded from answers).

## 14. How Dalil works end-to-end

1. **Capture**: Ministry of Commerce service pages (English + Arabic) were captured in a normal browser at a low rate. (Scripts were blocked by my.gov.sa, and Dalil does not bypass blocks.)
2. **Ingest**: `src/ingestion/` parses each page into the schema, cleans Unicode, validates URLs/domains/provenance, removes duplicates, and quarantines anything unverified.
3. **Chunk**: each service → section chunks per language.
4. **Embed + cache**: chunks → 384-d normalised vectors → `embeddings.npy`.
5. **Query**: the question is embedded; dense (cosine) and lexical (TF-IDF) scores are combined; chunk scores are max-pooled per service.
6. **Decide**: top score vs. calibrated threshold → answer or decline.
7. **Compose**: official fields in the question's language (if an official version exists), matching passages, and the source link.

## 15. Why we did not need FAISS

FAISS is a library for searching **millions** of vectors quickly. Dalil has a few hundred chunks: a NumPy dot product over all of them (`exact search`) takes well under a millisecond, is exact (no approximation), and has no installation problems. FAISS also failed to install on the development machine. If the corpus grows to hundreds of thousands of chunks, an approximate index (FAISS, hnswlib) would become worthwhile.

## 16. How the website talks to retrieval

There is no separate server/API. The Streamlit app (`app.py`) imports the Python modules directly:

- on start-up it loads `chunks.csv`, `embeddings.npy` and the model once (`st.cache_resource`),
- when you press *Ask*, it calls `Retriever.search()` and then `compose()`,
- it renders the result as HTML cards (every piece of text is HTML-escaped).

Streamlit re-runs the script on each interaction; caching keeps the heavy objects in memory.

## 17. Limitations that remain

- **Coverage**: one ministry (Ministry of Commerce), 73 indexed services (75 in its catalogue; 2 pages had no description and were quarantined). Not "all Saudi government services".
- **Benchmark**: written by the project author with AI assistance; small; may share phrasing habits with the author.
- **Threshold**: a single similarity score can't perfectly separate "government-like but not indexed" questions.
- **Freshness**: pages can change; the capture date is shown, and the data must be re-captured periodically.
- **No generation**: answers are official text, not a tailored explanation.
- **Arabic dialects**: tested with Modern Standard and some Saudi colloquial phrasing only.

---

## Interview questions (with short answers)

1. **What does Dalil do?** — It answers Arabic or English questions about Saudi public services by retrieving the matching official service and showing its official text and source link, and it declines when it has no verified information.
2. **Is it a chatbot?** — No. It's a retrieval-grounded assistant: it never generates facts; it shows official text.
3. **Why not use ChatGPT?** — Cost (no paid APIs), and hallucination risk for government information. Every sentence Dalil shows is traceable.
4. **Where does the data come from?** — The Ministry of Commerce's official e-services catalogue (mc.gov.sa), English and Arabic, captured at a low rate with provenance (URL, date, hash).
5. **Why not scrape my.gov.sa?** — It blocks automated access (403 / Cloudflare). Bypassing that would be unethical, so I used an official source that is openly accessible and designed loaders for open-data files and manually saved pages.
6. **What happened to the original 18 records?** — They were hand-written summaries I could not verify, so they're quarantined — visible on the Knowledge Base page, but never used for answers.
7. **Which model and why?** — `paraphrase-multilingual-MiniLM-L12-v2`: free, CPU-friendly, supports Arabic, aligns translations in the same vector space.
8. **What is cosine similarity?** — The cosine of the angle between two vectors; with normalised vectors it's a dot product.
9. **Why chunk by section?** — Precise evidence: a fee question matches the fee passage, which is then shown to the user.
10. **Lexical vs dense vs hybrid?** — Lexical = keyword overlap (TF-IDF over character n-grams), dense = meaning (embeddings), hybrid = weighted sum. I compared all three on the benchmark and picked the best on the calibration split.
11. **Why character n-grams for Arabic?** — Arabic attaches prefixes/suffixes (و، ب، ال…); n-grams still match across word forms.
12. **How did you choose the refusal threshold?** — On a calibration split, maximising balanced accuracy; reported on a separate test split to avoid fooling myself.
13. **What's MRR?** — Average of 1/rank of the first correct answer.
14. **Hardest failure case?** — Real government services that aren't indexed (e.g. VAT registration) — they look similar to indexed business services.
15. **How do you know Arabic works?** — The benchmark has Arabic, English and mixed questions, reported separately, plus a cross-lingual test (Arabic questions vs English-only text).
16. **Why no FAISS?** — A few hundred vectors: exact NumPy search is sub-millisecond; FAISS is for millions.
17. **How do you keep answers from being stale?** — Each answer shows the capture date and the page's own last-modified date, and links to the live page.
18. **How is it tested?** — pytest: schema validation, ingestion, duplicates, missing values, Arabic Unicode, retrieval, refusal, citation preservation, index caching and stale-cache detection, plus integration tests with the real model.
19. **How would you scale it?** — Add official sources via the loaders (open-data adapter, saved pages), re-run the pipeline; switch to an approximate vector index at large scale; add a cross-encoder re-ranker; optionally a generator constrained to cite chunks.
20. **What would you do differently?** — Get independent people (native Arabic speakers) to write benchmark questions, and seek an official open-data feed of services.
