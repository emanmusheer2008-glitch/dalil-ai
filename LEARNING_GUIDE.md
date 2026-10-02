# Dalil AI — Learning Guide

This guide explains every idea in Dalil using Dalil's own examples, so you can explain the project confidently in a university interview. Read Part 1 once, then practise Part 3 out loud.

---

## Part 1 — The project in 60 seconds

**Problem.** To get something done with the Saudi government you must know which ministry handles it, what the service is officially called, and the government wording. Someone who types "book a business name" won't find *Trade Name Reservation* by keyword search.

**Solution.** Dalil lets you ask once, in Arabic or English, in your own words. It searches 799 official services from 10 agencies, picks the matching ones, and shows a structured answer built **only from official sentences**, with links and capture dates. If it isn't sure it says "possible match", and if its sources don't cover the question it declines.

**Proof it works (held-out test set).** The right service is ranked first 81.7 % of the time and is in the top 3 93.3 % of the time (the V1 method on the same data: 70.0 % and 84.2 %). Arabic 86.8 %, English 79.4 % Top-1.

**What makes it trustworthy.** No language model writes text, so it can't invent a fee. Every fact has a source. Thresholds were calibrated on separate data, and the error rates are published.

---

## Part 2 — Concepts, explained with Dalil

### 2.1 Architecture (the pipeline)

```
official pages → capture → parse → validate → knowledge base → chunks → embeddings / indexes
question → expansion → hybrid retrieval → ranking → decision → grounded answer → official sources
```

The top line runs **offline**, once, when the data changes (`python -m src.indexing.build_index`). The bottom line runs **online** for each question (`src/engine.py`, about 90 ms).

### 2.2 Dataset; raw vs processed data

- **Raw data** is exactly what was downloaded: each service page's HTML, URL, HTTP status and time (`data/raw/official/v2/dalil_v2_zatca.json`). It is never edited. It is the *evidence*.
- **Processed data** is what Dalil uses: one clean record per service with fields like `steps_en`, `fees_ar`, `official_url_en` (`data/processed/services.jsonl`).
- Why keep both? If the parser has a bug (it did: Issue 5 in the development log), you fix the code and re-run. You never have to re-download or hand-edit the evidence.
- Example: ZATCA's raw file has 322 pages (161 services × EN + AR), all HTTP 200, so all 161 services were indexed. CHI's raw file had 15 Arabic pages returning 404, so CHI is English-only.

### 2.3 Parsing, validation, quarantine

Every government site is built differently: SharePoint (MoJ, CHI, ZATCA), Drupal (MoMAH, SFDA), custom (HRSD). The parser looks for *labelled* content (a "Steps" tab, a "Required Documents" heading, `<p class="fw-bold">Service Fees</p><p>Free</p>`) and maps labels to fields in both languages. Anything unlabelled is ignored. **Validation** then rejects records that aren't on an official `gov.sa` HTTPS page, aren't in the language they claim, are duplicates, or contain no real official content. Rejected records go to **quarantine** with a reason, visible on the Knowledge base page.

Real lesson: the parser once took a Municipalities service's "description" from the *Related services* cards, which is another service's text. It looked plausible, and that is exactly why it was dangerous. A cross-record duplicate check found it.

### 2.4 Embeddings and multilingual embeddings

An **embedding** turns text into a list of numbers (here 384) so that texts with similar *meaning* end up close together. Dalil uses `paraphrase-multilingual-MiniLM-L12-v2`, which was trained so that an Arabic sentence and its English translation land near each other. That is why «كيف أعادل شهادتي» can match an English-only service text, and why "book a business name" can land near "Trade Name Reservation" although they share no words.

### 2.5 Semantic search and cosine similarity

**Semantic search** = embed the question, embed every chunk, find the closest ones. "Close" is measured by **cosine similarity**: the cosine of the angle between two vectors (1 = same direction, 0 = unrelated). Dalil normalises all vectors to length 1, so cosine similarity is just a dot product. For 8,188 chunks that is one matrix-vector multiplication in NumPy, a few milliseconds. That is why FAISS (a library for millions of vectors) isn't needed.

### 2.6 Keyword (lexical) search and character n-grams

Semantic models can blur exact names: *Cancel trade name reservation* and *Trade Name Reservation* look almost the same to them. **Lexical search** checks the actual letters. Dalil uses TF-IDF over **character 3–5-grams**: "reservation" → "res", "ese", "ser", … This handles Arabic prefixes («والسجل», «بالسجل», «السجل» share n-grams) and typos ("reserv a trade nme"). TF-IDF gives rare n-grams more weight than common ones.

### 2.7 BM25

BM25 is the classic search-engine formula for *words*: it rewards documents that contain the query's words, more for rare words, and with diminishing returns for repeats. Dalil implements it with light Arabic/English stemming. **Honest result:** the selection on dev+val gave BM25 weight 0, because character n-grams already covered what it adds. You can say: "I implemented it, measured it, and dropped it because it didn't help."

### 2.8 Hybrid retrieval

Combine the signals: `score = 0.3 × dense + 0.3 × char-n-grams + 0.1 × title coverage`. On the test set: dense alone 71.7 % Top-1, char alone 68.3 %, **hybrid 81.7 %**. They fail on different questions, so together they're better.

### 2.9 Query expansion and the lexicon

People say "business name", the government says "trade name". `src/retrieval/lexicon.py` holds about 60 general entries (English and Saudi colloquial Arabic) that *add* the official term to the query: "I want to book a business name" → "+ trade name, reserve reservation". It never removes the user's words and contains no question-to-answer mappings. It is applied to the lexical signals only. Measured effect: Top-1 79.2 % without it vs 81.7 % with it.

### 2.10 Chunking and title weighting

Each service is split per language into a **title chunk** and sections: overview, conditions, documents, steps, fees & details, notes. Why? A question about fees should match the fees section, not a whole page, and the evidence shown can point to the exact paragraph. Each chunk is embedded together with its service title and agency, so "Pay the fees" isn't meaningless on its own. **Title coverage** is the share of a service title's words that appear in your question. It pushes down siblings whose titles add words you didn't say ("**Extension of** trade name reservation").

### 2.11 Ranking and max-pooling

Scores are per chunk, but answers are per service. Dalil takes each service's **best** chunk score (max-pooling) and ranks services by it. **Reranking** means a second, more careful pass over the top few results (e.g. a cross-encoder). Dalil does *not* use a learned reranker; title coverage is a cheap reranking signal. A cross-encoder is listed as future work.

### 2.12 Thresholds, false refusals and unsupported detection

The top score tells you how well the best service matches. Dalil uses **two thresholds**:
- below **0.285** → decline ("not enough verified information"),
- **0.285–0.391** → "possible match" (shown, with a warning),
- above **0.391** → confident answer.

They were chosen on dev+val data: the lower one keeps ≥80 % of unanswerable questions declined, and the upper one makes ≥90 % of confident answers correct.

- A **false refusal** is declining a question Dalil could have answered. That was V1's problem: "How can I reserve a trade name?" scored 0.413, just under V1's single threshold of 0.424, so it was refused although the right service was first.
- **Unsupported detection** is declining questions outside the corpus ("best pizza in Riyadh", "renew my driving licence"). Dalil declines 62 % of unanswerable test questions. Questions about *nearby* missing services (iqama, driving licence) are the hard part, because they look like government services.

### 2.13 Citations and provenance

**Provenance** = where a fact came from. Every record stores the official URL, capture time and a SHA-256 hash of the captured page. A hash is a fingerprint: if one character of the page changed, the hash changes. Every answer point carries `[n]` linking to its source. A test checks that every point shown occurs **verbatim** in the cited record.

### 2.14 Caching and indexing

- **Indexing** = doing the expensive work once: parsing, chunking, embedding (8,188 chunks ≈ 3 minutes on a laptop CPU). It is saved to `embeddings.npy`.
- **Caching:** the model loads once per server process (`st.cache_resource`). The fitted TF-IDF is saved too, which cut cold start from 8.7 s to 4.7 s. A repeated question's embedding is reused (14 ms). A *fingerprint* (hash of all chunk texts) detects when the cache is stale, and only changed chunks are re-embedded.

### 2.15 Metrics: Top-1, Top-3, MRR, precision/recall/F1

- **Top-1** = % of questions where the correct service is ranked first (81.7 %).
- **Top-3** = correct service in the first three (93.3 %).
- **MRR** (mean reciprocal rank) = the average of 1/rank: rank 1 → 1, rank 2 → 0.5, rank 3 → 0.33. Dalil: 0.884.
- **Precision / recall / F1 for refusal:** treat "decline" as the prediction. *Recall* = share of unanswerable questions that were declined (61.9 %). *Precision* = share of declines that were truly unanswerable (38.2 %; most declines are answerable questions that scored low). F1 combines the two (0.47).
- **Why not one "accuracy"?** A system that refuses everything is never wrong but is useless. Dalil reports usefulness (answerable success 74.2 %, false refusals 17.5 %) *and* safety (unanswerable declined 61.9 %, confident wrong answers on unanswerable questions 4.8 %) side by side.

### 2.16 Latency

Time from question to answer: about **90 ms median** on a 2-core CPU, plus a one-time **cold start of 4.7 s** when the server starts (mostly loading the language model).

### 2.17 Train/dev/val/test and evaluation leakage

- **Dev** split: try many settings (576 combinations). **Val:** confirm the choice. **Test:** report only, never used for decisions.
- **Leakage** = information from the test set influencing the system, which makes the scores too optimistic. Dalil's checks: the vocabulary list was **frozen before** 100 new questions were written; results are also reported on questions where the vocabulary never fired, and with expansion switched off (81.8 %, 80.3 % and 79.2 % Top-1, all close to the headline 81.7 %).
- **An honest disclosure to mention:** the selection rule was changed after a first run (the original 45-question validation split picked the old V1 setting by 0.002 MRR). The change and both results are documented.

### 2.18 Responsible AI

No invented facts (no generator; a verbatim-grounding test). Honest uncertainty (three-way decision). No bypassing of website protections. No machine translation presented as official. Published limitations. A disclaimer saying it is not an official government service. No user data stored.

### 2.19 Deployment

**Localhost** = running on your own computer. **Deployment** = running on a server with a public HTTPS link. Dalil is ready for Streamlit Community Cloud (free): the code reads only relative paths, the prebuilt index is in the repo, and the model downloads on first start. The difference from development: no raw captures on the server, a fresh cold start after each redeploy, and a shared CPU.

### 2.20 RAG vs Dalil

**RAG** (retrieval-augmented generation) = retrieve passages, then a language model *writes* an answer from them. Dalil does the **retrieval and grounding** part and arranges the passages itself, with no generation. So the honest name is "retrieval-grounded assistant", not "generative RAG".

---

## Part 3 — Interview questions and answers

### Beginner
**1. What problem does Dalil solve?**
Government service information is spread over many ministry websites with official wording people don't know. Dalil lets you ask once in Arabic or English and get the matching official service, its steps, documents and fees, with links, from 10 agencies.

**2. Who would use it?**
Residents, students, small-business owners: anyone who knows *what* they want ("book a name for my shop", «أبي أستقدم خادمة») but not *which* service or ministry.

**3. What does "bilingual" mean here?**
Questions and official text are in both Arabic and English. An Arabic question gets the official Arabic text when it exists (780 of 799 services). Otherwise Dalil shows the English text and says so; it never machine-translates.

**4. What's the dataset?**
799 official service pages from ZATCA, Municipalities, Justice, HRSD, Commerce, Education, Foreign Affairs, SFDA, CHI and Hajj, captured 30 Sep – 1 Oct 2026, each with URL, capture date and a hash.

### Intermediate
**5. Why multilingual embeddings?**
Users write "book a business name" or colloquial Arabic, and the pages use formal terms, sometimes in the other language. A multilingual model maps meaning across wording and across languages; that is how an Arabic question can match English-only text.

**6. Why not just keyword search?**
Keywords miss synonyms and paraphrases ("business name" ≠ "trade name"). On my test set, character-n-gram search alone gets 68.3 % Top-1 vs 81.7 % for hybrid.

**7. Why combine semantic and lexical retrieval?**
They fail differently. Embeddings understand meaning but blur sibling services; n-grams match exact names, Arabic word forms and typos. Weighted together they beat either alone by 10+ points Top-1.

**8. What is chunking and why section-level chunks?**
Splitting each service into overview, conditions, documents, steps and fees, per language. A fees question matches the fees text, and I can show exactly which paragraph supports the answer.

**9. What is MRR?**
The average of 1/(rank of the first correct result). It rewards getting the right answer near the top, not only first. Dalil: 0.884.

### Technical
**10. How does the decision to answer or decline work?**
I take the best service's score and compare it with two thresholds fitted on dev+val: below 0.285 decline, 0.285–0.391 "possible match", above that a confident answer. The targets were ≥80 % of unanswerable questions declined and ≥90 % of confident answers correct.

**11. Why didn't you use FAISS?**
At 8,188 vectors, an exact dot product in NumPy takes a few milliseconds. FAISS is for millions of vectors, and it also failed to install on my Windows setup.

**12. What is title coverage?**
The share of a service title's words that appear in the question. It helps with siblings: if you say "reserve a trade name", *Extension of trade name reservation* has extra words you didn't use, so it ranks lower.

**13. Did you use BM25?**
I implemented it with Arabic/English stemming and included it in the grid search. The dev+val selection gave it weight 0, so the final system doesn't use it. Measuring and dropping a component is a valid result.

**14. What is query expansion, and isn't it cheating?**
It adds official terms when everyday words appear ("business name" → "trade name"). It's general vocabulary, not question-to-answer rules. To check, I froze it before writing 100 new test questions and also measured with it switched off: 79.2 % vs 81.7 % Top-1, so most of the performance doesn't depend on it.

### Evaluation
**15. Why isn't accuracy enough?**
A system that refuses every question is never "wrong" but is useless. I report usefulness (74.2 % answered with the right service, 17.5 % false refusals) and safety (61.9 % of unanswerable questions declined, 4.8 % answered confidently) separately.

**16. How did you build the benchmark?**
250 questions in 121 "families". Each family is one need asked formally, conversationally, short, misspelled, in MSA and Saudi Arabic, and mixed. Families were split into dev/val/test, so paraphrases of a test question never appear in tuning.

**17. How did you prevent evaluation leakage?**
Tuning only on dev+val; families never split across splits; vocabulary frozen before writing new families; separate reporting on fresh families (81.8 %) and on questions where expansion never fired (80.3 %). I also disclose that I changed the selection rule after a first run.

**18. What is paraphrase robustness?**
Whether every phrasing of the same need works. In 85 % of test families, every phrasing finds the right service in the top 3 (V1 setting: 72 %).

**19. How does Arabic compare to English?**
Arabic 86.8 % Top-1, English 79.4 %. Colloquial questions are harder than formal ones (75 % vs 87 %), and mixed Arabic-English has only 4 test questions, too few to conclude anything.

### Failure and debugging
**20. What was one important failure?**
V1 refused "How can I reserve a trade name?" although the right service was ranked first: its score of 0.413 was just under a single threshold of 0.424. I fixed the general problem (better recall plus a "possible match" middle state), not the one question. Now it ranks first and is shown.

**21. What was the most dangerous bug?**
The Municipalities parser took a service's description from the "Related services" cards, which is another service's text. Nothing crashed, it just looked right. A check for identical text across many services exposed it, and I added a regression test.

**22. What happened with the data across two laptops?**
The handover said nine capture files existed; only five were on the machine, and the browser's stored copies had been cleared. I re-captured the four missing agencies with one generic script, verified HTTP statuses, and stored everything in the project instead of Downloads.

### Design
**23. Why didn't you simply use ChatGPT?**
Cost (I set $0), and risk: a generator can invent a fee or a document, which is unacceptable for government procedures. Retrieval plus verbatim official text is free, reproducible and checkable. A constrained generator could be future work, but only if it is evaluated for faithfulness.

**24. Is this RAG?**
It's the retrieval and grounding half of RAG. There is no generation step, so I call it a retrieval-grounded assistant.

**25. Why Streamlit?**
The whole project is Python, Streamlit deploys free, and the measured quality depends on the retrieval, not on the web framework. A separate API would add complexity without improving anything measured.

### Responsible AI
**26. How do you prevent unsupported answers?**
No generator. A grounding test checks every displayed point is verbatim in its source. A calibrated decline threshold. A "possible match" warning when unsure. Coverage gaps are listed in the app.

**27. Did you scrape government websites? Is that OK?**
I captured public pages through a normal browser at about one page every 2 seconds, with no logins and no bypassing. Sites that blocked access (my.gov.sa, Absher, MOH, GOSI) were left out and listed as gaps. Raw captures aren't published because redistribution terms are unclear.

### Performance
**28. Why was the app slow, and how did you improve it?**
I profiled it: the model loaded once, but the character TF-IDF index was refit at every start (3.1 s), and every rebuild re-embedded all chunks. I cached the fitted index (cold start 8.7 → 4.7 s) and made embedding incremental (a small data fix re-embeds 0 of 8,188 chunks). A warm answer takes ~90 ms.

### Deployment
**29. How does the deployed system differ from development?**
Development rebuilds from raw captures on my laptop. The deployed app only reads the prebuilt index and knowledge base from the repo, downloads the model on first start, runs on a shared CPU, and has no access to raw captures or my files.

### Limitations
**30. What can Dalil not do?**
It covers 10 agencies, not all government services (no Absher, MOH, GOSI). It sometimes offers a lookalike "possible match" for nearby services it doesn't have (e.g. driving licence). It confuses sibling services. It can't answer personal questions about your own case. The benchmark is self-written and small for unanswerable questions, and pages change, so the data needs re-capturing.

---

## Part 4 — The 10 things to master first

1. The pipeline (offline indexing vs online answering) — Part 2.1.
2. Embeddings and cosine similarity — 2.4, 2.5.
3. Why hybrid beats either alone (with the numbers) — 2.8.
4. Top-1, Top-3, MRR — 2.15.
5. Thresholds, false refusals, the trade-name story — 2.12, Q20.
6. Train/dev/val/test and leakage — 2.17.
7. Raw vs processed data and provenance — 2.2, 2.13.
8. Why no LLM / not generative RAG — 2.20, Q23.
9. The parser bugs and what they taught you — 2.3, Q21.
10. Limitations, stated before anyone asks — Q30.
