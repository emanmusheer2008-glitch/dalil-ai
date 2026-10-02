# CV / application wording (truthful — every number is in `data/evaluation/results.json`)

**Dalil AI — Bilingual Arabic–English Public-Service Information Assistant**
*Independent AI Engineering Project | 2026*

- Built and evaluated a multilingual semantic retrieval system that answers Arabic and English questions over 73 official Saudi Ministry of Commerce e-services, returning official text with source citations and declining questions outside its verified corpus.
- Engineered a provenance-tracked data pipeline (capture → parsing → Unicode/Arabic normalisation → validation → de-duplication → quarantine) producing a bilingual knowledge base with official English and Arabic text for every indexed service; unverifiable records were quarantined rather than used.
- Compared lexical (TF-IDF character n-grams), dense (multilingual sentence embeddings) and hybrid retrieval on a 129-question Arabic/English benchmark with a calibration/test split; the selected hybrid reached 69.6% Top-1, 82.6% Top-3 and 0.78 MRR on held-out questions.
- Showed that multilingual embeddings enable cross-lingual retrieval where keyword search fails (Arabic questions against English-only text: 59.1% vs 2.3% Top-1).
- Implemented calibrated refusal that declined 100% of held-out unanswerable questions (including real government services not in the corpus), and documented its cost: only half of answerable held-out questions were answered correctly.
- Delivered a tested, zero-cost Streamlit application with right-to-left Arabic support, cached embeddings (~19 ms mean query latency on CPU) and 50+ automated tests.

**Short version (one line):**
Built a bilingual Arabic–English retrieval-grounded assistant over official Saudi public-service data with provenance tracking, hybrid multilingual retrieval (0.78 MRR on held-out questions) and calibrated refusal.

**Skills line:** Python · NLP · multilingual embeddings (sentence-transformers) · information retrieval · scikit-learn · pandas/NumPy · evaluation design · Arabic text processing · Streamlit · pytest · data provenance / responsible AI

**Do not claim:** "covers all Saudi government services", "generative RAG / LLM-powered", "official government tool", or any accuracy figure not in `results.json`. If you re-run the evaluation, update the numbers here.
