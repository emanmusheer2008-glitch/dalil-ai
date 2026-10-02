# CV wording (final, V2)

Every number below is from the executed final evaluation (`data/evaluation/results_v2.json`, held-out test split) or the build report.

**Project title**
Dalil AI — Bilingual Saudi Public-Service Retrieval Assistant

**One-line description**
An Arabic–English assistant that answers natural questions about Saudi government services using only verbatim official text from 799 services across 10 agencies, with citations and calibrated refusal.

**CV bullets (exactly two)**
- Built a bilingual retrieval-grounded assistant over 799 official Saudi e-services from 10 government agencies (captured, parsed and validated from source-specific HTML with full provenance), combining multilingual embeddings, character-n-gram TF-IDF and title-coverage ranking. It reached 81.7% Top-1 / 93.3% Top-3 retrieval (MRR 0.884) on a held-out 250-question Arabic/English benchmark, up from 70.0% / 84.2% for the V1 method.
- Designed a leakage-checked evaluation (dev/val/test by paraphrase family, vocabulary frozen before new test questions) and a calibrated answer / possible-match / decline policy that cut false refusals from 26.7% to 17.5% with 91.9% of confident answers correct. Also profiled and cached the pipeline (cold start 8.7 s → 4.7 s, ~90 ms per answer) at $0 running cost, with 90 automated tests.

**GitHub description (≤ 350 characters)**
Bilingual (Arabic–English) retrieval-grounded assistant for official Saudi public-service information: 799 services from 10 agencies, hybrid multilingual retrieval, calibrated refusal, verbatim-grounded answers with citations. Streamlit app, leakage-checked evaluation, 90 tests. Independent educational project, not a government service.

**Portfolio-site description**
Dalil (دليل, "guide") lets people ask about Saudi government services in their own words, in Arabic or English, instead of searching ministry websites. It finds the matching official services across 10 agencies and shows a structured answer built only from official text, with sources, or says honestly when it doesn't know. I built the full pipeline: polite data capture, a parser for ten different government website layouts, hybrid multilingual retrieval, a calibrated three-way answer policy and a leakage-checked evaluation (81.7% Top-1 on held-out questions).

*Wording to avoid:* "covers all government services", "100% accurate", "RAG chatbot / generative AI", any implied official affiliation.
