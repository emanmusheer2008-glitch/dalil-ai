# Dalil V3 — lightweight runtime for free hosting

V2 (transformer runtime) stays in the repo unchanged as the research baseline. V3 is a second
runtime for the public API, selected by `DALIL_RUNTIME=lite` (the default). `DALIL_RUNTIME=full`
still serves V2.

## Why
V2 loads PyTorch + a multilingual MiniLM at runtime: ~1.5 GB peak RSS (measured), which crashed
on a 1 GB host and cannot run on free 512 MB tiers.

## What V3 changes (and what it does not)
| | V2 (full) | V3 (lite) |
|---|---|---|
| Query encoder | MiniLM transformer (torch) | static word-vector table distilled offline from the same MiniLM (numpy lookup, IDF-weighted mean, SIF: 2 common components removed) |
| Document vectors | MiniLM chunk embeddings | the **same** precomputed MiniLM chunk embeddings (a matrix; no model) |
| Lexical signals | char n-gram TF-IDF + title coverage + lexicon expansion | same, plus BM25 |
| Scoring / synthesis code | `Retriever`, `synthesize` | same classes (`LiteRetriever` subclasses `Retriever`, overrides only the encoder) |
| Thresholds | calibrated on dev+val | re-calibrated on dev+val with the same rules |
| Follow-ups | none | stateless `context_service_id` on `/ask` |
| Runtime deps | `requirements.txt` | `requirements-prod.txt` (no torch / sentence-transformers / transformers) |
| Peak RSS | ~1,560 MB | **~315 MB** (measured, cold start incl. building the lexical index) |

Static table: `python -m src.lite.build_static --doc-side model --remove-pc 2` (offline, needs the
V2 environment). Vocabulary = corpus + lexicon words and light stems; benchmark questions are not
used. The number of removed components (2) was chosen on dev+val MRR (pc0 0.796, pc1 0.816,
pc2 0.827, pc3 0.815), not on test.

## Evaluation (same benchmark, same split, same protocol — `python -m src.evaluation.evaluate_v3`)
Held-out test: 120 answerable + 21 unanswerable questions. Configuration chosen by pooled dev+val
MRR from 792 candidates; nothing tuned on test; questions unchanged.

| Test metric | V2 full | V3 lite |
|---|---|---|
| Top-1 | 81.7 % | 77.5 % |
| Top-3 | 93.3 % | 89.2 % |
| MRR | 0.884 | 0.839 |
| Arabic Top-1 / Top-3 (n=53) | 86.8 / 98.1 % | 83.0 / 92.5 % |
| English Top-1 / Top-3 (n=63) | 79.4 / 90.5 % | 74.6 / 87.3 % |
| Answerable answered correctly (top-1) | 74.2 % | 66.7 % |
| False refusal | 17.5 % | 24.2 % |
| Confident-answer precision | 91.9 % | 97.7 % |
| Unsupported refused | 61.9 % | 81.0 % |
| Unsupported answered confidently | 4.8 % | 4.8 % |

Reading: V3 retrieves a little worse (−4 pts Top-1) and therefore refuses more often, but when it
answers it is more often right, and it refuses more out-of-scope questions. It is the safer, not
the smarter, runtime. Lexical-only (no vectors) was worse still: Top-1 72.5 %, MRR 0.784.

## Follow-up questions
The client sends the previous answer's `sources[0].service_id` as `context_service_id`. If the new
question is short (≤ 9 words) and asks about an aspect (fees, documents, steps, time…) or refers back
("it", «هذه», «نفس»), the service title is added to the retrieval query. Nothing is stored server-side.
The answer is still produced only from that service's official text; a self-contained new question
ignores the context.

## Known limitations
- 4 pts lower Top-1 than V2; more "related services" instead of answers.
- Action-verb confusion is shared with V2: «كيف أستخرج سجل تجاري لمؤسسة فردية؟» is matched to the
  *deletion* service (V2 does the same). "How do I register a business?" returns related services
  (the right one, mc-38, is in the list) instead of an answer; V2 gives a possible match.
- Traffic violations / iqama renewal are not in the corpus; V3 refuses or, for "pay a traffic
  violation", shows a low-confidence possible match to labor-law violations (V2 does the same).
- No generative model: answers are verbatim official text arranged by templates.
