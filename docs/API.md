# Dalil AI — HTTP API (V2)

A thin FastAPI layer (`api/`) around the **same** verified V2 engine the Streamlit app uses (`src/engine.py` → `Dalil.ask`). It adds no retrieval or answering logic of its own: no model, threshold, corpus or policy changes.

```
React frontend (future) ──HTTP/JSON──▶ api/main.py (FastAPI) ──▶ src.engine.Dalil
                                                                  ├─ Retriever: multilingual embeddings + char n-grams + title coverage
                                                                  └─ synthesize: answer / possible match / related services / decline
```

Interactive docs: `/docs` (Swagger UI) and `/redoc`. OpenAPI schema: `/openapi.json`.

## Run locally

```bash
pip install -r requirements-dev.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000/docs
```

Startup: the knowledge base (799 services) loads instantly. The model and index load once in a background thread, about 12 s on a 2-core CPU (plus a one-time model download on a brand-new machine). During that time `/health` answers **503 `"loading"`**; every other read endpoint already works, and `/ask` waits up to `DALIL_ASK_WAIT_SECONDS` for the engine.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | 8000 (via the start command) | Port to listen on. The host (Render) sets it. |
| `DALIL_RUNTIME` | `lite` | `lite` = V3 runtime (no PyTorch, ~0.3 GB); `full` = V2 transformer engine (needs `requirements.txt`, ~1.5 GB). |
| `DALIL_CORS_ORIGINS` | *(unset)* → `*` | Comma-separated list of allowed browser origins, e.g. `https://dalil.vercel.app,https://www.dalil.app`. Unset or `*` allows any origin (no cookies/credentials are used, so this is safe for a public read-only API). |
| `DALIL_EAGER_LOAD` | `1` | `1`: load the engine in the background at startup. `0`: load on the first `/ask` (faster boot, slow first question). |
| `DALIL_ASK_WAIT_SECONDS` | `120` | How long `/ask` waits for the engine to finish loading before returning 503. |
| `DALIL_MODEL_PATH` | model id | Optional local path to the embedding model (offline use). |
| `HF_HOME` | `~/.cache/huggingface` | Where Hugging Face caches the model. Point it at a persistent volume to avoid re-downloading on every deploy. |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Name, version, status (cheap) |
| GET | `/health` | Readiness: 200 `{"status":"ok","engine_loaded":true,…}` when ready; **503** while loading or if loading failed |
| GET | `/info` | Project metadata derived from the artefacts: languages, service/agency counts, AR/EN coverage, chunks, capture dates, model, weights, thresholds, response types |
| POST | `/ask` | **Main endpoint**: question → structured, cited answer |
| GET | `/services` | Browse services: `language` (en/ar), `agency` (code or name), `search` (title substring, AR/EN), `limit` (1–100, default 20), `offset` (≥0) |
| GET | `/services/{service_id}` | Full official record (both languages) + provenance; 404 if unknown |
| GET | `/agencies` | Agencies with service counts and AR/EN coverage |
| GET | `/stats` | Corpus statistics + held-out evaluation summary read from `data/evaluation/results_v2.json` |

### POST /ask

Request:

```json
{ "question": "How can I renew a commercial registration?", "language": "auto" }
```

- `question`: 1–400 characters after trimming (same limit as the app). Empty or whitespace → **422**.
- `language`: `"auto"` (default; detected from the question, exactly as in the app), `"en"` or `"ar"` (forces the answer's language/labels; official text is still never machine-translated). Anything else → **422**.

Response (fields always present; lists may be empty):

```jsonc
{
  "question": "…",
  "detected_language": "en",            // from the question
  "answer_language": "en",              // language of labels / chosen official text
  "response_type": "answer",            // answer | possible_match | related_services | unsupported
  "engine_status": "answered",          // raw engine status: answered | tentative | insufficient
  "message": null,                      // text for related_services / unsupported
  "lead": "The official service that matches your question is …",   // Dalil's templated sentence
  "intent": "fees",                     // general | fees | time | documents | requirements | steps
  "sections": [                         // only sections the official source states
    { "key": "direct", "title": "Direct answer", "ordered": false,
      "points": [ { "text": "<verbatim official text>", "citation": 1, "label": null } ] },
    { "key": "steps", "title": "Steps", "ordered": true, "points": [ … ] }
    // keys: direct | need | docs | steps | fees | who | notes
  ],
  "sources": [ { "citation": 1, "service_id": "mc-1", "title": "…", "agency": "…",
                 "url": "https://…gov.sa/…", "alternate_language_url": "https://…",
                 "language": "en", "captured_at": "2026-09-30T…", "source_last_modified": "16 Sep 2026",
                 "score": 0.4404 } ],
  "related_services": [],               // ONLY for related_services — links, not an answer
  "notes": [],                          // e.g. "official text only available in Arabic"
  "evidence": [ { "citation": 1, "section": "steps", "text": "…", "score": 0.41, "language": "en" } ],
  "top_score": 0.4404,
  "thresholds": { "answer": 0.3906, "possible_match": 0.2852, "related": 0.1999 },
  "latency_ms": 81.2,
  "disclaimer": "Dalil AI is an independent educational project …"
}
```

How the frontend should treat `response_type`:

| Value | Meaning | Suggested UI |
|---|---|---|
| `answer` | confident match (score ≥ answer threshold) | Show sections + sources |
| `possible_match` | best match shown with a warning | Show a "possible match" badge + the `lead` (it already says it may not be exact) |
| `related_services` | no answer; closest official services | Show `message` + `related_services` as links only |
| `unsupported` | outside Dalil's sources | Show `message` |

Every `points[].citation` refers to a `sources[].citation`. Point text is verbatim official text; `label` and `lead` are Dalil's wording.

### Errors

| Status | When | Body |
|---|---|---|
| 422 | invalid body or query parameters | `{"detail": [ …pydantic errors… ]}` |
| 404 | unknown `service_id` | `{"detail": "Service not found"}` |
| 503 | engine still loading or failed to load (`/ask`, `/health`) | `{"detail": …}` / health body |
| 500 | unexpected error | `{"detail": "Internal server error"}` (no stack traces or paths) |

All responses are `application/json; charset=utf-8` and Arabic is returned as real UTF-8 characters, not `\u` escapes.

## Examples

```bash
curl -s -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
     -d '{"question":"كيف يمكنني تجديد السجل التجاري؟"}'
curl -s "http://127.0.0.1:8000/services?agency=zatca&limit=5&language=ar"
curl -s http://127.0.0.1:8000/services/mc-1
```

## Behaviour notes and known limitations

- `/ask` calls are serialised (one at a time per process). A warm answer takes about 80–100 ms on a 2-core CPU, and a repeated question about 25 ms. Scale horizontally (more replicas) rather than with threads.
- Memory is about **1.3 GB RSS** per process (PyTorch + model + index). Use an instance with ≥ 2 GB.
- Answers inherit the V2 engine's known limitations (README → *Limitations*): Interior/Absher topics (iqama, traffic fines, in-Kingdom passports) are not in the corpus, sibling services are the main error (e.g. "renew a commercial registration" → "renewal of Commercial **Agency**", while the corpus has the "annual confirmation of commercial registry data" services instead), and the corpus is a snapshot from 30 Sep – 1 Oct 2026.
- No authentication or rate limiting yet. Add rate limiting at the edge (or a small middleware) before heavy public use.


## V3 additions (API 3.0.0)
- `POST /ask` accepts optional `context_service_id` (the previous answer's `sources[0].service_id`) for
  follow-up questions such as "What documents do I need?" / «كم الرسوم؟». Stateless; validated against
  `^[a-z]{2,6}-[0-9a-z]{1,16}$`. The response echoes the context actually applied (`null` if the question
  was treated as new) and `runtime` (`lite` | `full`).
- `/`, `/health`, `/info` and `/stats` report `runtime`; in lite mode `/stats.evaluation` comes from
  `results_v3.json` and includes the V2 reference numbers. All other fields are unchanged.
