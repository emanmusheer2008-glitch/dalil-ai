# Deployment

**Status:** not deployed (by design; localhost until the owner approves). Two deployable surfaces share the same engine:

| Surface | Entry point | Host |
|---|---|---|
| **HTTP API** for the future React frontend | `api.main:app` (FastAPI) | **Railway** (`railway.json`) |
| Streamlit app | `app.py` | Streamlit Community Cloud |

## Render Free (HTTP API, V3 lite) — current public deployment

`render.yaml` (Blueprint): free web service, `pip install -r requirements-prod.txt` (no PyTorch),
`python -m uvicorn api.main:app --host 0.0.0.0 --port $PORT`, health check `/health`,
`DALIL_RUNTIME=lite`. Peak RSS measured ~315 MB (free limit 512 MB). Free instances sleep after
15 min idle; the first request after that takes ~30–60 s (cold start + ~3 s index load).

Steps: Render → *New → Blueprint* (or *Web Service*) → connect the GitHub repo → Instance type **Free**
→ deploy → open `https://<service>.onrender.com/docs`.

## Railway (V2 full runtime — not used; needs >1 GB RAM)


`railway.json` (repo root):

```json
{
  "deploy": {
    "startCommand": "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 300,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 5
  }
}
```

- **Build:** Railway's default Python builder installs `requirements.txt` (CPU-only PyTorch via `--extra-index-url`, sentence-transformers, FastAPI, uvicorn, …). `.python-version` pins Python **3.11**, the version the tests and the API were verified on.
- **Start:** POSIX command, binds `0.0.0.0` on Railway's `$PORT`. No Windows paths anywhere: every file path resolves from the repository root (`src/config.py`).
- **Health check:** `/health` returns 503 while the model loads and 200 once `/ask` can answer, so traffic is only routed to a ready instance. The 300 s timeout covers the first start, which also downloads the ~470 MB model.
- **Data at runtime:** `data/processed/` and `data/evaluation/` from the repo only. Raw captures are not needed. `lexical_index.pkl` is rebuilt on first start (~3 s) and written next to the data.
- **Variables to set in Railway:** usually none. Optionally `DALIL_CORS_ORIGINS=https://<your-frontend-domain>` once the frontend URL is known (default allows any origin), and `HF_HOME=/data/hf` with a Railway volume mounted at `/data` to keep the model between deploys.
- **Resources:** about 1.3 GB RAM per process (measured). Choose a plan or instance with ≥ 2 GB. Expect a 5–15 s engine load after each deploy or restart (longer on the very first start because of the model download).
- **Steps when approved:** push to GitHub → Railway *New project → Deploy from GitHub repo* → select the repo (it picks up `railway.json`) → wait for the health check → *Settings → Networking → Generate domain* → open `https://<domain>/docs`. About 15 minutes.

See `docs/API.md` for endpoints and response formats.

---

## Streamlit Community Cloud (Streamlit app)

Why: the app is a Streamlit app, the host is free, it builds directly from a GitHub repository, and it gives an HTTPS URL. Alternatives (Hugging Face Spaces with the Streamlit SDK, Render's free tier) would also work. They offer no advantage here, and Render's free instances sleep and have less memory.

### Readiness checklist (verified in this pass)

| Check | Status |
|---|---|
| Linux compatible | ✅ the final evaluation, tests and app all ran on Linux (Python 3.11) |
| Relative paths only | ✅ `src/config.py` resolves everything from the repo root. No Windows paths in code (checked with grep) |
| Requirements | ✅ `requirements.txt` (CPU-only PyTorch via `--extra-index-url`), `requirements-dev.txt` adds pytest |
| Prebuilt artefacts in repo | ✅ `data/processed/` (≈25 MB: knowledge base, chunks, embeddings, configs, reports) and `data/evaluation/` |
| Raw data needed at runtime? | ❌ no. The app never reads `data/raw/` |
| Model | downloaded from Hugging Face on first start (~470 MB), then cached by the host |
| Lexical cache | `lexical_index.pkl` is not in git; it is rebuilt automatically on first start (~3 s) |
| Secrets / API keys | none needed, none present (audited) |
| Memory | model + index use roughly 1–1.5 GB of RAM. Check the host's current free-tier limit before deploying |
| Cold start on a fresh server | ~5 s in-process after the model is downloaded; the very first start also downloads the model |

### Steps (the only human actions)

1. **Decide the data question below** (public vs private repository).
2. Push the repository to GitHub (`git push`; it's committed and ready).
3. Go to share.streamlit.io → *Create app* → choose the repo, branch `main`, main file `app.py`. In *Advanced settings* choose Python 3.11.
4. Wait for the build (installing PyTorch takes a few minutes), open the URL, and run the 10 manual acceptance questions in `docs/MANUAL_ACCEPTANCE_TESTS.md`.

Estimated human time: **15–20 minutes**.

## Data licence decision (needed before a *public* repository)

The processed knowledge base contains **verbatim text from official government pages** (that is the point: Dalil must not paraphrase official facts). The raw HTML captures are already excluded from git.

| Option | What to do | Trade-off |
|---|---|---|
| **A. Private repo** (simplest now) | Keep the GitHub repo private and deploy from it | No licence decision needed yet; the code isn't publicly visible to reviewers, so share access or screenshots |
| **B. Public repo with processed data** | First read each agency's website terms of use (ZATCA, MoMAH, MoJ, HRSD, MC, MoE, MOFA, SFDA, CHI, Hajj) and confirm reuse with attribution is allowed | Fully reproducible public demo |
| **C. Public code, private data** | Make the repo public without `data/processed/` and load the data from a private location at runtime | More engineering; not implemented |

Note: the V1 commit already contains the Ministry of Commerce raw capture (`data/raw/official/mc/…`). From V2 on it is untracked, but it remains in git **history**. If you choose a public repository and the MC terms don't allow redistribution, publish from a fresh repository (or rewrite history) instead of pushing this history.

## Development vs deployment

| | Development (laptop) | Deployed |
|---|---|---|
| Data | raw captures + rebuild scripts | only `data/processed/` |
| Index build | `python -m src.indexing.build_index` | never; reads prebuilt files |
| Model | cached in `~/.cache/huggingface` | downloaded once per container |
| Cold start | ~5 s | ~5 s after the first download; repeats after each redeploy or sleep |
| CPU | your laptop | shared CPU; expect somewhat slower answers |
