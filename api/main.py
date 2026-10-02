"""Dalil AI — production HTTP API (FastAPI).

    python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}

Runtimes (``DALIL_RUNTIME``)
---------------------------
* ``lite`` (default, V3): ``src.lite.engine.DalilLite`` -- numpy/scikit-learn only, no PyTorch, no
  sentence-transformers, no model download. Peak memory ~0.3 GB, fits free hosting tiers.
* ``full`` (V2 research baseline): ``src.engine.Dalil`` with the multilingual transformer
  (needs ``requirements.txt``; ~1.5 GB peak).

Both use the same corpus, the same scoring code and the same answer synthesizer; each has its own
thresholds calibrated on the same benchmark (see docs/V3_LITE.md).

Architecture
------------
    client (React / curl)  ->  this API  ->  DalilLite | Dalil
                                              -> Retriever (dense signal + char n-grams + BM25 + title coverage)
                                              -> synthesize (answer / possible match / related / decline)

Loading
-------
* The knowledge base (``data/processed/services.jsonl``, ~0.1 s) is loaded at startup, so the
  browsing endpoints (/services, /agencies, /info, /stats) work immediately.
* The expensive part -- embeddings / static vectors, lexical index (and the model in ``full``) -- is loaded ONCE per process in a
  background thread started at startup (set ``DALIL_EAGER_LOAD=0`` to load on the first /ask instead).
  ``/health`` returns 503 ``"loading"`` until it is ready, so a platform health check only
  routes traffic once the engine can answer. Every /ask reuses the same engine object.
* /ask calls are serialised with a lock: the engine keeps small in-process caches and the work is
  CPU-bound, so parallel calls would not be faster on a small instance.

Nothing here changes retrieval, thresholds, the corpus or the answering policy.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    MAX_QUESTION_CHARS, Agency, AnswerPoint, AnswerSection, AskRequest, AskResponse, ErrorResponse, Evidence,
    ServiceDetail, ServiceLanguageBlock, ServiceList, ServiceSummary, Source, Thresholds,
)
from src import config
from src.answering.synthesizer import query_language
from src.ingestion.pipeline import load_services
from src.ui import T as UI_TEXT

API_VERSION = "4.0.0"
RUNTIMES = ("v4", "lite", "full")


def selected_runtime() -> str:
    rt = os.environ.get("DALIL_RUNTIME", "v4").strip().lower() or "v4"
    return rt if rt in RUNTIMES else "v4"



class UTF8JSONResponse(JSONResponse):
    """JSON is UTF-8 by definition; say so explicitly for clients that default to Latin-1."""
    media_type = "application/json; charset=utf-8"
log = logging.getLogger("dalil.api")

LANG_FIELDS = ("title", "agency", "category", "description", "eligibility", "requirements", "required_documents",
               "steps", "fees", "processing_time", "target_audience", "service_languages", "notes", "official_url",
               "source_last_modified")


# ============================================================ engine state ==
class EngineState:
    """Holds the knowledge base and the (lazily/background-loaded) Dalil engine for this process."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.runtime = selected_runtime()
        self.records: dict = {}
        self.engine = None
        self.status = "not_loaded"          # not_loaded | loading | ready | error
        self.error: str | None = None
        self.load_seconds: float | None = None
        self.started_at = time.time()
        self.asks = 0
        self._load_lock = threading.Lock()
        self._ask_lock = threading.Lock()
        self._ready = threading.Event()

    def load_records(self) -> None:
        self.records = load_services()

    def load_engine(self) -> None:
        with self._load_lock:
            if self.status in ("ready", "loading") and threading.current_thread().name != "dalil-loader":
                return
            if self.status == "ready":
                return
            self.status = "loading"
            t0 = time.perf_counter()
            try:
                if self.runtime == "full":
                    from src.engine import Dalil  # heavy imports (torch, sentence-transformers) happen here
                    d = Dalil(records=self.records or None)
                elif self.runtime == "lite":
                    from src.lite.engine import DalilLite      # numpy / scikit-learn only
                    d = DalilLite(records=self.records or None)
                else:
                    from src.lite.engine import DalilLite
                    from src.v4.engine import DalilV4          # + stdlib HTTP client for Gemini
                    d = DalilV4(DalilLite(records=self.records or None))
                d.warm_up()
                self.engine = d
                self.records = d.records
                self.load_seconds = round(time.perf_counter() - t0, 2)
                self.status = "ready"
                log.info("Dalil engine (%s) ready in %.1f s", self.runtime, self.load_seconds)
            except Exception as e:  # keep the API up; /health reports the failure
                self.status = "error"
                self.error = type(e).__name__
                log.exception("Dalil engine failed to load")
            finally:
                self._ready.set()

    def start_background_load(self) -> None:
        self.status = "loading"
        threading.Thread(target=self.load_engine, name="dalil-loader", daemon=True).start()

    def wait_ready(self, timeout: float) -> bool:
        if self.status == "not_loaded":
            self.load_engine()
        self._ready.wait(timeout)
        return self.status == "ready"

    def ask(self, question: str, ui_lang: str | None, context_service_id: str | None = None,
            conversation: list | None = None, use_ai: bool | None = None):
        if self.runtime == "v4":          # Gemini is a network call: don't hold the lock for it
            with self._ask_lock:
                self.asks += 1
            return self.engine.ask(question, ui_lang=ui_lang, context_service_id=context_service_id,
                                   conversation=conversation, use_ai=use_ai)
        with self._ask_lock:
            self.asks += 1
            if self.runtime == "full":    # V2 engine is single-turn; context is ignored
                return self.engine.ask(question, ui_lang=ui_lang)
            return self.engine.ask(question, ui_lang=ui_lang, context_service_id=context_service_id)


STATE = EngineState()


def _cors_origins() -> list[str]:
    raw = os.environ.get("DALIL_CORS_ORIGINS", "").strip()
    if not raw or raw == "*":
        return ["*"]
    return [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.reset()                         # picks up DALIL_RUNTIME; fresh state per app start
    STATE.load_records()
    if os.environ.get("DALIL_EAGER_LOAD", "1").strip().lower() not in ("0", "false", "no"):
        STATE.start_background_load()
    yield


app = FastAPI(
    title="Dalil AI API",
    version=API_VERSION,
    description="Bilingual (Arabic–English) retrieval-grounded assistant for official Saudi public-service "
                "information. Answers are built only from verbatim official text with citations. "
                "Independent educational project — not an official government service.",
    lifespan=lifespan,
    default_response_class=UTF8JSONResponse,
    responses={500: {"model": ErrorResponse}},
)

_origins = _cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,              # no cookies/auth: keeps "*" valid and safe
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)


@app.exception_handler(RequestValidationError)
async def _validation(request: Request, exc: RequestValidationError):
    return UTF8JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})


@app.exception_handler(StarletteHTTPException)
async def _http(request: Request, exc: StarletteHTTPException):
    return UTF8JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):   # never leak stack traces or paths
    log.exception("Unhandled error on %s", request.url.path)
    return UTF8JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ================================================================ helpers ==
def _agency_code(service_id: str) -> str:
    return service_id.split("-", 1)[0]


def _read_json(path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _lang_block(rec, lang: str) -> ServiceLanguageBlock | None:
    vals = {f: rec.get(f, lang) for f in LANG_FIELDS}
    if not vals.get("title"):
        return None
    return ServiceLanguageBlock(**vals)


def _source(c) -> Source:
    return Source(citation=c.n, service_id=c.service_id, title=c.title, agency=c.agency, url=c.url,
                  alternate_language_url=c.other_url, language=c.lang, captured_at=c.date_collected,
                  source_last_modified=c.source_last_modified, score=round(float(c.score), 4))


def _response_type(ans) -> str:
    if ans.status == "answered":
        return "answer"
    if ans.status == "tentative":
        return "possible_match"
    return "related_services" if ans.related else "unsupported"


def _norm(s: str | None) -> str:
    from src.utils.arabic import normalize_for_matching
    return normalize_for_matching(s or "")


def _ai_info() -> dict | None:
    eng = STATE.engine
    if STATE.runtime != "v4":
        return None
    from src.v4.gemini import api_key, model_name   # never returns or logs the key itself
    return {"provider": "Google Gemini API", "model": model_name(), "configured": api_key() is not None,
            "enabled": bool(eng and eng.ai_available),
            "role": "language only: understands the question and writes answers from cited official evidence; "
                    "never a factual source. Falls back to verbatim official text if unavailable."}


# ============================================================== endpoints ==
@app.get("/", tags=["meta"])
def root() -> dict[str, Any]:
    return {"name": "Dalil AI API", "version": API_VERSION, "runtime": STATE.runtime, "status": "ok",
            "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["meta"], responses={503: {"description": "Engine loading or failed"}})
def health():
    body = {
        "status": "ok" if STATE.status == "ready" else STATE.status,
        "engine_loaded": STATE.status == "ready",
        "engine_status": STATE.status,
        "knowledge_base_loaded": bool(STATE.records),
        "services": len(STATE.records),
        "version": API_VERSION,
        "runtime": STATE.runtime,
        "engine_load_seconds": STATE.load_seconds,
        "uptime_seconds": round(time.time() - STATE.started_at, 1),
    }
    if STATE.status == "error":
        body["error"] = STATE.error
    # lazy mode (DALIL_EAGER_LOAD=0): the process is healthy even before the first /ask
    ok = STATE.status == "ready" or (STATE.status == "not_loaded" and bool(STATE.records))
    return UTF8JSONResponse(status_code=200 if ok else 503, content=body)


@app.get("/info", tags=["meta"])
def info() -> dict[str, Any]:
    recs = STATE.records
    meta = _read_json(config.INDEX_META_JSON) or {}
    rep = _read_json(config.BUILD_REPORT_JSON) or {}
    lite = STATE.runtime in ("lite", "v4")
    cfg = _read_json(config.PROCESSED_DIR / "retrieval_config_v3.json" if lite else config.RETRIEVAL_CONFIG_V2_JSON) or {}
    static = (_read_json(config.PROCESSED_DIR / "static_vocab.json") or {}) if lite else {}
    rng = rep.get("date_collected_range") or [None, None]
    if lite:
        retrieval = {
            "runtime": "lite (V3)",
            "transformer_loaded_at_runtime": False,
            "method": "hybrid: static word embeddings for the query (distilled offline from the multilingual "
                      "model) against the precomputed document embeddings, + character n-gram TF-IDF + BM25 + "
                      "title coverage, with bilingual query expansion (lexical signals)",
            "distilled_from": meta.get("model_name"),
            "static_vocabulary_words": len(static.get("words") or []) or None,
            "embedding_dim": meta.get("dim"),
            "weights": cfg.get("params"),
            "thresholds": {"answer": cfg.get("t_answer"), "possible_match": cfg.get("t_tentative"),
                           "related": cfg.get("t_related")},
            "follow_up_context": "stateless: send context_service_id (and, for v4, conversation_context) with /ask",
        }
        if STATE.runtime == "v4":
            v4cfg = _read_json(config.PROCESSED_DIR / "v4_config.json") or {}
            retrieval["runtime"] = "v4 (V3 lite retrieval + action-aware rerank + optional grounded AI layer)"
            retrieval["action_rerank"] = {"bonus": v4cfg.get("action_bonus"), "penalty": v4cfg.get("action_penalty")}
    else:
        retrieval = {
            "runtime": "full (V2)",
            "transformer_loaded_at_runtime": True,
            "embedding_model": meta.get("model_name"),
            "embedding_dim": meta.get("dim"),
            "method": "hybrid: multilingual embeddings + character n-gram TF-IDF + title coverage, "
                      "with bilingual query expansion (lexical signals)",
            "weights": cfg.get("params"),
            "thresholds": {"answer": cfg.get("t_answer"), "possible_match": cfg.get("t_tentative"),
                           "related": cfg.get("t_related")},
        }
    return {
        "name": "Dalil AI",
        "version": API_VERSION,
        "runtime": STATE.runtime,
        "description": "Bilingual retrieval-grounded assistant for official Saudi public-service information.",
        "supported_languages": ["ar", "en"],
        "question_max_chars": MAX_QUESTION_CHARS,
        "corpus": {
            "services": len(recs),
            "agencies": len({_agency_code(s) for s in recs}),
            "services_with_arabic_text": sum(bool(r.title_ar) for r in recs.values()),
            "services_with_english_text": sum(bool(r.title_en) for r in recs.values()),
            "evidence_chunks": meta.get("n_chunks"),
            "captured_from": (rng[0] or "")[:10] or None,
            "captured_to": (rng[1] or "")[:10] or None,
            "source_domains": rep.get("source_domains"),
        },
        "retrieval": retrieval,
        "response_types": {
            "answer": "confident match; sections of verbatim official text with citations",
            "possible_match": "best match shown with a warning that it may not be exact",
            "related_services": "no answer; closest official services returned as links only",
            "unsupported": "outside Dalil's official sources",
        },
        "generative_model": _ai_info(),
        "disclaimer": UI_TEXT["en"]["disclaimer"],
    }


@app.post("/ask", response_model=AskResponse, tags=["ask"],
          responses={422: {"description": "Empty, too long or invalid request"},
                     503: {"model": ErrorResponse, "description": "Engine still loading or unavailable"}})
def ask(req: AskRequest) -> AskResponse:
    if not STATE.wait_ready(timeout=float(os.environ.get("DALIL_ASK_WAIT_SECONDS", "120"))):
        raise HTTPException(status_code=503, detail="Dalil engine is not ready yet. Please retry shortly.")
    t0 = time.perf_counter()
    ui_lang = None if req.language == "auto" else req.language
    conv = [t.model_dump() for t in req.conversation_context or []]
    res = STATE.ask(req.question, ui_lang, req.context_service_id, conv, req.use_ai)
    ms = (time.perf_counter() - t0) * 1000
    v4 = res if STATE.runtime == "v4" else None
    ans = v4.base if v4 else res
    th = ans.thresholds or {}
    ai_text = bool(v4 and v4.text_origin != "official_verbatim")
    sections = v4.sections if ai_text else ans.sections
    sources = v4.sources if ai_text else ans.citations
    evidence = v4.evidence if ai_text else ans.evidence
    related = v4.related if v4 else ans.related
    if v4:
        rtype = {"grounded_answer": "answer", "partial_answer": "answer", "possible_match": "possible_match",
                 "related_services": "related_services"}.get(v4.response_mode,
                                                             "related_services" if related else "unsupported")
    else:
        rtype = _response_type(ans)
    return AskResponse(
        question=req.question,
        detected_language=query_language(req.question),
        answer_language=ans.ui_lang,
        response_type=rtype,
        engine_status=ans.status,
        message=ans.message,
        lead=ans.lead,
        intent=ans.intent,
        sections=[AnswerSection(key=s.key, title=s.title, ordered=s.ordered,
                                points=[AnswerPoint(text=p.text, citation=p.cite, label=p.label) for p in s.points])
                  for s in sections],
        sources=[_source(c) for c in sources],
        related_services=[_source(c) for c in related],
        notes=list(ans.notes) if not ai_text else [],
        evidence=[Evidence(citation=e[0], section=e[1], text=e[2], score=round(float(e[3]), 4), language=e[4])
                  for e in evidence],
        top_score=None if ans.top_score is None else round(float(ans.top_score), 4),
        thresholds=Thresholds(answer=th.get("answer"), possible_match=th.get("tentative"), related=th.get("related")),
        latency_ms=round(ms, 1),
        context_service_id=(v4.context_service_id if v4 else getattr(ans, "context_service_id", None)),
        runtime=STATE.runtime,
        response_mode=v4.response_mode if v4 else None,
        answer=v4.answer if v4 else None,
        text_origin=v4.text_origin if v4 else "official_verbatim",
        ai_enabled=bool(v4 and v4.ai_enabled),
        ai_used=bool(v4 and v4.ai_used),
        ai_model=(STATE.engine.client.model if v4 and v4.ai_enabled else None),
        ai_fallback_reason=v4.ai_error if v4 else None,
        verified_fields=v4.verified_fields if v4 else [],
        unverified_information=v4.unverified if v4 else [],
        follow_up_suggestions=v4.follow_ups if v4 else [],
        search_queries=v4.search_queries if v4 else [],
        redactions_applied=v4.redactions if v4 else 0,
        disclaimer=UI_TEXT[ans.ui_lang]["disclaimer"],
    )


@app.get("/services", response_model=ServiceList, tags=["services"])
def list_services(
    language: str = Query("en", pattern="^(en|ar)$", description="Language of titles/descriptions"),
    agency: str | None = Query(None, max_length=60, description="Agency code (e.g. 'zatca', 'mc') or name"),
    search: str | None = Query(None, max_length=MAX_QUESTION_CHARS, description="Substring in the title (AR/EN)"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> ServiceList:
    items = []
    a_norm = _norm(agency) if agency else None
    q_norm = _norm(search) if search else None
    for sid, r in sorted(STATE.records.items()):
        code = _agency_code(sid)
        if a_norm and a_norm not in (code, _norm(r.agency_en), _norm(r.agency_ar)):
            continue
        if q_norm and q_norm not in _norm(r.title_en) and q_norm not in _norm(r.title_ar):
            continue
        items.append((sid, r, code))
    total = len(items)
    out = []
    for sid, r, code in items[offset:offset + limit]:
        lang = language if r.get("title", language) else ("ar" if language == "en" else "en")
        desc = r.get("description", lang)
        out.append(ServiceSummary(
            service_id=sid, agency_code=code, title=r.get("title", lang), agency=r.get("agency", lang),
            official_url=r.get("official_url", lang),
            languages=[x for x in ("en", "ar") if r.get("title", x)],
            description=(desc[:300] + "…") if desc and len(desc) > 300 else desc,
        ))
    return ServiceList(total=total, limit=limit, offset=offset, items=out)


_SID = re.compile(r"^[a-z]{2,6}-[0-9a-z]{1,16}$")


@app.get("/services/{service_id}", response_model=ServiceDetail, tags=["services"],
         responses={404: {"model": ErrorResponse}})
def service_detail(service_id: str = Path(..., max_length=32)) -> ServiceDetail:
    rec = STATE.records.get(service_id) if _SID.match(service_id) else None
    if rec is None:
        raise HTTPException(status_code=404, detail="Service not found")
    return ServiceDetail(
        service_id=service_id, agency_code=_agency_code(service_id), source_domain=rec.source_domain,
        verification_status=rec.verification_status, captured_at=rec.date_collected,
        last_verified=rec.last_verified, contact_information=rec.contact_information,
        service_channel=rec.service_channel, source_sha256=rec.source_sha256,
        en=_lang_block(rec, "en"), ar=_lang_block(rec, "ar"),
    )


@app.get("/agencies", response_model=list[Agency], tags=["services"])
def agencies() -> list[Agency]:
    groups: dict[str, list] = defaultdict(list)
    for sid, r in STATE.records.items():
        groups[_agency_code(sid)].append(r)
    out = []
    for code, rs in groups.items():
        name_en = Counter(r.agency_en for r in rs if r.agency_en).most_common(1)
        name_ar = Counter(r.agency_ar for r in rs if r.agency_ar).most_common(1)
        out.append(Agency(code=code, name_en=name_en[0][0] if name_en else None,
                          name_ar=name_ar[0][0] if name_ar else None, domain=rs[0].source_domain,
                          services=len(rs), services_with_arabic=sum(bool(r.title_ar) for r in rs),
                          services_with_english=sum(bool(r.title_en) for r in rs)))
    return sorted(out, key=lambda a: -a.services)


@app.get("/stats", tags=["meta"])
def stats() -> dict[str, Any]:
    rep = _read_json(config.BUILD_REPORT_JSON) or {}
    res = _read_json(config.EVAL_DIR / "results_v2.json") or {}
    v2 = res.get("v2_test") or {}
    base = res.get("v1_setting_on_v2_corpus_test") or {}
    out: dict[str, Any] = {
        "corpus": {
            "services_indexed": len(STATE.records),
            "agencies": len({_agency_code(s) for s in STATE.records}),
            "services_quarantined": rep.get("services_quarantined"),
            "duplicates_removed": rep.get("duplicates_removed"),
            "with_arabic": rep.get("with_arabic"),
            "with_english": rep.get("with_english"),
            "knowledge_base_built_at": rep.get("built_at"),
        },
        "api": {"questions_answered_since_start": STATE.asks, "engine_status": STATE.status,
                "runtime": STATE.runtime},
    }
    r3 = _read_json(config.EVAL_DIR / "results_v3.json") or {}
    r4 = _read_json(config.EVAL_DIR / "results_v4.json") or {}
    if STATE.runtime == "v4" and r4:
        out["evaluation"] = {
            "runtime_evaluated": "v4 deterministic retrieval (V3 + action rerank); AI layer evaluated separately",
            "source": "data/evaluation/results_v4.json (same benchmark and held-out test split as V2/V3)",
            "generated_at": r4.get("generated_at"),
            "selected_action_rerank": r4.get("selected"),
            "test_retrieval": (r4.get("v4") or {}).get("test"),
            "test_retrieval_by_language": (r4.get("v4") or {}).get("test_by_language"),
            "test_decision": (r4.get("v4") or {}).get("test_decision"),
            "v3_reference_test": {"retrieval": (r4.get("v3") or {}).get("test"),
                                  "decision": (r4.get("v3") or {}).get("test_decision")},
            "ai_evaluation": _read_json(config.EVAL_DIR / "results_v4_ai.json"),
        }
        return out
    if STATE.runtime in ("lite", "v4") and r3:
        v3 = r3.get("v3_test") or {}
        out["evaluation"] = {
            "runtime_evaluated": "lite (V3) -- the runtime serving this API",
            "source": "data/evaluation/results_v3.json (same benchmark and held-out test split as V2)",
            "generated_at": r3.get("generated_at"),
            "protocol": r3.get("protocol"),
            "selected_configuration": (r3.get("selected") or {}).get("name"),
            "test_retrieval": v3.get("retrieval"),
            "test_retrieval_by_language": v3.get("retrieval_by_language"),
            "test_decision": v3.get("decision"),
            "test_retrieval_fresh_families": v3.get("retrieval_fresh_families"),
            "related_links_test": v3.get("related_links"),
            "v2_full_runtime_reference_test": r3.get("v2_reference_test"),
        }
        return out
    if res:
        out["evaluation"] = {
            "source": "data/evaluation/results_v2.json (held-out test split)",
            "generated_at": res.get("generated_at"),
            "benchmark": res.get("benchmark"),
            "selected_configuration": (res.get("selected") or {}).get("name"),
            "test_retrieval": v2.get("retrieval"),
            "test_retrieval_by_language": v2.get("retrieval_by_language"),
            "test_decision": v2.get("decision"),
            "v1_method_same_corpus_test_retrieval": base.get("retrieval"),
            "related_links_test": res.get("related_links_test"),
            "paraphrase_robustness_test": (res.get("paraphrase_robustness_test") or {}).get("v2"),
            "leakage_checks_test": {
                "fresh_families": v2.get("retrieval_fresh_families"),
                "lexicon_not_fired": v2.get("retrieval_lexicon_not_fired"),
            },
            "measured_latency": res.get("latency"),
        }
    return out
