"""DalilLite: the V3 public-runtime engine (no PyTorch, no sentence-transformers).

Same interface as ``src.engine.Dalil`` (``records``, ``retriever``, ``cfg``, ``warm_up``,
``search``, ``ask``) so the API can switch runtimes with one setting. Answers are built by the
same V2 synthesizer from verbatim official text, with V3's own calibrated thresholds
(``data/processed/retrieval_config_v3.json`` from ``src/evaluation/evaluate_v3.py``).

Follow-up questions (stateless): the client may pass ``context_service_id`` -- the service of
the previous answer. When the new question looks like a follow-up ("what documents do I need?",
"how much does it cost?", «كم الرسوم؟») the service title is added to the query, so retrieval
stays on the same verified service unless the question clearly names something else. No
server-side session storage is needed.
"""
from __future__ import annotations

import json
import re

import numpy as np

from src import config
from src.answering.synthesizer import SynthAnswer, detect_intent, query_language, synthesize
from src.ingestion.pipeline import load_services
from src.lite.retriever import LiteRetriever
from src.retrieval.retriever import SearchResult

CONFIG_V3 = config.PROCESSED_DIR / "retrieval_config_v3.json"

_REFERENTIAL = re.compile(
    r"\b(it|this|that|these|those|the service|same|there)\b|"
    r"(هذه|هذا|ذلك|تلك|نفس|الخدمه|لها|له|فيها|عليها)", re.I)
_FOLLOWUP_MAX_WORDS = 9


def looks_like_followup(question: str) -> bool:
    """Short question that asks about an aspect (fees, documents, steps, time...) or refers back."""
    n_words = len(question.split())
    if n_words > _FOLLOWUP_MAX_WORDS:
        return False
    return detect_intent(question) != "general" or bool(_REFERENTIAL.search(question))


class DalilLite:
    runtime = "lite"

    def __init__(self, retriever: LiteRetriever | None = None, records: dict | None = None, cfg: dict | None = None):
        self.cfg = cfg or json.loads(CONFIG_V3.read_text(encoding="utf-8"))
        self.retriever = retriever or LiteRetriever.from_disk(use_static=self.cfg.get("variant") != "lexical")
        if self.cfg.get("variant") == "model_docs":
            # V2's precomputed transformer chunk vectors as the document side (a matrix, no model)
            self.retriever.embeddings = np.load(config.EMBEDDINGS_NPY).astype(np.float32)
        self.records = records if records is not None else load_services()

    @property
    def t_answer(self) -> float:
        return float(self.cfg["t_answer"])

    @property
    def t_tentative(self) -> float:
        return float(self.cfg["t_tentative"])

    @property
    def t_related(self) -> float | None:
        v = self.cfg.get("t_related")
        return None if v is None else float(v)

    def warm_up(self) -> None:
        self.retriever.signals("warm up")

    def search(self, query: str, top_k: int = 5) -> SearchResult:
        c = self.cfg
        return self.retriever.search(query, method=c["method"], alpha=c.get("alpha", 0.0),
                                     params=c.get("params"), top_k=top_k)

    def ask(self, query: str, ui_lang: str | None = None, context_service_id: str | None = None,
            **_ignored) -> SynthAnswer:
        lang = ui_lang or query_language(query)
        used_context = None
        q = query
        if context_service_id and context_service_id in self.records and looks_like_followup(query):
            rec = self.records[context_service_id]
            title = rec.get("title", lang) or rec.title_en or rec.title_ar or ""
            q = f"{query} {title}"
            used_context = context_service_id
        result = self.search(q)
        result.query = query                       # display/intent use the user's own words
        ans = synthesize(result, self.records, self.t_answer, self.t_tentative, t_related=self.t_related,
                         ui_lang=lang)
        ans.context_service_id = used_context      # type: ignore[attr-defined]
        return ans
