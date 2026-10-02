"""Dalil's single question-answering entry point (used by the app, the tests and the CLI).

    python -m src.engine "How can I reserve a trade name?"
    python -m src.engine --json "كيف أعادل شهادتي؟"

Pipeline for one question
-------------------------
1. ``Retriever.search``   -- hybrid retrieval with the *calibrated* settings in
   ``data/processed/retrieval_config_v2.json`` (chosen on dev+val, never on test).
2. ``synthesize``          -- three-way decision (answered / tentative / insufficient)
   with the calibrated thresholds, then a sectioned answer built only from verbatim
   official text, with [n] citations.

Keeping this in one place guarantees the app, the tests and the evaluation use the
same settings.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict

from src.answering.synthesizer import SynthAnswer, query_language, synthesize
from src.ingestion.pipeline import load_services
from src.retrieval import embedder
from src.retrieval.retriever import Retriever, SearchResult, load_retrieval_config


class Dalil:
    def __init__(self, retriever: Retriever | None = None, records: dict | None = None, cfg: dict | None = None):
        self.retriever = retriever or Retriever.from_disk()
        self.records = records if records is not None else load_services()
        self.cfg = cfg or load_retrieval_config()

    # thresholds: V2 config has t_answer/t_tentative; a V1-only config has a single threshold
    @property
    def t_answer(self) -> float:
        return float(self.cfg.get("t_answer", self.cfg.get("threshold") or 0.5))

    @property
    def t_tentative(self) -> float:
        return float(self.cfg.get("t_tentative", self.t_answer))

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

    def ask(self, query: str, rank_sentences: bool = True) -> SynthAnswer:
        result = self.search(query)
        kwargs = {}
        if rank_sentences and query.strip():
            kwargs = {"encode": lambda sents: embedder.encode(sents, self.retriever.model_path),
                      "query_vec": self.retriever._encode(query)}
        return synthesize(result, self.records, self.t_answer, self.t_tentative, t_related=self.t_related,
                          ui_lang=query_language(query), **kwargs)


def _print_answer(ans: SynthAnswer, ms: float) -> None:
    print(f"\nQ: {ans.query}")
    print(f"decision: {ans.status}   top score: {ans.top_score if ans.top_score is None else round(ans.top_score, 3)}"
          f"   thresholds: {ans.thresholds}   time: {ms:.0f} ms")
    if ans.status in ("insufficient", "empty"):
        print(ans.message)
        for c in ans.related:
            print(f"  related: {c.title} — {c.agency} — {c.url}")
        return
    if getattr(ans, "lead", None):
        print(ans.lead)
    for sec in getattr(ans, "sections", []) or []:
        title = getattr(sec, "title", "")
        print(f"\n## {title}")
        for i, pt in enumerate(sec.points[:8], 1):
            lab = f"{pt.label}: " if pt.label else ""
            print(f"  {str(i) + '.' if sec.ordered else '-'} {lab}{pt.text} [{pt.cite}]")
    for note in ans.notes:
        print(f"\nNote: {note}")
    print("\nSources:")
    for c in ans.citations:
        print(f"  [{c.n}] {c.title} — {c.agency} — {c.url}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ask Dalil a question from the command line.")
    ap.add_argument("question", nargs="+")
    ap.add_argument("--json", action="store_true", help="print the raw answer object")
    a = ap.parse_args(argv)
    d = Dalil()
    d.warm_up()
    q = " ".join(a.question)
    t0 = time.perf_counter()
    ans = d.ask(q)
    ms = (time.perf_counter() - t0) * 1000
    if a.json:
        print(json.dumps(asdict(ans), ensure_ascii=False, indent=2, default=str))
    else:
        _print_answer(ans, ms)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
