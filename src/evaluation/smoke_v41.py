"""V4.1 coverage smoke test with the REAL Gemini API (5 questions, at most ~10 calls).

    PowerShell:  $env:GEMINI_API_KEY = ...;  python -m src.evaluation.smoke_v41

Writes data/evaluation/results_v41_smoke.json. Never writes the key.
"""
import json
import time
from datetime import datetime, timezone

from src import config
from src.utils.arabic import arabic_ratio
from src.v4.engine import DalilV4
from src.v4.gemini import GeminiClient, api_key

QUESTIONS = [("How do I renew my iqama?", "en"), ("How can I check my traffic violations?", "en"),
             ("How do I renew my driving licence?", "en"), ("I lost my passport. What should I do?", "en"),
             ("كيف أجدد الإقامة؟", "ar")]

if __name__ == "__main__":
    if not api_key():
        raise SystemExit("GEMINI_API_KEY is not set in this shell.")
    c = GeminiClient()
    d = DalilV4(client=c)
    rows = []
    for q, lang in QUESTIONS:
        a = d.ask(q)
        text = " ".join([a.answer or ""] + [p.text for s in a.sections for p in s.points])
        rows.append({"question": q, "mode": a.response_mode, "ai_used": a.ai_used, "ai_error": a.ai_error,
                     "text_origin": a.text_origin, "calls": a.gemini_calls, "points_rejected": a.points_rejected,
                     "language_ok": (not text) or ((arabic_ratio(text) >= 0.3) == (lang == "ar")),
                     "urls_from_records": all(s.url == d.records[s.service_id].get("official_url", s.lang)
                                              for s in a.sources + a.related),
                     "sources": [f"{s.service_id} | {s.title} | {s.url}" for s in a.sources],
                     "related": [f"{s.service_id} | {s.title}" for s in a.related][:3],
                     "answer": a.answer, "sections": {s.key: [p.text for p in s.points] for s in a.sections},
                     "unverified": a.unverified})
        time.sleep(4)
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": c.model,
           "gemini_calls": c.calls, "tokens": c.tokens, "samples": rows}
    (config.EVAL_DIR / "results_v41_smoke.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    for r in rows:
        print(f"{r['mode']:16s} ai={r['text_origin'][:6]} err={r['ai_error']} lang_ok={r['language_ok']} "
              f"urls_ok={r['urls_from_records']} | {r['question']} -> {r['sources'][:1] or r['related'][:1]}")
    print("gemini calls:", c.calls)
