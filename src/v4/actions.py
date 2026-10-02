"""Action-verb understanding (EN/AR) and action-aware reranking -- deterministic, no LLM.

Many services share a subject ("commercial registration") and differ only by the action:
issue / renew / cancel / modify / transfer / inquire / replace / pay ... V3 sometimes ranks the
right subject with the wrong action (e.g. "get a CR" -> the *deletion* service). This module

* detects the action(s) a question asks for (strong cues, plus weak cues such as "register",
  "get", "apply", "pay" that only count when no strong cue is present), and
* detects the primary action of each service title (first action word in the title),

then nudges the candidate order: a matching action gets a small bonus; a *different* strong action
gets a small penalty. Weights are chosen on dev+val by ``src/evaluation/evaluate_v4.py``.
It never adds candidates and never raises confidence: the decision score stays the raw retrieval
score of whichever service ends up first.
"""
from __future__ import annotations

import re
from dataclasses import replace

from src.utils.arabic import normalize_for_matching

# Arabic patterns are written in normalize_for_matching form (no hamza variants, ة -> ه, ى -> ي).
STRONG = {
    "issue": r"\b(issu(e|es|ing|ance)|establish(ing|ed)?|incorporat\w*|creat(e|ing|ion)|set ?up|start(ing)? a)\b"
             r"|اصدار|اصدر|استخراج|استخرج|اطلع |اطلع$|طلع |تاسيس|اسس |انشاء|انشي",
    "reserve": r"\b(reserv\w*|book(ing)?)\b|حجز|احجز",
    "renew": r"\b(renew\w*|extend\w*|extension)\b|تجديد|اجدد|جدد|تمديد|امدد",
    "cancel": r"\b(cancel\w*|delet\w*|remov\w*|clos(e|ing|ure)|write[- ]?off|strike[- ]?off|terminat\w*|deregist\w*|"
              r"shut ?down)\b|الغاء|الغي|شطب|اشطب|حذف|احذف|اغلاق|اقفل|اقفال|ايقاف|اوقف",
    "modify": r"\b(modif\w*|updat\w*|amend\w*|chang(e|es|ing)|edit\w*|correct\w*)\b|تعديل|اعدل|عدل |تحديث|احدث|"
              r"تغيير|اغير|تصحيح",
    "transfer": r"\b(transfer\w*|assign\w*)\b|نقل|انقل|تنازل|التنازل",
    "inquire": r"\b(inquir\w*|enquir\w*|quer(y|ies)|check(ing)?|verif\w*|view(ing)?|track\w*|status)\b|استعلام|"
               r"استعلم|التحقق|تحقق|اتحقق|استعراض|الاطلاع|اطلاع|متابعه|اتابع",
    "replace": r"\b(replac\w*|lost|los(e|ing)|damaged|stolen|reissu\w*)\b|بدل|فاقد|تالف|فقدت|فقد |ضاع|ضايع|ضاعت",
    "unlock": r"\b(unlock\w*|lift\w*|reactivat\w*|unsuspend\w*)\b|رفع تعليق|رفع ايقاف|فك",
    "object": r"\b(object(ion)?|appeal\w*|complain\w*|dispute)\b|اعتراض|الاعتراض|اعترض|تظلم",
    "extract": r"\b(extract\w*|certificate|print\w*)\b|مستخرج|شهاده|طباعه|اطبع",
    "add": r"\b(add(ing)?)\b|اضافه|اضيف",
}
WEAK = {
    "issue": r"\b(regist(er|ering|ration)|get(ting)?|obtain\w*|apply(ing)?|application|new|open(ing)?)\b"
             r"|تسجيل|التسجيل|اسجل|قيد|الحصول|احصل|تقديم|اقدم|جديد|افتح|فتح",
    "pay": r"\b(pay(ing|ment)?|settle\w*)\b|سداد|اسدد|دفع|ادفع",
}
_STRONG = {k: re.compile(v) for k, v in STRONG.items()}
_WEAK = {k: re.compile(v) for k, v in WEAK.items()}
ACTIONS = tuple(sorted(set(STRONG) | set(WEAK)))


def _norm(text: str) -> str:
    return " " + normalize_for_matching(text or "") + " "


def query_actions(text: str) -> tuple[frozenset, bool]:
    """(actions, strong). Weak cues count only when no strong cue is present."""
    t = _norm(text)
    strong = frozenset(a for a, rx in _STRONG.items() if rx.search(t))
    if strong:
        return strong, True
    return frozenset(a for a, rx in _WEAK.items() if rx.search(t)), False


def title_action(title: str) -> str | None:
    """Primary action of a service title: the earliest strong cue, else the earliest weak cue."""
    t = _norm(title)
    for table in (_STRONG, _WEAK):
        best = None
        for a, rx in table.items():
            m = rx.search(t)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), a)
        if best:
            return best[1]
    return None


def service_actions(rec) -> frozenset:
    out = set()
    for lang in ("en", "ar"):
        a = title_action(rec.get("title", lang) or "")
        if a:
            out.add(a)
    return frozenset(out)


def adjust(query_acts: frozenset, strong: bool, svc_acts: frozenset, bonus: float, penalty: float) -> float:
    if not query_acts or not svc_acts:
        return 0.0
    if query_acts & svc_acts:
        return bonus
    return -penalty if strong else 0.0


def rerank_hits(hits, records, query: str, bonus: float, penalty: float, extra_actions=None):
    """Reorder hits by score + action adjustment. Hit scores stay the RAW retrieval scores."""
    acts, strong = query_actions(query)
    if not acts and extra_actions:
        acts, strong = frozenset(extra_actions), False
    if not acts or (bonus == 0 and penalty == 0):
        return list(hits)
    keyed = [(h.score + adjust(acts, strong, service_actions(records[h.service_id]), bonus, penalty), i, h)
             for i, h in enumerate(hits)]
    keyed.sort(key=lambda x: (-x[0], x[1]))
    return [replace(h) for _, _, h in keyed]
