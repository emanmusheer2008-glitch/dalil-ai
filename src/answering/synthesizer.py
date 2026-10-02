"""V2 answer synthesis: one organised answer assembled from several official passages.

What is *Dalil's* and what is *official*
----------------------------------------
Every factual line in an answer is an **official passage** (a sentence or list
item copied from a captured official page) and carries a citation number.
Dalil's own contribution is limited to organisation:

* choosing which services and passages are relevant,
* ordering them under headings (Direct answer, What you need, Documents,
  Steps, Fees & processing, Important notes, Official sources),
* one templated lead sentence naming the service and agency (both taken from
  the official page), and
* notes about uncertainty, language availability and conflicting sources.

No fee, requirement, document, deadline or step is ever written by Dalil. If a
source doesn't state something, the heading is simply omitted. This keeps the
answer detailed *and* checkable without a generative model (zero cost, no
hallucination risk).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from src.retrieval.retriever import SearchResult
from src.schema import ServiceRecord
from src.utils.arabic import arabic_ratio, normalize_for_matching

# ----------------------------------------------------------------- strings --
UI = {
    "en": {
        "lead": "The official service that matches your question is “{title}” ({agency}).",
        "lead_tentative": "I'm not certain this is exactly what you mean. The closest official service in Dalil's "
                          "sources is “{title}” ({agency}) — please check it against your situation.",
        "insufficient": "I couldn't find enough information in Dalil's verified official sources to answer this "
                        "question responsibly. Dalil only answers from the official pages it has indexed.",
        "insufficient_related": "Dalil's official sources don't answer this exactly. These official services are the "
                                "closest matches — open them to check whether one fits your situation.",
        "empty": "Please enter a question.",
        "lang_note": "Official text for [{n}] is only published in {lang}; it is shown as published, without machine "
                     "translation.",
        "conflict": "Sources state different values for {what}; each is shown with its source. Check which one "
                    "applies to you.",
        "sec_direct": "Direct answer", "sec_need": "What you need", "sec_docs": "Documents",
        "sec_steps": "Steps", "sec_fees": "Fees & processing", "sec_who": "Who it is for",
        "sec_notes": "Important notes", "sec_also": "Also relevant", "sec_sources": "Official sources",
        "fees": "Fees", "time": "Processing time", "langs": "Service languages", "channel": "Channel",
        "contact": "Contact",
    },
    "ar": {
        "lead": "الخدمة الرسمية المطابقة لسؤالك هي «{title}» ({agency}).",
        "lead_tentative": "لست متأكداً أن هذا ما تقصده تماماً. أقرب خدمة رسمية في مصادر دليل هي «{title}» ({agency}) "
                          "— يرجى التحقق من مناسبتها لحالتك.",
        "insufficient": "لم أجد معلومات كافية في مصادر دليل الرسمية الموثقة للإجابة عن هذا السؤال بشكل مسؤول. "
                        "يجيب دليل فقط من الصفحات الرسمية المفهرسة لديه.",
        "insufficient_related": "لا تجيب مصادر دليل الرسمية عن هذا السؤال بدقة. هذه أقرب الخدمات الرسمية — "
                                "افتحها للتحقق مما إذا كانت إحداها تناسب حالتك.",
        "empty": "يرجى كتابة سؤال.",
        "lang_note": "النص الرسمي للمصدر [{n}] منشور باللغة {lang} فقط، ويُعرض كما نُشر دون ترجمة آلية.",
        "conflict": "تذكر المصادر قيماً مختلفة لـ{what}؛ تُعرض كل قيمة مع مصدرها. تحقّق أيها ينطبق عليك.",
        "sec_direct": "الإجابة المباشرة", "sec_need": "ما تحتاجه", "sec_docs": "المستندات",
        "sec_steps": "الخطوات", "sec_fees": "الرسوم والمدة", "sec_who": "الفئة المستفيدة",
        "sec_notes": "ملاحظات مهمة", "sec_also": "خدمات ذات صلة", "sec_sources": "المصادر الرسمية",
        "fees": "الرسوم", "time": "مدة التنفيذ", "langs": "لغة الخدمة", "channel": "القناة",
        "contact": "التواصل",
    },
}
LANG_NAME = {"en": {"en": "English", "ar": "Arabic"}, "ar": {"en": "الإنجليزية", "ar": "العربية"}}

# Query intents -> the section the user most likely wants first.
INTENT_PATTERNS = {
    "fees": r"\b(fee|fees|cost|costs|price|how much|charge|pay|payment)\b|رسوم|تكلف|كم سعر|بكم|قيمه|مبلغ|سداد|ادفع",
    "time": r"\b(how long|duration|processing time|days|time does|when will)\b|مده|كم ياخذ|متي|كم يوم|ايام",
    "documents": r"\b(documents?|papers?|paperwork|bring|attach|required docs?)\b|مستند|اوراق|وثاءق|مرفقات|وش احتاج|ايش احتاج",
    "requirements": r"\b(eligib\w*|requirements?|conditions?|who can|qualif\w*|allowed)\b|شروط|اشتراطات|يحق|مين يقدر|الاهليه|متطلبات",
    "steps": r"\b(how (do|can|to)|steps?|procedure|process|apply|register|where can)\b|كيف|خطوات|طريقه|اجراءات|اقدم|اسجل",
}


def detect_intent(query: str) -> str:
    q = normalize_for_matching(query)
    for intent in ("fees", "time", "documents", "requirements", "steps"):
        if re.search(INTENT_PATTERNS[intent], q):
            return intent
    return "general"


def query_language(query: str) -> str:
    return "ar" if arabic_ratio(query) >= 0.5 else "en"


# ------------------------------------------------------------ data model --
@dataclass
class Citation:
    n: int
    service_id: str
    title: str
    agency: str | None
    url: str | None
    other_url: str | None
    lang: str
    date_collected: str | None
    source_last_modified: str | None
    score: float


@dataclass
class Point:
    text: str          # official text, verbatim
    cite: int          # citation number
    label: str | None = None   # optional inline label ("Fees") -- Dalil's heading, not a fact


@dataclass
class Section:
    key: str
    title: str
    points: list[Point]
    ordered: bool = False


@dataclass
class SynthAnswer:
    query: str
    ui_lang: str
    status: str                       # answered | tentative | insufficient | empty
    lead: str | None = None           # Dalil's templated sentence (names only)
    sections: list[Section] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)      # Dalil's notes (language, conflicts)
    evidence: list[tuple[int, str, str, float, str]] = field(default_factory=list)  # (cite, section, text, score, lang)
    intent: str = "general"
    top_score: float | None = None
    thresholds: dict | None = None
    latency_ms: float | None = None
    message: str | None = None
    related: list[Citation] = field(default_factory=list)   # closest services when Dalil declines (links only)


# ----------------------------------------------------------------- helpers --
_SENT_SPLIT = re.compile(r"(?<=[.!?؟])\s+(?=\S)|\n+")


def split_items(text: str | None) -> list[str]:
    """Split an official field into items: list lines ("- x") or sentences."""
    if not text:
        return []
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if any(ln.startswith("- ") for ln in lines):
        return [ln[2:].strip() if ln.startswith("- ") else ln for ln in lines]
    out = []
    for ln in lines:
        out.extend(s.strip() for s in _SENT_SPLIT.split(ln) if s.strip())
    return out


def _pick_lang(rec: ServiceRecord, ui_lang: str) -> str:
    langs = rec.languages()
    return ui_lang if ui_lang in langs else langs[0]


def _relevant_sentences(query_vec, sentences: list[str], encode, k: int = 2) -> list[str]:
    if not sentences:
        return []
    if len(sentences) <= k or encode is None or query_vec is None:
        return sentences[:k]
    vecs = encode(sentences)
    sims = vecs @ query_vec
    keep = sorted(np.argsort(-sims)[:k])          # keep original order for readability
    return [sentences[i] for i in keep]


# -------------------------------------------------------------- main API --
def synthesize(
    result: SearchResult,
    records: dict[str, ServiceRecord],
    t_answer: float,
    t_tentative: float | None = None,
    t_related: float | None = None,
    support_margin: float = 0.06,
    max_support: int = 2,
    ui_lang: str | None = None,
    encode=None,          # optional: callable(list[str]) -> np.ndarray (normalised) for sentence ranking
    query_vec=None,
) -> SynthAnswer:
    ui_lang = ui_lang or query_language(result.query)
    s = UI[ui_lang]
    t_tentative = t_answer if t_tentative is None else min(t_tentative, t_answer)
    thresholds = {"answer": t_answer, "tentative": t_tentative}
    if not result.query.strip():
        return SynthAnswer(result.query, ui_lang, "empty", message=s["empty"], thresholds=thresholds)
    hits = result.hits
    top = hits[0].score if hits else None
    if not hits or top < t_tentative:
        # Not enough to answer. If the question is still clearly about public services (score above the
        # related floor), show the closest official services as LINKS ONLY -- titles, agencies, URLs --
        # never their fees/steps as if they answered the question.
        related = []
        if hits and t_related is not None and top >= t_related:
            for i, h in enumerate(hits[:3], 1):
                if h.score < t_related:
                    break
                rec = records[h.service_id]
                lang = _pick_lang(rec, ui_lang)
                other = "ar" if lang == "en" else "en"
                related.append(Citation(i, rec.service_id, rec.get("title", lang), rec.get("agency", lang),
                                        rec.get("official_url", lang), rec.get("official_url", other), lang,
                                        rec.date_collected, rec.get("source_last_modified", lang), h.score))
        thresholds["related"] = t_related
        return SynthAnswer(result.query, ui_lang, "insufficient",
                           message=s["insufficient_related"] if related else s["insufficient"], top_score=top,
                           thresholds=thresholds, latency_ms=result.latency_ms, related=related)
    status = "answered" if top >= t_answer else "tentative"

    # --- services used: the best one + close, compatible supporting services
    used = [hits[0]]
    for h in hits[1:]:
        if len(used) > max_support:
            break
        if h.score >= t_tentative and top - h.score <= support_margin:
            used.append(h)

    citations: list[Citation] = []
    for i, h in enumerate(used, 1):
        rec = records[h.service_id]
        lang = _pick_lang(rec, ui_lang)
        other = "ar" if lang == "en" else "en"
        citations.append(Citation(i, rec.service_id, rec.get("title", lang), rec.get("agency", lang),
                                  rec.get("official_url", lang), rec.get("official_url", other), lang,
                                  rec.date_collected, rec.get("source_last_modified", lang), h.score))

    primary_rec = records[used[0].service_id]
    plang = citations[0].lang
    intent = detect_intent(result.query)
    notes = []
    for c in citations:
        if c.lang != ui_lang:
            notes.append(s["lang_note"].format(n=c.n, lang=LANG_NAME[ui_lang][c.lang]))

    sections: list[Section] = []
    # ---- direct answer: the official passages that best answer the intent
    # The direct answer is the service description; the section matching the question's intent
    # (fees / documents / steps ...) is moved right after it below, so nothing is shown twice.
    direct: list[Point] = []
    desc_sents = split_items(primary_rec.get("description", plang))
    for sent in _relevant_sentences(query_vec, desc_sents, encode, k=2):
        direct.append(Point(sent, 1))
    if direct:
        sections.append(Section("direct", s["sec_direct"], direct))

    # ---- structured sections (primary service first, then supporting ones)
    def gather(field_names, ordered=False):
        pts = []
        for c, h in zip(citations, used):
            rec = records[h.service_id]
            for name in field_names:
                for item in split_items(rec.get(name, c.lang)):
                    pts.append(Point(item, c.n))
            if pts and c.n == 1 and len(citations) > 1 and ordered:
                break               # steps: don't interleave procedures of different services
        return pts

    need = gather(["eligibility", "requirements"])
    if need:
        sections.append(Section("need", s["sec_need"], need))
    docs = gather(["required_documents"])
    if docs:
        sections.append(Section("docs", s["sec_docs"], docs))
    steps = gather(["steps"], ordered=True)
    if steps:
        sections.append(Section("steps", s["sec_steps"], steps, ordered=True))

    fees_pts: list[Point] = []
    fee_values: dict[str, set] = {}
    for c, h in zip(citations, used):
        rec = records[h.service_id]
        for name, label in (("fees", s["fees"]), ("processing_time", s["time"]), ("service_languages", s["langs"])):
            val = rec.get(name, c.lang)
            if val:
                fees_pts.append(Point(val, c.n, label))
                if name == "fees":
                    fee_values.setdefault(normalize_for_matching(val), set()).add(c.n)
    if fees_pts:
        sections.append(Section("fees", s["sec_fees"], fees_pts))
    # Conflict surfacing: only between *different agencies* describing fees
    agencies = {c.agency for c in citations}
    if len(fee_values) > 1 and len(agencies) > 1:
        notes.append(s["conflict"].format(what=s["fees"]))

    who = [Point(v, c.n) for c, h in zip(citations, used)
           for v in [records[h.service_id].get("target_audience", c.lang)] if v]
    if who:
        sections.append(Section("who", s["sec_who"], who[:3]))
    notes_pts = gather(["notes"])
    if primary_rec.contact_information:
        notes_pts.append(Point(primary_rec.contact_information, 1, s["contact"]))
    if notes_pts:
        sections.append(Section("notes", s["sec_notes"], notes_pts))

    # Move the section matching the intent right after the direct answer
    order_pref = {"documents": "docs", "requirements": "need", "steps": "steps", "fees": "fees", "time": "fees"}
    want = order_pref.get(intent)
    if want:
        sections.sort(key=lambda sec: (sec.key != "direct", sec.key != want))

    lead_tpl = s["lead"] if status == "answered" else s["lead_tentative"]
    lead = lead_tpl.format(title=citations[0].title, agency=citations[0].agency or "")

    evidence = []
    for c, h in zip(citations, used):
        for e in h.evidence:
            evidence.append((c.n, e.section, e.text, e.score, e.lang))

    return SynthAnswer(result.query, ui_lang, status, lead, sections, citations, notes, evidence, intent, top,
                       thresholds, result.latency_ms)


def official_texts(ans: SynthAnswer) -> list[str]:
    """All factual passages in the answer (used by tests to check grounding)."""
    return [p.text for sec in ans.sections for p in sec.points]
