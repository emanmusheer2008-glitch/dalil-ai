"""Evidence-grounded answer composer (no generative model).

Dalil does not write free text about government services. It:

1. decides whether the retrieved evidence is strong enough (calibrated
   threshold; otherwise it says it does not have enough information),
2. picks the best-matching service(s),
3. shows the *official* text of that service, field by field, only for fields
   the source actually contains, in the user's language when an official
   version in that language exists,
4. highlights the passages that matched the question, and
5. links to the official page.

Nothing is paraphrased, summarised or translated, so every sentence shown can
be traced back to the captured official source.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.indexing.chunking import SECTION_LABELS
from src.retrieval.retriever import SearchResult, ServiceHit
from src.schema import ServiceRecord
from src.utils.arabic import arabic_ratio

DISPLAY_FIELDS = [
    "description",
    "requirements",
    "eligibility",
    "required_documents",
    "steps",
    "fees",
    "processing_time",
    "target_audience",
    "service_languages",
]

MESSAGES = {
    "en": {
        "insufficient": (
            "I couldn't find enough information in Dalil's verified official sources to answer "
            "this question. Dalil only answers from the services it has indexed."
        ),
        "empty": "Please enter a question.",
        "lang_fallback": (
            "Official text for this service is only available in {available}. It is shown as "
            "published, without machine translation."
        ),
        "strong": "Strong match",
        "possible": "Possible match — check the official page",
    },
    "ar": {
        "insufficient": (
            "لم أجد معلومات كافية في المصادر الرسمية الموثقة لدى دليل للإجابة عن هذا السؤال. "
            "يجيب دليل فقط من الخدمات المفهرسة لديه."
        ),
        "empty": "يرجى كتابة سؤال.",
        "lang_fallback": "النص الرسمي لهذه الخدمة متوفر باللغة {available} فقط، ويُعرض كما نُشر دون ترجمة آلية.",
        "strong": "تطابق قوي",
        "possible": "تطابق محتمل — يرجى مراجعة الصفحة الرسمية",
    },
}

LANG_NAMES = {"en": {"en": "English", "ar": "Arabic"}, "ar": {"en": "الإنجليزية", "ar": "العربية"}}


@dataclass
class ServiceAnswer:
    service_id: str
    title: str
    agency: str | None
    category: str | None
    shown_lang: str
    fields: list[tuple[str, str, str]]          # (field_name, label, official text)
    evidence: list[tuple[str, str, float, str]]  # (section label, text, score, lang)
    official_url: str | None
    other_lang_url: str | None
    date_collected: str | None
    source_last_modified: str | None
    score: float
    confidence_label: str
    language_note: str | None = None


@dataclass
class Answer:
    query: str
    ui_lang: str
    status: str                      # "answered" | "insufficient" | "empty"
    message: str | None
    primary: ServiceAnswer | None = None
    related: list[ServiceAnswer] = field(default_factory=list)
    top_score: float | None = None
    threshold: float | None = None
    latency_ms: float | None = None


def query_language(query: str) -> str:
    return "ar" if arabic_ratio(query) >= 0.5 else "en"


def _service_answer(hit: ServiceHit, rec: ServiceRecord, ui_lang: str, threshold: float, strong_margin: float) -> ServiceAnswer:
    available = rec.languages()
    shown = ui_lang if ui_lang in available else available[0]
    labels = SECTION_LABELS[shown]
    fields_ = [
        (name, labels[name] if name in labels else name, rec.get(name, shown))
        for name in DISPLAY_FIELDS
        if rec.get(name, shown)
    ]
    ev_labels = SECTION_LABELS[ui_lang]
    evidence = [(ev_labels.get(e.section, e.section), e.text, e.score, e.lang) for e in hit.evidence]
    note = None
    if shown != ui_lang:
        note = MESSAGES[ui_lang]["lang_fallback"].format(available=LANG_NAMES[ui_lang][shown])
    other = "ar" if shown == "en" else "en"
    msg = MESSAGES[ui_lang]
    return ServiceAnswer(
        service_id=rec.service_id,
        title=rec.get("title", shown),
        agency=rec.get("agency", shown),
        category=rec.get("category", shown),
        shown_lang=shown,
        fields=fields_,
        evidence=evidence,
        official_url=rec.get("official_url", shown),
        other_lang_url=rec.get("official_url", other),
        date_collected=rec.date_collected,
        source_last_modified=rec.get("source_last_modified", shown),
        score=hit.score,
        confidence_label=msg["strong"] if hit.score >= threshold + strong_margin else msg["possible"],
        language_note=note,
    )


def compose(
    result: SearchResult,
    records: dict[str, ServiceRecord],
    threshold: float,
    related_margin: float = 0.05,
    strong_margin: float = 0.10,
    max_related: int = 2,
    ui_lang: str | None = None,
) -> Answer:
    ui_lang = ui_lang or query_language(result.query)
    msg = MESSAGES[ui_lang]
    if not result.query.strip():
        return Answer(result.query, ui_lang, "empty", msg["empty"])
    hits = result.hits
    top = hits[0].score if hits else None
    if not hits or top < threshold:
        return Answer(result.query, ui_lang, "insufficient", msg["insufficient"], top_score=top,
                      threshold=threshold, latency_ms=result.latency_ms)
    primary = _service_answer(hits[0], records[hits[0].service_id], ui_lang, threshold, strong_margin)
    related = [
        _service_answer(h, records[h.service_id], ui_lang, threshold, strong_margin)
        for h in hits[1:]
        if h.score >= threshold and top - h.score <= related_margin
    ][:max_related]
    return Answer(result.query, ui_lang, "answered", None, primary, related, top, threshold, result.latency_ms)
