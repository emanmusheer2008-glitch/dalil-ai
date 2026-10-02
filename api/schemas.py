"""Request / response models for the Dalil AI API.

Field names mirror the engine's own objects (``src.answering.synthesizer``): an answer
is made of *sections* of *points*, every point carries a citation number that refers to
an entry in ``sources``. Nothing here adds information the engine does not produce.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_QUESTION_CHARS = 400          # same limit as the Streamlit app (app.py, MAX_QUERY_CHARS)

ResponseType = Literal["answer", "possible_match", "related_services", "unsupported"]


# ------------------------------------------------------------------ /ask --
class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(..., min_length=1, max_length=1200)


class AskRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [
        {"question": "How can I renew a commercial registration?", "language": "auto"},
        {"question": "كيف يمكنني تجديد السجل التجاري؟", "language": "ar"},
        {"question": "كم الرسوم؟", "language": "auto", "context_service_id": "mc-1"},
    ]})

    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS,
                          description="The user's question, in Arabic, English or mixed.")
    language: Literal["auto", "en", "ar"] = Field(
        "auto", description="Language of the answer text and labels. 'auto' (default) detects it from the "
                            "question, exactly like the app.")
    context_service_id: str | None = Field(
        None, max_length=32, pattern=r"^[a-z]{2,6}-[0-9a-z]{1,16}$",
        description="Optional, for follow-up questions: the service_id of the previous answer (e.g. the first "
                    "entry of `sources`). If the new question is a short follow-up (\"what documents do I need?\", "
                    "«كم الرسوم؟») retrieval stays on that service. Stateless: nothing is stored on the server. "
                    "Used by the lite and v4 runtimes.")
    conversation_context: list[ConversationTurn] | None = Field(
        None, max_length=8, description="Optional, v4 only: the last few turns (oldest first) so the AI layer can "
                                        "resolve follow-ups. Stateless; never stored.")
    use_ai: bool | None = Field(
        None, description="v4 only: false forces the deterministic (V3-style, verbatim) answer. Default: use the "
                          "grounded AI layer when the server has it configured.")

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        return v


class Source(BaseModel):
    citation: int = Field(..., description="The [n] number used by answer points.")
    service_id: str
    title: str | None
    agency: str | None
    url: str | None = Field(..., description="Official page in the language shown.")
    alternate_language_url: str | None = Field(None, description="Official page in the other language, if any.")
    language: str = Field(..., description="Language of the official text shown for this source (en/ar).")
    captured_at: str | None = Field(None, description="When Dalil captured the official page (ISO-8601).")
    source_last_modified: str | None = Field(None, description="'Last modified' date printed on the page, if any.")
    score: float = Field(..., description="Retrieval score of this service for the question.")


class AnswerPoint(BaseModel):
    text: str = Field(..., description="Verbatim official text.")
    citation: int
    label: str | None = Field(None, description="Dalil's inline label (e.g. 'Fees'); not part of the official text.")


class AnswerSection(BaseModel):
    key: str = Field(..., description="direct | need | docs | steps | fees | who | notes")
    title: str
    ordered: bool = Field(..., description="True for step-by-step lists.")
    points: list[AnswerPoint]


class Evidence(BaseModel):
    citation: int
    section: str
    text: str
    score: float
    language: str


class Thresholds(BaseModel):
    answer: float | None = None
    possible_match: float | None = None
    related: float | None = None


class AskResponse(BaseModel):
    question: str
    detected_language: Literal["en", "ar"] = Field(..., description="Language detected from the question.")
    answer_language: Literal["en", "ar"] = Field(..., description="Language used for the answer labels/text.")
    response_type: ResponseType = Field(
        ..., description="answer = confident; possible_match = shown with a warning; related_services = no "
                         "answer, closest official services as links; unsupported = outside Dalil's sources.")
    engine_status: str = Field(..., description="Raw engine status: answered | tentative | insufficient.")
    message: str | None = Field(None, description="Engine message for unsupported / related-services cases.")
    lead: str | None = Field(None, description="Dalil's templated lead sentence (service and agency names only).")
    intent: str
    sections: list[AnswerSection]
    sources: list[Source]
    related_services: list[Source] = Field(
        default_factory=list, description="Only for related_services: links, NOT an answer.")
    notes: list[str] = Field(default_factory=list, description="Dalil's notes (language availability, conflicts).")
    evidence: list[Evidence] = Field(default_factory=list, description="Retrieved official passages behind the answer.")
    top_score: float | None
    thresholds: Thresholds
    latency_ms: float
    context_service_id: str | None = Field(
        None, description="The follow-up context actually applied (null if the question was treated as new).")
    runtime: str = Field(..., description="Engine runtime: 'v4' (V3 + grounded AI layer), 'lite' (V3) or 'full' (V2).")
    response_mode: str | None = Field(
        None, description="v4: grounded_answer | partial_answer | possible_match | related_services | "
                          "insufficient_evidence.")
    answer: str | None = Field(None, description="v4 AI: short direct answer, written only from cited evidence.")
    text_origin: str = Field("official_verbatim", description="official_verbatim (V3 text) or "
                                                                "ai_generated_from_cited_evidence (v4 AI).")
    ai_enabled: bool = False
    ai_used: bool = False
    ai_model: str | None = None
    ai_fallback_reason: str | None = Field(None, description="Why the AI answer was not used (e.g. timeout).")
    verified_fields: list[str] = Field(default_factory=list, description="Evidence fields the answer cites.")
    unverified_information: list[str] = Field(default_factory=list,
                                              description="What Dalil could NOT verify from official evidence.")
    follow_up_suggestions: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list, description="v4: queries used for retrieval.")
    redactions_applied: int = Field(0, description="Personal identifiers removed before AI processing.")
    disclaimer: str


# --------------------------------------------------------- services ------
class ServiceSummary(BaseModel):
    service_id: str
    agency_code: str
    title: str | None
    agency: str | None
    official_url: str | None
    languages: list[str]
    description: str | None = Field(None, description="Official description (first 300 characters).")


class ServiceList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ServiceSummary]


class ServiceLanguageBlock(BaseModel):
    title: str | None = None
    agency: str | None = None
    category: str | None = None
    description: str | None = None
    eligibility: str | None = None
    requirements: str | None = None
    required_documents: str | None = None
    steps: str | None = None
    fees: str | None = None
    processing_time: str | None = None
    target_audience: str | None = None
    service_languages: str | None = None
    notes: str | None = None
    official_url: str | None = None
    source_last_modified: str | None = None


class ServiceDetail(BaseModel):
    service_id: str
    agency_code: str
    source_domain: str
    verification_status: str
    captured_at: str | None
    last_verified: str | None
    contact_information: str | None
    service_channel: str | None
    source_sha256: str | None
    en: ServiceLanguageBlock | None
    ar: ServiceLanguageBlock | None


class Agency(BaseModel):
    code: str
    name_en: str | None
    name_ar: str | None
    domain: str | None
    services: int
    services_with_arabic: int
    services_with_english: int


class ErrorResponse(BaseModel):
    detail: str
