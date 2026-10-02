"""Canonical service-record schema.

Design rules
------------
* A field is populated ONLY when the source states it. Missing -> ``None``.
* Language-bearing text fields exist in ``_en`` and ``_ar`` variants. Each
  variant holds the *official* text in that language, captured verbatim from
  an official page in that language. Dalil never stores machine translations
  in these fields.
* Every record carries provenance: where it came from, when it was captured,
  and how it was verified.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

# Text fields that come in an English and an Arabic variant.
BILINGUAL_FIELDS = [
    "title",
    "agency",
    "category",
    "description",
    "eligibility",
    "requirements",
    "required_documents",
    "steps",
    "fees",
    "processing_time",
    "target_audience",
    "service_languages",
    "notes",
    "official_url",
    "source_last_modified",
]

# Controlled vocabularies ---------------------------------------------------
SOURCE_TYPES = {
    "official_web_page",        # captured from an official government web page
    "official_open_data",       # official open-data download / API
    "manual_saved_page",        # official page saved by a person from a browser
    "seed_manual_summary",      # hand-written summary (development seed)
}

VERIFICATION_STATUSES = {
    # Text captured verbatim from the official page on `date_collected`,
    # with a SHA-256 of the captured HTML stored for audit.
    "verified_official_capture",
    # Not traceable to captured official text -> excluded from the index.
    "unverified",
}

INDEXABLE_STATUSES = {"verified_official_capture"}


@dataclass
class ServiceRecord:
    service_id: str
    source_type: str
    source_domain: str
    verification_status: str
    date_collected: Optional[str] = None     # ISO-8601 UTC timestamp
    last_verified: Optional[str] = None      # date the official page was last re-checked
    contact_information: Optional[str] = None
    service_channel: Optional[str] = None
    source_record_id: Optional[str] = None   # identifier used by the source (e.g. sID)
    source_sha256: Optional[str] = None      # hash(es) of captured raw HTML
    provenance_notes: Optional[str] = None
    # bilingual fields --------------------------------------------------------
    title_en: Optional[str] = None
    title_ar: Optional[str] = None
    agency_en: Optional[str] = None
    agency_ar: Optional[str] = None
    category_en: Optional[str] = None
    category_ar: Optional[str] = None
    description_en: Optional[str] = None
    description_ar: Optional[str] = None
    eligibility_en: Optional[str] = None
    eligibility_ar: Optional[str] = None
    requirements_en: Optional[str] = None
    requirements_ar: Optional[str] = None
    required_documents_en: Optional[str] = None
    required_documents_ar: Optional[str] = None
    steps_en: Optional[str] = None
    steps_ar: Optional[str] = None
    fees_en: Optional[str] = None
    fees_ar: Optional[str] = None
    processing_time_en: Optional[str] = None
    processing_time_ar: Optional[str] = None
    target_audience_en: Optional[str] = None
    target_audience_ar: Optional[str] = None
    service_languages_en: Optional[str] = None
    service_languages_ar: Optional[str] = None
    notes_en: Optional[str] = None
    notes_ar: Optional[str] = None
    official_url_en: Optional[str] = None
    official_url_ar: Optional[str] = None
    source_last_modified_en: Optional[str] = None
    source_last_modified_ar: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    # helpers -----------------------------------------------------------------
    def get(self, name: str, lang: str) -> Optional[str]:
        return getattr(self, f"{name}_{lang}", None)

    def languages(self) -> list[str]:
        return [lang for lang in ("en", "ar") if self.get("title", lang)]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ServiceRecord":
        names = {f.name for f in fields(cls)}
        clean = {k: (None if _is_missing(v) else v) for k, v in d.items() if k in names}
        if clean.get("extra") is None:
            clean["extra"] = {}
        return cls(**clean)


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        import math

        if isinstance(v, float) and math.isnan(v):
            return True
    except Exception:  # pragma: no cover
        pass
    return isinstance(v, str) and not v.strip()


def schema_columns() -> list[str]:
    return [f.name for f in fields(ServiceRecord)]
