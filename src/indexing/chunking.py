"""Turn service records into section-level chunks.

Each service is split into small, labelled pieces of evidence -- one per
section and language -- so that retrieval can point at *the* paragraph that
answers a question (e.g. the fees line) instead of a whole page.

Every chunk keeps the metadata needed to cite it: service id, title, agency,
section, language and official URL.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from src.schema import ServiceRecord

# (section name, fields that feed it). Order = display order.
SECTIONS: list[tuple[str, list[str]]] = [
    ("overview", ["description"]),
    ("requirements", ["eligibility", "requirements"]),
    ("documents", ["required_documents"]),
    ("steps", ["steps"]),
    ("service_facts", ["fees", "processing_time", "target_audience", "service_languages"]),
]

SECTION_LABELS = {
    "en": {
        "overview": "Overview",
        "requirements": "Conditions",
        "documents": "Required documents",
        "steps": "Steps",
        "service_facts": "Fees & service details",
        "description": "Description",
        "eligibility": "Eligibility",
        "required_documents": "Required documents",
        "fees": "Fees",
        "processing_time": "Processing time",
        "target_audience": "Beneficiaries",
        "service_languages": "Service languages",
    },
    "ar": {
        "overview": "نظرة عامة",
        "requirements": "الشروط",
        "documents": "المستندات المطلوبة",
        "steps": "الخطوات",
        "service_facts": "الرسوم وتفاصيل الخدمة",
        "description": "الوصف",
        "eligibility": "الأهلية",
        "required_documents": "المستندات المطلوبة",
        "fees": "الرسوم",
        "processing_time": "مدة التنفيذ",
        "target_audience": "الفئة المستفيدة",
        "service_languages": "لغة الخدمة",
    },
}

CHUNK_COLUMNS = [
    "chunk_id", "service_id", "lang", "section", "text", "embed_text",
    "title", "agency", "category", "official_url",
]


def _chunk_id(service_id: str, lang: str, section: str) -> str:
    return f"{service_id}::{lang}::{section}"


def record_to_chunks(rec: ServiceRecord) -> list[dict]:
    chunks = []
    for lang in ("en", "ar"):
        title = rec.get("title", lang)
        if not title:
            continue
        labels = SECTION_LABELS[lang]
        for section, field_names in SECTIONS:
            parts = []
            for name in field_names:
                value = rec.get(name, lang)
                if not value:
                    continue
                # Multi-field sections carry an inline label so the evidence
                # reads naturally ("Fees: 500 ...").
                parts.append(f"{labels[name]}: {value}" if len(field_names) > 1 else value)
            if not parts:
                continue
            text = "\n".join(parts)
            # The text that gets embedded is prefixed with the title and the
            # agency: a chunk like "Pay the fees" is meaningless on its own.
            agency = rec.get("agency", lang) or ""
            category = rec.get("category", lang) or ""
            embed_text = f"{title}. {agency}. {category}. {labels[section]}: {text}"
            chunks.append(
                {
                    "chunk_id": _chunk_id(rec.service_id, lang, section),
                    "service_id": rec.service_id,
                    "lang": lang,
                    "section": section,
                    "text": text,
                    "embed_text": embed_text,
                    "title": title,
                    "agency": agency,
                    "category": category,
                    "official_url": rec.get("official_url", lang) or rec.official_url_en or rec.official_url_ar,
                }
            )
    return chunks


def build_chunks(records: list[ServiceRecord]) -> pd.DataFrame:
    rows = [c for rec in records for c in record_to_chunks(rec)]
    df = pd.DataFrame(rows, columns=CHUNK_COLUMNS)
    if df["chunk_id"].duplicated().any():
        raise ValueError("duplicate chunk ids")
    return df


def chunks_fingerprint(df: pd.DataFrame) -> str:
    """Stable hash of the chunk texts -- used to detect a stale embedding cache."""
    h = hashlib.sha256()
    for cid, text in zip(df["chunk_id"], df["embed_text"]):
        h.update(cid.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()
