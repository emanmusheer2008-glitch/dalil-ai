"""Validation, provenance checks and duplicate detection.

A record that fails a *blocking* check is quarantined (kept on disk for
review, excluded from the search index). Warnings are reported but do not
block indexing.
"""
from __future__ import annotations

import re

from dataclasses import dataclass, field
from urllib.parse import urlparse

from src.schema import (
    INDEXABLE_STATUSES,
    SOURCE_TYPES,
    VERIFICATION_STATUSES,
    ServiceRecord,
)
from src.utils.arabic import arabic_ratio, normalize_for_matching

# Domains accepted as official Saudi government sources. Extend deliberately.
OFFICIAL_DOMAIN_SUFFIXES = (".gov.sa", ".edu.sa")
OFFICIAL_EXACT_DOMAINS = {"my.gov.sa"}


@dataclass
class ValidationResult:
    record: ServiceRecord
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def is_valid_url(url: str | None) -> bool:
    if not url:
        return False
    p = urlparse(url)
    return p.scheme == "https" and bool(p.netloc)


def is_official_domain(domain: str | None) -> bool:
    if not domain:
        return False
    domain = domain.lower().split(":")[0]
    return domain in OFFICIAL_EXACT_DOMAINS or domain.endswith(OFFICIAL_DOMAIN_SUFFIXES)


def validate_record(rec: ServiceRecord) -> ValidationResult:
    res = ValidationResult(rec)

    # --- identity & required content -------------------------------------
    if not rec.service_id:
        res.errors.append("missing service_id")
    if not (rec.title_en or rec.title_ar):
        res.errors.append("missing title (en and ar)")
    # A title alone is not knowledge: require at least one substantive official field.
    # (A description is preferred but some agencies, e.g. SFDA, publish only steps/requirements.)
    content = ("description", "steps", "requirements", "required_documents", "eligibility")
    def substantive(v):  # "- -" or "N/A" placeholders are not content
        return bool(v) and len(re.findall(r"[^\W\d_]", str(v))) >= 15
    if not any(substantive(getattr(rec, f"{f}_{l}", None)) for f in content for l in ("en", "ar")):
        res.errors.append("no official content (description/steps/requirements/documents)")
    elif not (rec.description_en or rec.description_ar):
        res.warnings.append("no description (other official fields present)")
    if not (rec.official_url_en or rec.official_url_ar):
        res.errors.append("missing official_url")

    # --- URLs & domain ---------------------------------------------------
    for lang in ("en", "ar"):
        url = rec.get("official_url", lang)
        if url and not is_valid_url(url):
            res.errors.append(f"invalid official_url_{lang}: {url!r}")
    if not is_official_domain(rec.source_domain):
        res.errors.append(f"source_domain not an official Saudi domain: {rec.source_domain!r}")

    # --- provenance -------------------------------------------------------
    if rec.source_type not in SOURCE_TYPES:
        res.errors.append(f"unknown source_type {rec.source_type!r}")
    if rec.verification_status not in VERIFICATION_STATUSES:
        res.errors.append(f"unknown verification_status {rec.verification_status!r}")
    elif rec.verification_status not in INDEXABLE_STATUSES:
        res.errors.append(f"verification_status {rec.verification_status!r} is not indexable")
    if rec.verification_status == "verified_official_capture":
        if not rec.date_collected:
            res.errors.append("verified record without date_collected")
        if not rec.source_sha256:
            res.errors.append("verified record without source_sha256")

    # --- language sanity: Arabic fields should be Arabic, English Latin ----
    for name in ("title", "description"):
        ar = rec.get(name, "ar")
        en = rec.get(name, "en")
        if ar and arabic_ratio(ar) < 0.5:
            res.warnings.append(f"{name}_ar is mostly non-Arabic script")
        if en and arabic_ratio(en) > 0.5:
            res.warnings.append(f"{name}_en is mostly Arabic script")

    if not rec.title_ar:
        res.warnings.append("no official Arabic text")
    if not rec.title_en:
        res.warnings.append("no official English text")
    return res


def duplicate_key(rec: ServiceRecord) -> str:
    """Key used to detect the same service ingested twice."""
    url = (rec.official_url_en or rec.official_url_ar or "").lower().rstrip("/")
    url = url.replace("://www.", "://")
    title = normalize_for_matching(rec.title_en or rec.title_ar or "")
    agency = normalize_for_matching(rec.agency_en or rec.agency_ar or "")
    return url or f"{agency}|{title}"


def find_duplicates(records: list[ServiceRecord]) -> tuple[list[ServiceRecord], list[tuple[ServiceRecord, str]]]:
    """Return (unique_records, [(dropped_record, kept_service_id), ...]).

    Duplicates are detected by canonical URL and, independently, by
    normalised (agency, English title). The first occurrence is kept.
    """
    seen_url: dict[str, str] = {}
    seen_title: dict[str, str] = {}
    unique, dropped = [], []
    for rec in records:
        url_key = duplicate_key(rec)
        title_key = normalize_for_matching(f"{rec.agency_en or rec.agency_ar}|{rec.title_en or rec.title_ar}")
        hit = seen_url.get(url_key) or seen_title.get(title_key)
        if hit:
            dropped.append((rec, hit))
            continue
        seen_url[url_key] = rec.service_id
        seen_title[title_key] = rec.service_id
        unique.append(rec)
    return unique, dropped
