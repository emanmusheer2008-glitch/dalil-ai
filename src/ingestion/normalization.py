"""Text clean-up applied to every record before validation.

Clean-up is *non-semantic*: Unicode normalisation (NFC), removal of zero-width
characters, whitespace tidying. It never rewrites wording, so stored text stays
faithful to the official source.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

from src.schema import ServiceRecord
from src.utils.arabic import clean_unicode


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = clean_unicode(str(value))
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in value.splitlines()]
    value = "\n".join(ln for ln in lines if ln)
    return value or None


def domain_of(url: str | None) -> str | None:
    if not url:
        return None
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host or None


def normalize_record(rec: ServiceRecord) -> ServiceRecord:
    for name, value in list(rec.__dict__.items()):
        if isinstance(value, str):
            setattr(rec, name, clean_text(value))
    if not rec.source_domain:
        rec.source_domain = domain_of(rec.official_url_en or rec.official_url_ar)
    return rec
