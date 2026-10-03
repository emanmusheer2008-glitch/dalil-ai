"""Absher (Ministry of Interior) public e-services USER GUIDE -> ServiceRecord.

Source: the public, no-login service guide on absher.sa (Passport and Traffic sectors), captured by
``python -m src.ingestion.capture_absher`` into ``data/raw/official/absher/absher_guide_capture.json``.
Each guide page embeds its official content as a JS object (``currentServiceData``) with fields
such as title / about / terms / requirements / steps / beneficiary / fee / duration / channels.
The English and Arabic guides are separate pages; they are paired by Absher's own service key
(sector + data-name). Text is stored verbatim (HTML line breaks -> newlines, tags removed).
Fields the page leaves empty stay empty -- nothing is inferred.

Why Absher: my.gov.sa blocks automated access (Cloudflare) and moi.gov.sa did not respond; we do
not bypass either. Absher's public guide is the Ministry of Interior's own service documentation.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path

from src.ingestion.base import BaseLoader
from src.schema import ServiceRecord

AGENCY = {
    "Passport-Services": ("Ministry of Interior – Absher (Passports)", "وزارة الداخلية – أبشر (الجوازات)",
                          "Passports", "الجوازات"),
    "Traffic-Services": ("Ministry of Interior – Absher (Traffic)", "وزارة الداخلية – أبشر (المرور)",
                         "Traffic", "المرور"),
    "Business-Services": ("Ministry of Interior – Absher Business", "وزارة الداخلية – أبشر أعمال",
                          "Establishments (Absher Business)", "المنشآت (أبشر أعمال)"),
}
BIZ_SECTIONS = {         # Absher Business guide section id -> Absher key used below
    "service-purpose": "about", "service-requirements": "requirements", "service-terms": "terms",
    "service-beneficiary": "beneficiary", "service-implementation": "duration",
    "service-communicatewith": "steps",
}
FIELD_MAP = {            # Absher key -> ServiceRecord field
    "about": "description",
    "terms": "eligibility",            # "Terms and conditions" of the service
    "requirements": "requirements",
    "steps": "steps",
    "beneficiary": "target_audience",
    "fee": "fees",
    "duration": "processing_time",
    "language1": "service_languages",
}
_FIELD = re.compile(r'"(\w+)"\s*:\s*\{\s*"(en|ar)"\s*:\s*`(.*?)`', re.S)
_TAG = re.compile(r"<[^>]+>")


def clean(text: str) -> str | None:
    t = re.sub(r"<\s*/?\s*br\s*/?\s*>", "\n", text or "", flags=re.I)
    t = html.unescape(_TAG.sub("", t))
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in t.splitlines()]
    t = "\n".join(ln for ln in lines if ln)
    return t or None


def parse_business_html(main_html: str) -> dict[str, str]:
    from bs4 import BeautifulSoup
    sp = BeautifulSoup(main_html or "", "html.parser")
    out = {}
    t = sp.select_one("#service-title span.title-21")
    if t and (c := clean(t.get_text(" "))):
        out["title"] = c
    for sec_id, key in BIZ_SECTIONS.items():
        el = sp.find(id=sec_id)
        body = el.select_one(".jumbotron-body") if el else None
        if body and (c := clean(body.decode_contents())):
            out[key] = c
    return out


def parse_service_js(block: str) -> dict[str, str]:
    """{absher_key: text} for the language the page is in."""
    return {k: c for k, _lang, v in _FIELD.findall(block or "") if (c := clean(v))}


class AbsherGuideLoader(BaseLoader):
    name = "absher_user_guide"

    def load(self) -> list[ServiceRecord]:
        cap = json.loads(Path(self.path).read_text(encoding="utf-8"))
        pages: dict[tuple[str, str], dict[str, dict]] = {}
        for item in cap["records"]:
            if item.get("http_status") != 200 or not (item.get("service_data_js") or item.get("main_html")):
                continue
            data = (parse_service_js(item["service_data_js"]) if item.get("service_data_js")
                    else parse_business_html(item["main_html"]))
            if not data.get("title"):
                continue
            # an "ar" page must actually be Arabic, and vice versa
            is_ar = bool(re.search(r"[؀-ۿ]", data["title"]))
            if is_ar != (item["lang"] == "ar"):
                continue
            pages.setdefault((item["sector"], item["id"]), {})[item["lang"]] = {**item, "data": data}
        out = []
        for (sector, sid), by_lang in sorted(pages.items()):
            if sector not in AGENCY:
                continue
            a_en, a_ar, c_en, c_ar = AGENCY[sector]
            dates = sorted(p["fetched_at"] for p in by_lang.values())
            rec = ServiceRecord(
                service_id="absher-" + hashlib.sha1(f"{sector}/{sid}".encode()).hexdigest()[:10],
                source_type="official_web_page", source_domain="absher.sa",
                verification_status="verified_official_capture",
                date_collected=dates[0], last_verified=dates[-1][:10], source_record_id=f"{sector}/{sid}",
                service_channel="Absher (Ministry of Interior e-services platform)",
                source_sha256=";".join(p["sha256"] for _, p in sorted(by_lang.items())),
                provenance_notes="Captured from the public Absher e-services user guide (absher.sa, Ministry of "
                                 "Interior); text stored verbatim apart from HTML/whitespace normalisation.",
            )
            for lang, p in by_lang.items():
                d = p["data"]
                setattr(rec, f"title_{lang}", d["title"])
                setattr(rec, f"agency_{lang}", a_en if lang == "en" else a_ar)
                setattr(rec, f"category_{lang}", c_en if lang == "en" else c_ar)
                setattr(rec, f"official_url_{lang}", p["url"])
                for k, f in FIELD_MAP.items():
                    if d.get(k):
                        setattr(rec, f"{f}_{lang}", d[k])
                pay = d.get("payment")
                if pay and not getattr(rec, f"fees_{lang}"):
                    setattr(rec, f"fees_{lang}", pay)
                if d.get("channels"):
                    rec.extra[f"channels_{lang}"] = d["channels"]
            out.append(rec)
        return out
