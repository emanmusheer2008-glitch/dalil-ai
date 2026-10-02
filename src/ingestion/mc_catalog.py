"""Parser for the Ministry of Commerce e-services catalogue (mc.gov.sa).

Input: ``data/raw/official/mc/mc_services_capture.json`` -- a capture of the
official service-detail pages, English and Arabic, made from a normal browser
session at a low request rate (see docs/data_provenance.md). Each capture item
stores the page URL, fetch time, HTTP status and the page's ``<article>`` HTML
exactly as served.

The parser is deterministic: re-running it on the same capture produces the
same records. Only labelled sections of the page are mapped to schema fields;
anything the page does not state stays ``None``.
"""
from __future__ import annotations

import hashlib
import json
import re

from bs4 import BeautifulSoup

from src.ingestion.base import BaseLoader
from src.ingestion.normalization import clean_text
from src.schema import ServiceRecord

AGENCY = {"en": "Ministry of Commerce", "ar": "وزارة التجارة"}

# Labels used on the page, per language -> schema field.
META_LABELS = {
    "en": {
        "Beneficiary Group": "target_audience",
        "Duration of service": "processing_time",
        "The service is provided in": "service_languages",
        "Service Fees": "fees",
        "Pay Methods": "_pay_methods",
    },
    "ar": {
        "الفئة المستفيدة": "target_audience",
        "مدة تنفيذ الخدمة": "processing_time",
        "الخدمة مقدمة باللغة": "service_languages",
        "رسوم الخدمة": "fees",
        "طرق الدفع": "_pay_methods",
    },
}
# Lines that end the meta block (FAQ / contact / rating / related services).
META_STOP = re.compile(
    r"^(For Frequently Asked Questions|You can call|To contact|Download|Rate:|Related Services|"
    r"للاطلاع على الأسئلة|يمكنكم الاتصال|للتواصل|تحميل دليل|التقييم|خدمات ذات صلة)"
)
START_LABEL = {"en": "Start", "ar": "ابدأ الخدمة"}
SLA_LABEL = re.compile(r"^(Service level agreement|إتفاقية مستوى الخدمة|اتفاقية مستوى الخدمة)$")
TAB_HEADINGS = {"Steps", "Conditions", "Required Documents", "الخطوات", "الشروط", "المستندات المطلوبة"}
LAST_MODIFIED = re.compile(r"^(Last Modified|آخر تعديل)\s*(.+)$")
PHONE_AFTER = re.compile(r"(unified number|الرقم الموحد)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def _lines(el) -> list[str]:
    text = clean_text(el.get_text("\n", strip=True)) or ""
    return [ln for ln in text.split("\n") if ln]


def _list_section(soup, element_id: str) -> str | None:
    el = soup.find(id=element_id)
    if el is None:
        return None
    items = [clean_text(li.get_text(" ", strip=True)) for li in el.find_all("li")]
    items = [i for i in items if i]
    if not items:
        items = _lines(el)
    return "\n".join(f"- {i}" for i in items) or None


def parse_article(article_html: str, lang: str, page_title: str | None) -> dict:
    soup = BeautifulSoup(article_html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    out: dict = {
        "steps": _list_section(soup, "steps"),
        "requirements": _list_section(soup, "terms"),
        "required_documents": _list_section(soup, "document"),
    }
    for el_id in ("steps", "terms", "document"):
        el = soup.find(id=el_id)
        if el is not None:
            el.decompose()

    lines = _lines(soup)

    # ---- title ---------------------------------------------------------
    title = clean_text(page_title) if page_title else None
    if title and "|" in title:
        title = clean_text(title.split("|")[-1])
    # The page repeats the service title on two consecutive lines; prefer it.
    for a, b in zip(lines, lines[1:]):
        if a == b and len(a) > 3:
            title = a
            break
    out["title"] = title

    # ---- header: tags + description --------------------------------------
    try:
        start = lines.index(START_LABEL[lang])
    except ValueError:
        start = None
    # The header ends at the "Service level agreement" link or, when a page has
    # none, at the first tab heading ("Steps" / "الخطوات").
    sla = next((i for i, ln in enumerate(lines) if SLA_LABEL.match(ln) or ln in TAB_HEADINGS), None)
    description, tags = None, []
    if start is not None and sla is not None and sla > start:
        head = lines[start + 1 : sla]
        # Tags are short labels; the description is the prose that follows them.
        split = next((i for i, ln in enumerate(head) if len(ln) > 60), len(head))
        tags, desc_lines = head[:split], head[split:]
        description = "\n".join(desc_lines) or None
    out["description"] = description
    out["tags"] = tags

    # ---- meta block ------------------------------------------------------
    labels = META_LABELS[lang]
    current, buf, meta = None, [], {}
    for ln in lines[(sla or 0):]:
        if META_STOP.match(ln):
            if current:
                meta[current] = "\n".join(buf) or None
            current, buf = None, []
            if ln.startswith(("Related Services", "خدمات ذات صلة")):
                break
            continue
        if ln in labels:
            if current:
                meta[current] = "\n".join(buf) or None
            current, buf = labels[ln], []
        elif current and ln not in TAB_HEADINGS:
            buf.append(ln)
    if current:
        meta[current] = "\n".join(buf) or None
    for k, v in meta.items():
        if not k.startswith("_"):
            out[k] = v

    # ---- contact & last modified ----------------------------------------
    contact = []
    for i, ln in enumerate(lines):
        if PHONE_AFTER.search(ln):
            phone = re.search(r"\d{3,}", ln) or (re.search(r"\d{3,}", lines[i + 1]) if i + 1 < len(lines) else None)
            if phone:
                contact.append(phone.group(0))
        m = EMAIL.search(ln)
        if m and m.group(0) not in contact:
            contact.append(m.group(0))
        lm = LAST_MODIFIED.match(ln)
        if lm:
            out["source_last_modified"] = lm.group(2).strip()
    out["contact"] = ", ".join(dict.fromkeys(contact)) or None
    return out


class MCCatalogLoader(BaseLoader):
    name = "mc_catalog"

    def load(self) -> list[ServiceRecord]:
        capture = json.loads(self.path.read_text(encoding="utf-8"))
        by_sid: dict[int, dict[str, dict]] = {}
        for item in capture["records"]:
            if item.get("http_status") != 200 or not item.get("article_html"):
                continue
            by_sid.setdefault(int(item["sid"]), {})[item["lang"]] = item

        records = []
        for sid in sorted(by_sid):
            pages = by_sid[sid]
            dates = sorted(p["fetched_at"] for p in pages.values())
            rec = ServiceRecord(
                service_id=f"mc-{sid}",
                source_type="official_web_page",
                source_domain="mc.gov.sa",
                verification_status="verified_official_capture",
                date_collected=dates[0],
                last_verified=dates[-1][:10],
                source_record_id=str(sid),
                service_channel="Online (Ministry of Commerce / Saudi Business Center e-services)",
                provenance_notes=(
                    "Captured from the official Ministry of Commerce e-services catalogue "
                    "(service-detail pages, English and Arabic). Text is stored verbatim "
                    "apart from Unicode/whitespace normalisation."
                ),
            )
            hashes, tags = [], {}
            for lang, page in sorted(pages.items()):
                parsed = parse_article(page["article_html"], lang, page.get("page_title"))
                setattr(rec, f"title_{lang}", parsed["title"])
                setattr(rec, f"agency_{lang}", AGENCY[lang])
                setattr(rec, f"official_url_{lang}", page["url"])
                for name in ("description", "requirements", "required_documents", "steps", "fees",
                             "processing_time", "target_audience", "service_languages",
                             "source_last_modified"):
                    setattr(rec, f"{name}_{lang}", parsed.get(name))
                # The page shows a few short official labels (e.g. "Merchant",
                # "Commercial Register", "Business sector"). Their order is not
                # consistent across pages, so they are kept verbatim as the
                # category string rather than guessing which one is "the" category.
                if parsed["tags"]:
                    setattr(rec, f"category_{lang}", " · ".join(parsed["tags"]))
                tags[lang] = parsed["tags"]
                if parsed.get("contact") and not rec.contact_information:
                    rec.contact_information = parsed["contact"]
                hashes.append(f"{lang}:{hashlib.sha256(page['article_html'].encode('utf-8')).hexdigest()}")
            rec.source_sha256 = ";".join(hashes)
            rec.extra = {"page_tags": tags}
            records.append(rec)
        return records
