"""Load official pages that a person saved from their own browser.

This is the main legitimate route for adding sources that block automated
access. For every saved page, put two files in ``data/raw/manual/``:

    <name>.html   (or <name>.txt)   the page as saved (Ctrl+S)
    <name>.meta.json                 who/what/when, e.g.

    {
      "service_key": "moe-noncert-equivalency",   # same key for EN + AR pages
      "lang": "en",
      "official_url": "https://www.moe.gov.sa/en/...",
      "agency": "Ministry of Education",
      "date_saved": "2026-09-30T10:00:00Z",
      "saved_by": "project author"
    }

Only the title and main text are extracted (structured fields such as fees
are left ``None`` because a generic page does not label them reliably).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bs4 import BeautifulSoup

from src.ingestion.base import BaseLoader
from src.ingestion.normalization import clean_text, domain_of
from src.schema import ServiceRecord

_DROP_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "form", "aside"]


def html_to_title_and_text(html: str) -> tuple[str | None, str | None]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    title_tag = soup.find("h1") or soup.find("h2") or soup.find("title")
    title = clean_text(title_tag.get_text(" ", strip=True)) if title_tag else None
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = clean_text(main.get_text("\n", strip=True))
    return title, text


class SavedPageLoader(BaseLoader):
    name = "saved_pages"

    def load(self) -> list[ServiceRecord]:
        grouped: dict[str, ServiceRecord] = {}
        for meta_path in sorted(Path(self.path).glob("*.meta.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            stem = meta_path.name[: -len(".meta.json")]
            page = next((p for p in (meta_path.with_name(stem + ext) for ext in (".html", ".htm", ".txt")) if p.exists()), None)
            if page is None:
                continue
            raw = page.read_bytes()
            content = raw.decode("utf-8", errors="replace")
            if page.suffix == ".txt":
                title, text = meta.get("title"), clean_text(content)
            else:
                title, text = html_to_title_and_text(content)
                title = meta.get("title") or title
            lang = meta["lang"]
            key = meta.get("service_key", stem)
            rec = grouped.get(key) or ServiceRecord(
                service_id=f"manual-{key}",
                source_type="manual_saved_page",
                source_domain=domain_of(meta["official_url"]),
                verification_status="verified_official_capture",
                date_collected=meta.get("date_saved"),
                provenance_notes=f"Saved from an official page by {meta.get('saved_by', 'a person')}.",
            )
            setattr(rec, f"title_{lang}", title)
            setattr(rec, f"agency_{lang}", meta.get("agency"))
            setattr(rec, f"description_{lang}", text)
            setattr(rec, f"official_url_{lang}", meta["official_url"])
            digest = hashlib.sha256(raw).hexdigest()
            rec.source_sha256 = ";".join(filter(None, [rec.source_sha256, f"{lang}:{digest}"]))
            grouped[key] = rec
        return list(grouped.values())
