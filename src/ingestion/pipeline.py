"""Build the canonical knowledge base from every raw source.

    python -m src.ingestion.pipeline

Steps: load -> normalise -> validate -> de-duplicate -> write
``data/processed/services.jsonl`` (+ ``services.csv`` for inspection),
``quarantine.jsonl`` (records excluded, with reasons) and ``build_report.json``.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

from src import config
from src.ingestion.base import BaseLoader
from src.ingestion.csv_loader import LegacySeedCSVLoader
from src.ingestion.document_loader import SavedPageLoader
from src.ingestion.mc_catalog import MCCatalogLoader
from src.ingestion.normalization import normalize_record
from src.ingestion.official_open_data import OpenDataFileLoader
from src.ingestion.validation import find_duplicates, validate_record
from src.schema import ServiceRecord, schema_columns


def default_loaders() -> list[BaseLoader]:
    raw = config.RAW_DIR
    loaders: list[BaseLoader] = []
    mc = raw / "official" / "mc" / "mc_services_capture.json"
    if mc.exists():
        loaders.append(MCCatalogLoader(mc))
    if (raw / "manual").exists():
        loaders.append(SavedPageLoader(raw / "manual"))
    if (raw / "open_data").exists():
        loaders.append(OpenDataFileLoader(raw / "open_data"))
    seed = raw / "seed" / "services_seed_v0.csv"
    if seed.exists():
        loaders.append(LegacySeedCSVLoader(seed))
    return loaders


def run(loaders: list[BaseLoader] | None = None, write: bool = True) -> dict:
    loaders = default_loaders() if loaders is None else loaders
    accepted: list[ServiceRecord] = []
    quarantined: list[dict] = []
    per_source = {}

    for loader in loaders:
        recs = [normalize_record(r) for r in loader.load()]
        ok = 0
        for rec in recs:
            res = validate_record(rec)
            if res.ok:
                accepted.append(rec)
                ok += 1
            else:
                quarantined.append({**rec.to_dict(), "_errors": res.errors, "_warnings": res.warnings})
        per_source[loader.describe()] = {"loaded": len(recs), "accepted": ok}

    unique, dropped = find_duplicates(accepted)
    for rec, kept in dropped:
        quarantined.append({**rec.to_dict(), "_errors": [f"duplicate of {kept}"], "_warnings": []})

    warnings = Counter(w for rec in unique for w in validate_record(rec).warnings)
    report = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": per_source,
        "services_indexed": len(unique),
        "services_quarantined": len(quarantined),
        "duplicates_removed": len(dropped),
        "agencies": sorted({r.agency_en or r.agency_ar for r in unique}),
        "page_labels_en": dict(Counter(t for r in unique for t in r.extra.get("page_tags", {}).get("en", []))),
        "source_domains": sorted({r.source_domain for r in unique}),
        "with_arabic": sum(bool(r.title_ar) for r in unique),
        "with_english": sum(bool(r.title_en) for r in unique),
        "field_coverage": {
            f: sum(bool(getattr(r, f)) for r in unique)
            for f in schema_columns()
            if f.endswith(("_en", "_ar")) and not f.startswith(("official_url", "agency"))
        },
        "warnings": dict(warnings),
        "date_collected_range": [
            min((r.date_collected for r in unique if r.date_collected), default=None),
            max((r.date_collected for r in unique if r.date_collected), default=None),
        ],
    }

    if write:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.SERVICES_JSONL, "w", encoding="utf-8") as f:
            for rec in unique:
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        with open(config.QUARANTINE_JSONL, "w", encoding="utf-8") as f:
            for q in quarantined:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        flat = pd.DataFrame([{**r.to_dict(), "extra": json.dumps(r.extra, ensure_ascii=False)} for r in unique],
                            columns=schema_columns())
        flat.to_csv(config.SERVICES_CSV, index=False, encoding="utf-8-sig")
        config.BUILD_REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def load_services() -> dict[str, ServiceRecord]:
    """Read the processed knowledge base (keyed by service_id)."""
    out = {}
    with open(config.SERVICES_JSONL, encoding="utf-8") as f:
        for line in f:
            rec = ServiceRecord.from_dict(json.loads(line))
            out[rec.service_id] = rec
    return out


if __name__ == "__main__":
    rep = run()
    print(json.dumps({k: rep[k] for k in ("services_indexed", "services_quarantined", "duplicates_removed",
                                          "with_arabic", "with_english", "sources")}, ensure_ascii=False, indent=2))
