"""Adapter for official open-data files (e.g. from open.data.gov.sa).

Status: **ready but unused**. At the time of writing no official open dataset
listing individual government services (with descriptions, requirements, etc.)
had been obtained. The adapter exists so that, once such a file is
legitimately downloaded, it can be added without new code:

    data/raw/open_data/<dataset>.csv        (the file as downloaded)
    data/raw/open_data/<dataset>.map.json   (column mapping + provenance)

Example ``.map.json``::

    {
      "dataset_url": "https://open.data.gov.sa/en/datasets/view/<id>",
      "license": "Open Data License (as stated on the dataset page)",
      "downloaded_at": "2026-10-01T09:00:00Z",
      "id_column": "ServiceID",
      "columns": {"title_en": "ServiceNameEn", "title_ar": "ServiceNameAr",
                   "agency_en": "EntityEn", "description_en": "DescEn",
                   "official_url_en": "Link"}
    }

Only mapped columns are used; nothing is inferred. No credentials are stored
in the repository — if an official API later requires a key, read it from an
environment variable (never commit it).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.ingestion.base import BaseLoader
from src.ingestion.normalization import domain_of
from src.schema import BILINGUAL_FIELDS, ServiceRecord

ALLOWED_TARGETS = {f"{f}_{lang}" for f in BILINGUAL_FIELDS for lang in ("en", "ar")} | {
    "contact_information",
    "service_channel",
}


class OpenDataFileLoader(BaseLoader):
    name = "official_open_data"

    def load(self) -> list[ServiceRecord]:
        records: list[ServiceRecord] = []
        for map_path in sorted(Path(self.path).glob("*.map.json")):
            mapping = json.loads(map_path.read_text(encoding="utf-8"))
            stem = map_path.name[: -len(".map.json")]
            data_path = next(
                (p for p in (map_path.with_name(stem + e) for e in (".csv", ".xlsx", ".json")) if p.exists()),
                None,
            )
            if data_path is None:
                continue
            bad = set(mapping["columns"]) - ALLOWED_TARGETS
            if bad:
                raise ValueError(f"{map_path}: unknown target fields {sorted(bad)}")
            if data_path.suffix == ".csv":
                df = pd.read_csv(data_path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
            elif data_path.suffix == ".xlsx":
                df = pd.read_excel(data_path, dtype=str).fillna("")
            else:
                df = pd.read_json(data_path, dtype=str).fillna("")
            digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
            for row in df.to_dict(orient="records"):
                rec = ServiceRecord(
                    service_id=f"od-{stem}-{row[mapping['id_column']]}",
                    source_type="official_open_data",
                    source_domain=domain_of(mapping["dataset_url"]),
                    verification_status="verified_official_capture",
                    date_collected=mapping.get("downloaded_at"),
                    source_sha256=f"file:{digest}",
                    provenance_notes=f"Official open dataset {mapping['dataset_url']} ({mapping.get('license', 'license not recorded')}).",
                )
                for target, column in mapping["columns"].items():
                    value = str(row.get(column, "")).strip()
                    setattr(rec, target, value or None)
                records.append(rec)
        return records
