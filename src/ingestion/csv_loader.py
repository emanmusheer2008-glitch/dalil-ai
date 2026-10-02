"""CSV loaders.

* :class:`CanonicalCSVLoader` reads a CSV whose columns already follow the
  canonical schema (e.g. ``data/processed/services.csv`` or a curated file).
* :class:`LegacySeedCSVLoader` reads the original 18-row development seed
  (``data/raw/seed/services_seed_v0.csv``). Those rows are hand-written
  summaries whose text was never captured from an official page, so they are
  marked ``unverified`` and are quarantined by validation.
"""
from __future__ import annotations

import pandas as pd

from src.ingestion.base import BaseLoader
from src.ingestion.normalization import domain_of
from src.schema import ServiceRecord, schema_columns


class CanonicalCSVLoader(BaseLoader):
    name = "canonical_csv"

    def load(self) -> list[ServiceRecord]:
        df = pd.read_csv(self.path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        unknown = set(df.columns) - set(schema_columns())
        if unknown:
            raise ValueError(f"{self.path}: unknown columns {sorted(unknown)}")
        return [ServiceRecord.from_dict(row) for row in df.to_dict(orient="records")]


class LegacySeedCSVLoader(BaseLoader):
    name = "legacy_seed_csv"

    NOTE = (
        "Development seed record written by hand during project setup. The "
        "description is a paraphrased summary, not text captured from the "
        "official page, and the page could not be re-checked because "
        "my.gov.sa blocks automated access (HTTP 403 / Cloudflare). "
        "Quarantined: excluded from the search index."
    )

    def load(self) -> list[ServiceRecord]:
        df = pd.read_csv(self.path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        records = []
        for row in df.to_dict(orient="records"):
            url = row.get("official_url") or None
            records.append(
                ServiceRecord(
                    service_id=f"seed-{row['service_id']}",
                    source_type="seed_manual_summary",
                    source_domain=domain_of(url),
                    verification_status="unverified",
                    title_en=row.get("title") or None,
                    agency_en=row.get("agency") or None,
                    category_en=row.get("category") or None,
                    description_en=row.get("description") or None,
                    official_url_en=url,
                    provenance_notes=self.NOTE,
                )
            )
        return records
