"""Load canonical records from JSON (a list) or JSON Lines files."""
from __future__ import annotations

import json

from src.ingestion.base import BaseLoader
from src.schema import ServiceRecord


class JSONLoader(BaseLoader):
    name = "canonical_json"

    def load(self) -> list[ServiceRecord]:
        text = self.path.read_text(encoding="utf-8-sig")
        if self.path.suffix == ".jsonl":
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            data = json.loads(text)
            rows = data["records"] if isinstance(data, dict) else data
        return [ServiceRecord.from_dict(r) for r in rows]
