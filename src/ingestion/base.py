"""Common interface for every ingestion source.

A loader turns one source (a file, a folder, a dataset) into a list of
:class:`~src.schema.ServiceRecord` objects. Loaders never invent values: if a
source does not state a field, the field stays ``None``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.schema import ServiceRecord


class BaseLoader(ABC):
    #: short machine name, used in build reports
    name: str = "base"

    def __init__(self, path: Path | str):
        self.path = Path(path)

    @abstractmethod
    def load(self) -> list[ServiceRecord]:
        """Return canonical records from the source."""

    def describe(self) -> str:
        return f"{self.name}:{self.path}"
