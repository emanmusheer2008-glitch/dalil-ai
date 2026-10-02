"""Shared fixtures.

Unit tests use a tiny deterministic *fake* embedder so they run in seconds
without downloading the real model. Integration tests (tests/test_integration.py)
use the real multilingual model and the real processed data, and are skipped
automatically if the model cannot be loaded.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schema import ServiceRecord  # noqa: E402
from src.utils.arabic import normalize_for_matching  # noqa: E402


def fake_encode(texts, model_path=None, show_progress=False):
    """Bag-of-hashed-character-trigrams embedding: deterministic, L2-normalised."""
    dim = 256
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        t = normalize_for_matching(t)
        for j in range(len(t) - 2):
            h = int(hashlib.md5(t[j : j + 3].encode("utf-8")).hexdigest(), 16) % dim
            out[i, h] += 1.0
        n = np.linalg.norm(out[i])
        if n:
            out[i] /= n
    return out


@pytest.fixture
def fake_embedder(monkeypatch):
    from src.retrieval import embedder

    calls = {"n": 0}

    def _enc(texts, model_path=None, show_progress=False):
        calls["n"] += 1
        return fake_encode(texts)

    monkeypatch.setattr(embedder, "encode", _enc)
    return calls


def make_record(sid="t-1", **kw) -> ServiceRecord:
    base = dict(
        service_id=sid,
        source_type="official_web_page",
        source_domain="mc.gov.sa",
        verification_status="verified_official_capture",
        date_collected="2026-09-30T10:00:00Z",
        source_sha256="en:abc",
        title_en="Trade Name Reservation",
        agency_en="Ministry of Commerce",
        description_en="An electronic service that allows reserving a trade name for a new business.",
        official_url_en=f"https://mc.gov.sa/en/eservices/Pages/ServiceDetails.aspx?sID={sid}",
    )
    base.update(kw)
    return ServiceRecord(**base)


@pytest.fixture
def records():
    return [
        make_record(
            "mc-1",
            title_ar="حجز اسم تجاري",
            agency_ar="وزارة التجارة",
            description_ar="خدمة إلكترونية تتيح حجز اسم تجاري للمنشأة الجديدة.",
            official_url_ar="https://mc.gov.sa/ar/eservices/Pages/ServiceDetails.aspx?sID=1",
            fees_en="200 riyals",
            steps_en="- Log in\n- Choose the name\n- Pay the fees",
        ),
        make_record(
            "mc-99",
            title_en="Establish a limited liability company",
            description_en="An electronic service to establish a limited liability company online.",
            official_url_en="https://mc.gov.sa/en/eservices/Pages/ServiceDetails.aspx?sID=99",
            title_ar="تأسيس شركة ذات مسؤولية محدودة",
            agency_ar="وزارة التجارة",
            description_ar="خدمة إلكترونية لتأسيس شركة ذات مسؤولية محدودة.",
            official_url_ar="https://mc.gov.sa/ar/eservices/Pages/ServiceDetails.aspx?sID=99",
        ),
        make_record(
            "mc-30",
            title_en="Defective Products Inquiry",
            description_en="Check recalled and defective consumer products.",
            official_url_en="https://mc.gov.sa/en/eservices/Pages/ServiceDetails.aspx?sID=30",
        ),
    ]
