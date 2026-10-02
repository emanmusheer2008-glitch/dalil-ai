"""End-to-end checks with the REAL model and the REAL processed knowledge base (V2).

They go through ``src.engine.Dalil`` -- the same entry point as the app -- so they
use the calibrated settings in ``data/processed/retrieval_config_v2.json``.

Skipped automatically when the processed index or the model is unavailable
(e.g. on a machine without internet and without a local model copy).

None of these questions is special-cased anywhere in the code; several are
benchmark questions, so they document behaviour rather than measure it (the
measurement is ``python -m src.evaluation.evaluate_v2`` on the held-out split).
"""
import re

import pytest

from src import config

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def dalil():
    if not config.EMBEDDINGS_NPY.exists():
        pytest.skip("index not built")
    try:
        from src.engine import Dalil

        d = Dalil()
        d.warm_up()
    except Exception as e:  # model not downloadable here
        pytest.skip(f"model unavailable: {e}")
    return d


SHOWN = ("answered", "tentative")


@pytest.mark.parametrize("q, expected", [
    ("Steps to set up a limited liability company", "mc-99"),
    ("How do I reserve a trade name for my business?", "mc-1"),
    ("How can I reserve a trade name?", "mc-1"),          # the V1 false refusal
    ("خطوات تأسيس شركة ذات مسؤولية محدودة", "mc-99"),
    ("كيف أحجز اسم تجاري لمنشأتي؟", "mc-1"),              # was a known xfail in V1
])
def test_finds_commerce_services(dalil, q, expected):
    ans = dalil.ask(q)
    assert ans.status in SHOWN
    assert ans.citations[0].service_id == expected


@pytest.mark.parametrize("q, source", [
    ("How do I register my company for VAT?", "zatca"),
    ("تجديد جواز السفر وأنا خارج المملكة", "mofa"),
    ("How do I issue a power of attorney online?", "moj"),
    ("إصدار رخصة بناء", "momah"),
])
def test_other_agencies(dalil, q, source):
    ans = dalil.ask(q)
    assert ans.status in SHOWN
    assert ans.citations[0].service_id.startswith(source + "-")


@pytest.mark.parametrize("q", ["What's the best pizza in Riyadh?", "Tell me a joke", "طريقة عمل الكبسة",
                               "Write a Python function to sort a list"])
def test_declines_out_of_domain(dalil, q):
    assert dalil.ask(q).status == "insufficient"


def test_empty_question(dalil):
    assert dalil.ask("   ").status == "empty"


def test_arabic_question_gets_official_arabic_text_and_citation(dalil):
    ans = dalil.ask("خطوات تأسيس شركة ذات مسؤولية محدودة")
    c = ans.citations[0]
    assert ans.ui_lang == "ar" and c.lang == "ar"
    assert c.url.startswith("https://mc.gov.sa/ar/")
    assert c.date_collected


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


@pytest.mark.parametrize("q", ["How much does the annual confirmation of a commercial registration cost?",
                               "ما خطوات إصدار تأشيرة عاملة منزلية؟", "How do I file my VAT return?"])
def test_every_answer_point_is_verbatim_official_text(dalil, q):
    """Grounding: each point Dalil shows must occur verbatim in the cited service's official fields."""
    ans = dalil.ask(q)
    if ans.status not in SHOWN:
        pytest.skip("declined")
    by_n = {c.n: c for c in ans.citations}
    for sec in ans.sections:
        for pt in sec.points:
            c = by_n[pt.cite]
            rec = dalil.records[c.service_id].to_dict()
            # every official text field of that record in the shown language (not URLs/ids/agency names)
            official = " ".join(_norm(str(v)) for k, v in rec.items()
                                if v and (k.endswith("_" + c.lang) or k == "contact_information")
                                and not k.startswith(("official_url", "agency")))
            assert _norm(pt.text) in official, (sec.key, pt.text[:80])


def test_every_citation_is_an_official_gov_sa_https_url(dalil):
    for q in ("How do I get a permit to perform Umrah?", "التسجيل في ضريبة القيمة المضافة"):
        for c in dalil.ask(q).citations:
            assert c.url.startswith("https://")
            host = re.sub(r"^https://([^/]+)/.*$", r"\1", c.url)
            assert host.endswith(".gov.sa"), host
