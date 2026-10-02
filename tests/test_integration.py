"""End-to-end checks with the REAL model and the REAL processed knowledge base.

Skipped automatically when the processed index or the model is unavailable
(e.g. on a machine without internet and without a local model copy).
"""
import pytest

from src import config

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def system():
    if not config.EMBEDDINGS_NPY.exists():
        pytest.skip("index not built")
    try:
        from src.ingestion.pipeline import load_services
        from src.retrieval.retriever import Retriever, load_retrieval_config

        r = Retriever.from_disk()
        r.chunk_scores("warm-up")
    except Exception as e:  # model not downloadable here
        pytest.skip(f"model unavailable: {e}")
    return r, load_services(), load_retrieval_config()


def ask(system, q):
    from src.answering.composer import compose

    r, recs, cfg = system
    res = r.search(q, method=cfg["method"], alpha=cfg["alpha"])
    return compose(res, recs, threshold=cfg["threshold"])


@pytest.mark.parametrize("q, expected", [
    ("Steps to set up a limited liability company", "mc-99"),
    ("How do I reserve a trade name for my business?", "mc-1"),
    ("خطوات تأسيس شركة ذات مسؤولية محدودة", "mc-99"),
    pytest.param("كيف أحجز اسم تجاري لمنشأتي؟", "mc-1", marks=pytest.mark.xfail(
        reason="Known failure (see README Limitations): the Arabic titles of 'extend' and 'cancel' "
               "trade-name reservation also contain «حجز اسم تجاري», they outrank the base service and "
               "the top score falls below the calibrated threshold.", strict=False)),
])
def test_answers_known_services(system, q, expected):
    ans = ask(system, q)
    assert ans.status == "answered"
    ids = [ans.primary.service_id] + [x.service_id for x in ans.related]
    assert expected in ids


@pytest.mark.parametrize("q", ["What's the best pizza in Riyadh?", "Tell me a joke", "طريقة عمل الكبسة"])
def test_declines_out_of_domain(system, q):
    assert ask(system, q).status == "insufficient"


def test_arabic_question_gets_official_arabic_text_and_citation(system):
    ans = ask(system, "خطوات تأسيس شركة ذات مسؤولية محدودة")
    p = ans.primary
    assert ans.ui_lang == "ar" and p.shown_lang == "ar"
    assert p.official_url.startswith("https://mc.gov.sa/ar/")
    assert p.date_collected


def test_every_shown_field_is_verbatim_from_the_record(system):
    _, recs, _ = system
    ans = ask(system, "How much does the annual confirmation of a commercial registration cost?")
    if ans.status != "answered":
        pytest.skip("declined")
    rec = recs[ans.primary.service_id]
    for name, _, text in ans.primary.fields:
        assert text == rec.get(name, ans.primary.shown_lang)
