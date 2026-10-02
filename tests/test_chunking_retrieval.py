import numpy as np
import pytest

from src.answering.composer import compose, query_language
from src.indexing.chunking import build_chunks, chunks_fingerprint, record_to_chunks
from src.retrieval.retriever import Retriever, StaleIndexError
from tests.conftest import fake_encode, make_record


def test_chunks_only_for_present_sections_and_keep_metadata(records):
    assert {c["section"] for c in record_to_chunks(records[0], include_title=False)} == {
        "overview", "steps", "service_facts"}
    chunks = record_to_chunks(records[0])
    sections = {(c["lang"], c["section"]) for c in chunks}
    assert sections == {("en", "title"), ("en", "overview"), ("en", "steps"), ("en", "service_facts"),
                        ("ar", "title"), ("ar", "overview")}
    for c in chunks:
        assert c["service_id"] == "mc-1"
        assert c["official_url"].startswith("https://mc.gov.sa/")
        assert c["title"] in c["embed_text"]
    facts = next(c for c in chunks if c["section"] == "service_facts")
    assert facts["text"] == "Fees: 200 riyals"


def test_fingerprint_is_deterministic_and_sensitive(records):
    a = build_chunks(records)
    b = build_chunks(records)
    assert chunks_fingerprint(a) == chunks_fingerprint(b)
    records[0].description_en += " changed"
    assert chunks_fingerprint(build_chunks(records)) != chunks_fingerprint(a)


@pytest.fixture
def retriever(records, fake_embedder):
    chunks = build_chunks(records)
    return Retriever(chunks, fake_encode(chunks["embed_text"].tolist()))


def test_search_returns_each_service_once(retriever):
    res = retriever.search("reserve a trade name", method="dense", top_k=5)
    ids = [h.service_id for h in res.hits]
    assert len(ids) == len(set(ids)) == 3
    assert ids[0] == "mc-1"
    assert res.hits[0].evidence and res.hits[0].evidence[0].score == pytest.approx(res.hits[0].score)


def test_lexical_matches_arabic_morphology(retriever):
    res = retriever.search("تأسيس الشركات ذات المسؤولية المحدودة", method="lexical")
    assert res.hits[0].service_id == "mc-99"


def test_language_restriction(retriever):
    res = retriever.search("حجز اسم تجاري", method="dense", langs=("en",))
    assert all(e.lang == "en" for h in res.hits for e in h.evidence)


def test_hybrid_is_weighted_sum(retriever):
    d, lx = retriever.chunk_scores("trade name")
    res = retriever.search("trade name", method="hybrid", alpha=0.7, top_k=1)
    assert res.hits[0].score == pytest.approx(float(np.max(0.7 * d + 0.3 * lx)), rel=1e-5)


def test_mismatched_embeddings_rejected(records):
    chunks = build_chunks(records)
    with pytest.raises(StaleIndexError):
        Retriever(chunks, np.zeros((len(chunks) - 1, 8), dtype=np.float32))


def test_composer_refuses_below_threshold(retriever, records):
    recs = {r.service_id: r for r in records}
    res = retriever.search("best pizza in Riyadh", method="dense")
    ans = compose(res, recs, threshold=0.99)
    assert ans.status == "insufficient" and ans.primary is None
    assert "couldn't find enough information" in ans.message


def test_composer_shows_only_official_fields_with_citation(retriever, records):
    recs = {r.service_id: r for r in records}
    ans = compose(retriever.search("reserve a trade name", method="dense"), recs, threshold=0.0)
    assert ans.status == "answered"
    p = ans.primary
    assert p.service_id == "mc-1"
    assert p.official_url == records[0].official_url_en            # citation preserved
    names = [f[0] for f in p.fields]
    assert names == ["description", "steps", "fees"]          # nothing invented
    assert all(text for _, _, text in p.fields)


def test_composer_arabic_query_prefers_official_arabic_text(retriever, records):
    recs = {r.service_id: r for r in records}
    ans = compose(retriever.search("حجز اسم تجاري", method="dense"), recs, threshold=0.0)
    assert ans.ui_lang == "ar" and ans.primary.shown_lang == "ar"
    assert ans.primary.title == "حجز اسم تجاري" and ans.primary.language_note is None


def test_composer_flags_english_only_service_for_arabic_user(retriever, records):
    recs = {r.service_id: r for r in records}
    res = retriever.search("المنتجات المعيبة Defective Products", method="dense")
    res.query = "المنتجات المعيبة"
    res.hits = [h for h in res.hits if h.service_id == "mc-30"]
    ans = compose(res, recs, threshold=0.0)
    assert ans.primary.shown_lang == "en" and "دون ترجمة آلية" in ans.primary.language_note


def test_query_language():
    assert query_language("كيف أسجل LLC؟") == "ar"
    assert query_language("How to register LLC") == "en"


def test_empty_query(retriever, records):
    res = retriever.search("   ", method="dense")
    assert compose(res, {r.service_id: r for r in records}, threshold=0.3).status == "empty"


def test_record_languages():
    assert make_record(title_ar="x").languages() == ["en", "ar"]
