"""V2: lexicon expansion, BM25, title coverage, synthesis grounding, decisions."""
import numpy as np
import pandas as pd
import pytest

from src.answering.synthesizer import detect_intent, official_texts, split_items, synthesize
from src.evaluation.evaluate_v2 import decision_metrics, fit_thresholds
from src.indexing.chunking import build_chunks
from src.retrieval.bm25 import BM25Index, tokenize
from src.retrieval.lexicon import expand_query, expansions
from src.retrieval.retriever import Retriever
from tests.conftest import fake_encode, make_record


# ---------------------------------------------------------------- lexicon
def test_lexicon_adds_official_terms_without_removing_words():
    q = "I want to book a business name"
    x = expand_query(q)
    assert x.startswith(q) and "trade name" in x and "reserve" in x
    assert "اسم تجاري" in " ".join(expansions("ابي احجز اسم لمحلي"))


def test_lexicon_is_word_bounded():
    assert expansions("the crane operator") == []          # "cr" must not match inside "crane"
    assert "commercial registration" in expansions("renew my CR")


# ------------------------------------------------------------------- bm25
def test_bm25_tokenizer_light_stemming():
    assert tokenize("Registering companies") == ["register", "compani"]
    assert tokenize("والسجلات التجارية") == ["سجل", "تجاري"] or tokenize("والسجلات التجارية")[0].startswith("سجل")
    assert "كيف" not in tokenize("كيف احجز")


def test_bm25_prefers_matching_document():
    idx = BM25Index(["trade name reservation service", "vat registration for businesses", "divorce notarization"])
    s = idx.scores("reserve a trade name")
    assert s.argmax() == 0 and s[2] == 0


# ------------------------------------------------------- title coverage
@pytest.fixture
def siblings(fake_embedder):
    recs = [
        make_record("mc-1", title_en="Trade Name Reservation", description_en="Reserve a trade name for a business."),
        make_record("mc-34", title_en="Extension of a trade name reservation",
                    description_en="Extend an existing trade name reservation."),
    ]
    chunks = build_chunks(recs)
    return recs, Retriever(chunks, fake_encode(chunks["embed_text"].tolist()))


def test_title_coverage_prefers_service_without_extra_title_words(siblings):
    recs, r = siblings
    sig = r.signals("reserve trade name", expand=True)
    cov = dict(zip(r.service_ids, r.title_coverage(sig["qtokens"])))
    assert cov["mc-1"] > cov["mc-34"]
    res = r.search("How can I reserve a trade name?", method="v2",
                   params={"w_dense": 0.4, "w_char": 0.15, "w_bm25": 0.3, "w_tcov": 0.2})
    assert res.hits[0].service_id == "mc-1"
    assert res.expanded_query and "reserve" in res.expanded_query


def test_query_embedding_is_cached(siblings, fake_embedder):
    _, r = siblings
    n0 = fake_embedder["n"]
    r.signals("same question")
    r.signals("same question")
    assert fake_embedder["n"] == n0 + 1


# -------------------------------------------------------------- synthesis
@pytest.fixture
def kb(fake_embedder):
    recs = [
        make_record("mc-1", title_ar="حجز اسم تجاري", agency_ar="وزارة التجارة",
                    description_ar="خدمة إلكترونية لحجز اسم تجاري.",
                    official_url_ar="https://mc.gov.sa/ar/x?sID=1",
                    description_en="An electronic service to reserve a trade name. It is valid for 60 days.",
                    steps_en="- Log in\n- Enter the name\n- Pay the fees", fees_en="200 riyals",
                    requirements_en="- The applicant must be 18 or older", processing_time_en="Immediate"),
        make_record("hrsd-9", title_en="Domestic worker visa", agency_en="Ministry of Human Resources",
                    source_domain="hrsd.gov.sa", official_url_en="https://hrsd.gov.sa/x",
                    description_en="Issue a visa for a domestic worker.", fees_en="2000 riyals"),
    ]
    chunks = build_chunks(recs)
    return {r.service_id: r for r in recs}, Retriever(chunks, fake_encode(chunks["embed_text"].tolist()))


def all_official_strings(rec):
    out = []
    for f in ("description", "requirements", "eligibility", "required_documents", "steps", "fees",
              "processing_time", "target_audience", "service_languages", "notes"):
        for lang in ("en", "ar"):
            v = rec.get(f, lang)
            if v:
                out.extend([v] + split_items(v))
    if rec.contact_information:
        out.append(rec.contact_information)
    return out


def test_every_fact_in_answer_is_verbatim_official_text(kb):
    recs, r = kb
    res = r.search("how much does it cost to reserve a trade name", method="v2", top_k=3)
    ans = synthesize(res, recs, t_answer=0.0, t_tentative=0.0)
    assert ans.status == "answered"
    allowed = set()
    for c in ans.citations:
        allowed.update(all_official_strings(recs[c.service_id]))
    for text in official_texts(ans):
        assert text in allowed, text
    # every point cites a real source
    ns = {c.n for c in ans.citations}
    assert all(p.cite in ns for s in ans.sections for p in s.points)


def test_intent_puts_fees_first_and_omits_missing_sections(kb):
    recs, r = kb
    res = r.search("trade name reservation fees", method="v2", top_k=1)
    ans = synthesize(res, recs, t_answer=0.0, t_tentative=0.0)
    assert ans.intent == "fees"
    keys = [sec.key for sec in ans.sections]
    assert keys[0] == "direct" and keys[1] == "fees"            # intent section comes right after the answer
    assert any(p.text == "200 riyals" for p in ans.sections[1].points)
    assert not any(p.text == "200 riyals" for p in ans.sections[0].points)   # not shown twice
    assert "docs" not in {s.key for s in ans.sections}          # source has no documents -> no heading
    steps = next(s for s in ans.sections if s.key == "steps")
    assert steps.ordered and [p.text for p in steps.points] == ["Log in", "Enter the name", "Pay the fees"]


def test_three_way_decision(kb):
    recs, r = kb
    res = r.search("reserve a trade name", method="v2", top_k=2)
    top = res.hits[0].score
    assert synthesize(res, recs, t_answer=top - 0.01, t_tentative=0.0).status == "answered"
    tent = synthesize(res, recs, t_answer=top + 0.5, t_tentative=top - 0.01)
    assert tent.status == "tentative" and "not certain" in tent.lead
    assert synthesize(res, recs, t_answer=top + 0.5, t_tentative=top + 0.4).status == "insufficient"


def test_arabic_answer_uses_official_arabic_or_flags_language(kb):
    recs, r = kb
    res = r.search("حجز اسم تجاري", method="v2", top_k=2)
    ans = synthesize(res, recs, t_answer=0.0, t_tentative=0.0, support_margin=1.0)
    assert ans.ui_lang == "ar"
    c1 = ans.citations[0]
    assert c1.service_id == "mc-1" and c1.lang == "ar" and c1.url.startswith("https://mc.gov.sa/ar/")
    en_only = [c for c in ans.citations if c.lang == "en"]
    assert all(any(f"[{c.n}]" in n for n in ans.notes) for c in en_only)


def test_conflicting_fees_across_agencies_are_surfaced(kb):
    recs, r = kb
    res = r.search("fees", method="v2", top_k=2)
    ans = synthesize(res, recs, t_answer=0.0, t_tentative=0.0, support_margin=10)
    if len({c.agency for c in ans.citations}) > 1:
        assert any("different values" in n for n in ans.notes)


def test_detect_intent():
    assert detect_intent("كم رسوم تجديد السجل؟") == "fees"
    assert detect_intent("What documents do I need for a visa?") == "documents"
    assert detect_intent("how long does it take") == "time"
    assert detect_intent("شروط الاستقدام") == "requirements"


# -------------------------------------------------------------- decisions
def test_threshold_fitting_respects_targets():
    df = pd.DataFrame({
        "top_score": [0.9, 0.8, 0.7, 0.6, 0.35, 0.3, 0.25, 0.5],
        "supported": [1, 1, 1, 1, 0, 0, 0, 0],
        "rank": [1, 1, 2, 1, None, None, None, None],
    })
    t_ans, t_tent = fit_thresholds(df, min_unsup_refusal=0.75, min_conf_precision=0.9)
    m = decision_metrics(df, t_ans, t_tent)
    assert m["unsupported_refusal_rate"] >= 0.75
    assert m["false_refusal_rate"] == 0.0
    assert t_ans >= t_tent
