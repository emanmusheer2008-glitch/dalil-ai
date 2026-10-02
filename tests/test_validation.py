from src.ingestion.normalization import clean_text, domain_of, normalize_record
from src.ingestion.validation import find_duplicates, is_official_domain, validate_record
from tests.conftest import make_record


def test_valid_record_passes():
    assert validate_record(make_record()).ok


def test_missing_title_is_blocking():
    res = validate_record(make_record(title_en=None))
    assert not res.ok and any("title" in e for e in res.errors)


def test_missing_url_is_blocking():
    res = validate_record(make_record(official_url_en=None))
    assert not res.ok


def test_non_https_url_rejected():
    res = validate_record(make_record(official_url_en="http://mc.gov.sa/x"))
    assert any("invalid official_url_en" in e for e in res.errors)


def test_non_official_domain_rejected():
    rec = make_record(source_domain="example.com", official_url_en="https://example.com/x")
    assert not validate_record(rec).ok
    assert is_official_domain("mc.gov.sa") and is_official_domain("my.gov.sa")
    assert not is_official_domain("gov.sa.example.com")


def test_unverified_records_are_quarantined():
    res = validate_record(make_record(verification_status="unverified", source_type="seed_manual_summary"))
    assert not res.ok and any("not indexable" in e for e in res.errors)


def test_verified_record_needs_provenance():
    res = validate_record(make_record(source_sha256=None, date_collected=None))
    assert "verified record without source_sha256" in res.errors
    assert "verified record without date_collected" in res.errors


def test_language_mismatch_warns():
    res = validate_record(make_record(title_ar="Trade Name Reservation"))
    assert any("title_ar" in w for w in res.warnings)


def test_duplicates_detected_by_url_and_title():
    a = make_record("a", official_url_en="https://www.mc.gov.sa/en/x")
    b = make_record("b", official_url_en="https://mc.gov.sa/en/x/")        # same URL
    c = make_record("c", official_url_en="https://mc.gov.sa/en/other")    # same agency+title
    d = make_record("d", title_en="Something else", official_url_en="https://mc.gov.sa/en/d")
    unique, dropped = find_duplicates([a, b, c, d])
    assert [r.service_id for r in unique] == ["a", "d"]
    assert {(r.service_id, kept) for r, kept in dropped} == {("b", "a"), ("c", "a")}


def test_normalisation_keeps_wording_and_nulls_empty():
    assert clean_text("  Pay  the​ fees \n\n next ") == "Pay the fees\nnext"
    assert clean_text("   ") is None
    rec = normalize_record(make_record(source_domain=None, fees_en="  "))
    assert rec.fees_en is None
    assert rec.source_domain == "mc.gov.sa"
    assert domain_of("https://www.moe.gov.sa/en/a") == "moe.gov.sa"
