from src.utils.arabic import (
    arabic_ratio,
    clean_unicode,
    detect_language,
    normalize_arabic,
    normalize_for_matching,
)


def test_alef_and_taa_marbuta_are_unified():
    assert normalize_arabic("إصدار أمر آخر") == "اصدار امر اخر"
    assert normalize_arabic("شركة") == "شركه"
    assert normalize_arabic("مستشفى") == "مستشفي"


def test_diacritics_and_tatweel_removed():
    assert normalize_arabic("تِجَارَة") == "تجاره"
    assert normalize_arabic("تـــجارة") == "تجاره"


def test_arabic_indic_digits_become_ascii():
    assert normalize_arabic("٥٠٠ ريال") == "500 ريال"


def test_zero_width_and_nbsp_removed():
    assert clean_unicode("abc​def x") == "abcdef x"


def test_matching_normalisation_is_case_and_punctuation_insensitive():
    assert normalize_for_matching("Trade-Name   Reservation!") == "trade name reservation"
    assert normalize_for_matching("حجز  الاسم، التجاري؟") == "حجز الاسم التجاري"


def test_language_detection():
    assert detect_language("كيف أجدد السجل التجاري؟") == "ar"
    assert detect_language("How do I renew my commercial registration?") == "en"
    assert detect_language("تجديد الوكالة commercial") == "mixed"
    assert 0.0 <= arabic_ratio("abc") <= 1.0
    assert arabic_ratio("") == 0.0


def test_utf8_roundtrip(tmp_path):
    text = "خدمة إلكترونية تقدمها وزارة التجارة — ٣٠ يومًا"
    p = tmp_path / "ar.txt"
    p.write_text(text, encoding="utf-8")
    assert p.read_text(encoding="utf-8") == text


def test_tanween_accusative_alef_is_normalised_only_when_tanween_is_typed():
    from src.utils.arabic import normalize_arabic
    assert normalize_arabic("اسماً تجارياً") == "اسم تجاري"
    assert normalize_arabic("اسمًا") == "اسم"
    assert normalize_arabic("كتابا") == "كتابا"      # bare final alef: ambiguous, left alone
    assert normalize_arabic("ماءً") == "ماء"
