"""Arabic text utilities.

These functions are used for *matching* (lexical retrieval, duplicate
detection, language detection). They are never applied to text that is shown
to the user as official source text — displayed text is always the verbatim
captured string.
"""
from __future__ import annotations

import re
import unicodedata

# Arabic diacritics (tashkeel) + Quranic annotation marks + superscript alef.
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_ZERO_WIDTH = re.compile(r"[​‌‍‎‏﻿]")
_ARABIC_CHAR = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
_LATIN_CHAR = re.compile(r"[A-Za-z]")

# Arabic-Indic and Eastern Arabic-Indic digits -> ASCII digits.
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def clean_unicode(text: str) -> str:
    """NFC-normalise, drop zero-width/bidi marks, turn NBSP into spaces."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text)
    return text.replace(" ", " ")


def normalize_arabic(text: str) -> str:
    """Light, standard orthographic normalisation for Arabic *matching*.

    - remove diacritics and tatweel
    - unify alef variants (أ إ آ ٱ -> ا)
    - alef maqsura (ى) -> ya (ي)
    - taa marbuta (ة) -> ha (ه)
    - hamza on waw/ya (ؤ ئ) -> ء
    - Arabic-Indic digits -> ASCII
    """
    text = clean_unicode(text)
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    text = re.sub("[أإآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = text.replace("ؤ", "ء").replace("ئ", "ء")
    return text.translate(_DIGITS)


def normalize_for_matching(text: str) -> str:
    """Lower-case Latin text, normalise Arabic, collapse whitespace."""
    text = normalize_arabic(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def arabic_ratio(text: str) -> float:
    """Share of alphabetic characters that are Arabic script (0..1)."""
    ar = len(_ARABIC_CHAR.findall(text or ""))
    la = len(_LATIN_CHAR.findall(text or ""))
    return ar / (ar + la) if (ar + la) else 0.0


def detect_language(text: str) -> str:
    """Return 'ar', 'en' or 'mixed' using script proportions."""
    r = arabic_ratio(text)
    if r >= 0.7:
        return "ar"
    if r <= 0.3:
        return "en"
    return "mixed"
