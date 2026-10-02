"""Lightweight redaction of obvious personal identifiers before text is sent to Gemini.

Deliberately simple (regex): Saudi national ID / iqama numbers (10 digits starting 1 or 2),
card numbers (13-19 digits, spaces/dashes allowed), IBANs, passport-like codes, and OTP-style
codes next to an OTP keyword. Arabic-Indic digits are handled. Not a full PII detector.
"""
from __future__ import annotations

import re

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

_RULES = [
    ("[IBAN]", re.compile(r"\bSA\d{2}(?:[ ]?[0-9A-Z]{4}){5}\b", re.I)),
    ("[CARD_NUMBER]", re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")),
    ("[ID_NUMBER]", re.compile(r"(?<!\d)[12]\d{9}(?!\d)")),
    ("[OTP]", re.compile(r"(?i)(?:otp|one[- ]time|verification code|code|رمز(?: التحقق)?|كود)\W{0,3}\d{4,8}\b")),
    ("[PASSPORT_NUMBER]", re.compile(r"(?i)(?:passport|جواز)(?:\s*(?:no\.?|number|رقم))?\W{0,3}[A-Z]{1,2}\d{6,8}\b")),
]


def redact(text: str) -> tuple[str, int]:
    """Return (redacted_text, number_of_redactions)."""
    t = (text or "").translate(_DIGITS)
    n = 0
    for label, rx in _RULES:
        t, k = rx.subn(label, t)
        n += k
    return (t if n else text), n
