"""Bilingual public-service terminology lexicon for query expansion.

Why this exists
---------------
Official pages use administrative terms ("trade name reservation",
"commercial registration", "إصدار"), while people write "book a business
name", "CR", "أطلع". Character n-grams cannot connect "business name" to
"trade name", and the small multilingual model compresses near-synonyms of
sibling services together. The lexicon adds the official term(s) to the query
when a common lay phrasing appears -- it never removes or rewrites the user's
words, and it never adds facts.

Rules
-----
* Entries are *general vocabulary* (synonyms, abbreviations, colloquial
  Arabic), not question-specific answers. There is no entry that maps a
  question to a service id.
* The lexicon was written from general knowledge of Saudi public-service
  vocabulary and the page titles of the indexed agencies. Its effect is
  measured on the development split; if it doesn't help it can be switched off
  (``expand=False``) and the evaluation reports both.
"""
from __future__ import annotations

import re

from src.utils.arabic import normalize_arabic

# (patterns that may appear in a query) -> official phrasing(s) appended.
# Patterns are matched on lower-cased, Arabic-normalised text, on word
# boundaries.
_EN = [
    (["business name", "company name", "shop name", "store name", "brand name for my business", "commercial name"],
     "trade name"),
    (["book", "booking", "hold", "secure"], "reserve reservation"),
    (["cr", "c.r", "commercial register", "business registration", "business license", "trade license"],
     "commercial registration"),
    (["llc", "l.l.c"], "limited liability company"),
    (["jsc"], "joint stock company"),
    (["sole trader", "one person business", "one-person business", "own business", "sole proprietor"],
     "sole proprietorship establishment"),
    (["open a company", "start a company", "set up a company", "form a company", "incorporate"],
     "establish company incorporation"),
    (["shut down", "close my business", "close my shop", "closing my business"], "cancel deletion write off"),
    (["renew", "renewal", "extend", "extension"], "renewal extension"),
    (["refund", "money back", "scam", "cheated", "bad product"], "complaint consumer"),
    (["tax number", "vat number", "tin"], "tax registration number"),
    (["vat"], "value added tax"),
    (["zakat"], "zakat"),
    (["import", "importing", "ship goods into"], "import customs"),
    (["export", "exporting"], "export customs"),
    (["job", "jobs", "worker", "workers", "employee", "employees", "staff", "maid", "housemaid", "driver"],
     "labor employment"),
    (["work visa", "recruit", "recruitment", "hire from abroad"], "visa recruitment labor"),
    (["iqama", "residence permit", "residency", "resident id", "resident identity"], "residence iqama resident identity"),
    # V4.1 (Absher coverage): general MOI vocabulary
    (["driving licence", "driving license", "driver license", "driver's license", "drivers license", "driving permit"],
     "driving license"),
    (["traffic fine", "traffic fines", "traffic ticket", "traffic violation", "traffic violations", "speeding ticket",
      "saher"], "traffic violations"),
    (["istimara", "istemarah", "vehicle registration", "car registration"], "vehicle registration"),
    (["exit re-entry", "exit reentry", "re-entry visa", "reentry visa", "re entry visa"], "exit re-entry visa"),
    (["final exit"], "final exit visa"),
    (["passport"], "passport"),
    (["transfer sponsorship", "change sponsor", "sponsor transfer", "kafala"], "transfer services employer"),
    (["wage", "salary", "salaries", "unpaid salary"], "wage protection"),
    (["disabled", "disability", "special needs"], "persons with disabilities"),
    (["building permit", "construction permit", "build a house"], "building permit"),
    (["shop license", "store license", "municipal license", "balady", "baladi"], "commercial activity license municipal"),
    (["pilgrim", "pilgrimage"], "hajj umrah"),
    (["rawdah", "rawda", "prophet's mosque"], "al rawdah al sharifah"),
    (["power of attorney", "poa"], "power of attorney"),
    (["divorce", "marriage", "custody", "alimony", "inheritance"], "family personal status"),
    (["school transfer", "move school", "change school"], "transferring student school"),
    (["certificate equivalency", "degree recognition", "recognize my degree", "recognise my degree"],
     "equivalency certificates"),
    (["medicine", "drug", "drugs", "cosmetics", "cosmetic", "medical device"], "food drug registration"),
    (["how much", "cost", "price", "charge", "charges"], "fees"),
    (["how long", "processing time", "time does it take", "duration"], "duration of service"),
    (["papers", "paperwork", "what do i need to bring"], "required documents"),
    (["who can apply", "eligible", "eligibility", "qualify"], "conditions eligibility"),
]

_AR = [
    (["اسم محل", "اسم المحل", "اسم محلي", "اسم لمحلي", "اسم لمحل", "اسم شركه", "اسم الشركه", "اسم شركتي", "اسم لشركتي",
      "اسم مءسستي", "اسم لمءسستي", "اسم لمشروعي", "اسم مشروع", "اسم نشاط", "اسم تجاري"], "اسم تجاري"),
    (["احجز", "حجز", "اسجل اسم"], "حجز"),
    (["سجل", "السجل"], "السجل التجاري"),
    (["ذات مسءوليه محدوده", "llc"], "شركه ذات مسءوليه محدوده"),
    (["مءسسه فرديه", "مءسستي"], "مءسسه فرديه"),
    (["اطلع", "استخرج", "اسوي", "اسوي لي"], "اصدار"),
    (["اقفل", "اسكر", "اغلق", "الغي"], "شطب الغاء"),
    (["اجدد", "تجديد", "امدد", "تمديد"], "تجديد تمديد"),
    (["فلوسي", "استرجاع", "نصب", "احتيال", "غش"], "شكوي"),
    (["ضريبه القيمه المضافه", "الضريبه", "الفاتوره"], "ضريبه القيمه المضافه"),
    (["عامل", "عماله", "خادمه", "سايق", "موظف", "موظفين"], "العمل العماله"),
    (["اقامه", "الاقامه", "هويه مقيم"], "الاقامه هويه مقيم"),
    # V4.1 (Absher coverage): general MOI vocabulary
    (["رخصة القيادة", "رخصة قيادة", "رخصة السواقة", "ليسن"], "رخصة القيادة"),
    (["مخالفة مرورية", "مخالفات مرورية", "المخالفات المرورية", "مخالفات المرور", "ساهر", "مخالفاتي"],
     "المخالفات المرورية"),
    (["استمارة", "الاستمارة", "رخصة سير", "رخصة السير"], "رخصة سير"),
    (["خروج وعودة", "خروج و عودة"], "تأشيرة خروج وعودة"),
    (["خروج نهائي"], "تأشيرة الخروج النهائي"),
    (["فقدت", "ضاع", "ضاعت", "ضايع", "مفقود"], "بدل مفقود فقدان"),
    (["جواز", "الجواز"], "جواز السفر"),
    (["نقل كفاله", "نقل الكفاله", "نقل خدمات"], "نقل خدمات"),
    (["راتب", "رواتب", "الراتب"], "حمايه الاجور"),
    (["رخصه بلدي", "بلدي", "رخصه المحل", "رخصه محل"], "رخصه نشاط تجاري بلديه"),
    (["رخصه بناء", "تصريح بناء", "ابني"], "رخصه بناء"),
    (["الحج", "حج", "العمره", "عمره", "معتمر", "حاج"], "الحج العمره"),
    (["الروضه"], "الروضه الشريفه"),
    (["وكاله", "توكيل"], "وكاله"),
    (["طلاق", "زواج", "حضانه", "نفقه", "ورث", "ميراث", "المواريث"], "احوال شخصيه"),
    (["نقل طالب", "انقل ولدي", "انقل بنتي", "مدرسه ثانيه"], "نقل طالب مدرسه"),
    (["معادله", "اعادل", "معادله شهاده"], "معادله الشهادات"),
    (["دواء", "ادويه", "مستحضرات تجميل", "جهاز طبي"], "تسجيل الغذاء والدواء"),
    (["كم رسوم", "كم سعر", "بكم", "التكلفه", "كم تكلف"], "رسوم الخدمه"),
    (["كم ياخذ", "كم مده", "متي يطلع", "مده"], "مده تنفيذ الخدمه"),
    (["وش احتاج", "ايش احتاج", "الاوراق", "اوراق"], "المستندات المطلوبه"),
    (["مين يقدر", "من يحق", "الشروط"], "الشروط"),
]


def _compile(entries):
    out = []
    for pats, add in entries:
        alts = "|".join(re.escape(p) for p in sorted(pats, key=len, reverse=True))
        out.append((re.compile(rf"(?<![\w؀-ۿ])(?:{alts})(?![\w؀-ۿ])"), add))
    return out


_EN_C = _compile(_EN)
_AR_C = _compile([(list(map(normalize_arabic, p)), normalize_arabic(a)) for p, a in _AR])


def expansions(query: str) -> list[str]:
    """Official terms to append for this query (may be empty)."""
    q = normalize_arabic(query).lower()
    adds = []
    for rx, add in _EN_C + _AR_C:
        if rx.search(q) and add not in adds:
            adds.append(add)
    return adds


def expand_query(query: str) -> str:
    adds = expansions(query)
    return f"{query} {' '.join(adds)}" if adds else query
