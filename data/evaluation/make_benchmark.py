"""Source of truth for the Dalil evaluation benchmark.

Run ``python data/evaluation/make_benchmark.py`` to regenerate
``data/evaluation/benchmark.csv``.

How the benchmark was written
-----------------------------
* Written by the project author with AI assistance (Claude), from the list of
  Ministry of Commerce service *titles*, BEFORE running retrieval on it, so the
  questions were not tuned to what the system happens to retrieve.
* Questions are phrased the way a person would ask, deliberately avoiding
  copying the official titles; "hard" questions share few or no keywords with
  the title.
* Arabic questions are written in natural (partly Saudi colloquial) Arabic.
* ``acceptable`` lists sibling services that would also be a correct answer
  (e.g. the establishment vs. company version of the same procedure).
* Unsupported questions are of two kinds:
    - ``out_of_domain``: not about government services at all.
    - ``not_indexed``: real Saudi government services that are NOT in Dalil's
      verified corpus (e.g. passports). Dalil must refuse these too, because
      it has no verified source for them. Several are deliberately close to
      the indexed business domain (VAT, GOSI, Balady) -- "hard negatives".
* ``split``: services are assigned to a calibration half (used to choose the
  method, the hybrid weight and the refusal threshold) and a held-out test half
  (used only for reporting). Both language versions of a service go to the
  same split to avoid leakage.
"""
from __future__ import annotations

import csv
from pathlib import Path

# (service sID, [acceptable alternative sIDs], difficulty, EN question, AR question)
SUPPORTED = [
    (1, [], "paraphrase", "I want to reserve a name for my new business", "أبغى أحجز اسم تجاري لمشروعي الجديد"),
    (34, [], "paraphrase", "My trade name reservation is about to expire, can I extend it?", "كيف أمدد حجز الاسم التجاري قبل ما ينتهي؟"),
    (22, [], "paraphrase", "How do I cancel a business name I reserved?", "أبي ألغي حجز اسم تجاري حجزته"),
    (38, [], "paraphrase", "How do I register a commercial registration for a sole proprietorship?", "كيف أطلع سجل تجاري لمؤسسة فردية؟"),
    (15, [2, 11], "paraphrase", "Do I need to confirm my sole proprietorship's commercial registry data every year?", "التأكيد السنوي لبيانات السجل التجاري للمؤسسة كم رسومه؟"),
    (68, [], "hard", "I closed my shop for good. How do I cancel my establishment's commercial registration?", "كيف أشطب السجل التجاري لمؤسستي الفردية؟"),
    (36, [], "hard", "I'm selling my one-person business to someone else. How do I move the registry into their name?", "أبغى أنقل ملكية سجل مؤسستي لشخص ثاني"),
    (33, [91], "paraphrase", "Where can I get an official extract of a commercial registration?", "أحتاج مستخرج سجل تجاري أو إفادة تجارية"),
    (91, [33, 25], "paraphrase", "Look up a company's details using its commercial registration number", "أبغى أستعلم عن بيانات سجل تجاري لشركة"),
    (99, [], "paraphrase", "Steps to set up a limited liability company in Saudi Arabia", "خطوات تأسيس شركة ذات مسؤولية محدودة"),
    (98, [], "paraphrase", "How do I establish a simple company?", "تأسيس شركة توصية بسيطة"),
    (92, [94], "paraphrase", "What are the requirements to incorporate a joint stock company?", "كيف أأسس شركة مساهمة؟"),
    (94, [92], "paraphrase", "What do I need to start a simplified joint stock company?", "تأسيس شركة مساهمة مبسطة"),
    (97, [], "hard", "Two partners want to form a general partnership where both are fully liable. How?", "تأسيس شركة تضامن"),
    (20, [23], "paraphrase", "I'm a foreign investor with an investment license. How do I establish my company?", "تأسيس شركة بموجب ترخيص استثماري"),
    (52, [], "paraphrase", "Can I convert my sole proprietorship into a company?", "تحويل المؤسسة الفردية إلى شركة"),
    (16, [], "hard", "How can a company manager allow an employee to handle the company's transactions online on his behalf?", "كيف أفوض شخص ينجز معاملات منشأتي إلكترونيًا؟"),
    (25, [91], "paraphrase", "How can I check that a certificate of origin issued by the ministry is genuine?", "التحقق من صحة مستند صادر من وزارة التجارة"),
    (41, [], "hard", "I became the exclusive distributor of a foreign brand. How do I register the agency?", "تسجيل وكالة تجارية جديدة"),
    (43, [], "paraphrase", "Renew my commercial agency registration", "تجديد قيد وكالة تجارية"),
    (18, [], "hard", "Who is the official registered agent for a brand in Saudi Arabia?", "أبغى أعرف مين الوكيل المعتمد لعلامة تجارية"),
    (44, [], "paraphrase", "As the agent, how do I write off a commercial agency?", "شطب وكالة تجارية من قبل الوكيل"),
    (42, [], "paraphrase", "Change the details of a registered commercial agency", "تعديل بيانات وكالة تجارية مسجلة"),
    (46, [], "hard", "Do I need permission to run a discount sale in my store?", "أبغى أسوي تخفيضات في محلي، هل أحتاج ترخيص؟"),
    (24, [], "paraphrase", "How do I register a franchise?", "تسجيل امتياز تجاري (فرنشايز)"),
    (14, [], "paraphrase", "Report a franchisor who broke the franchise law", "بلاغ عن مخالفة نظام الامتياز التجاري"),
    (10, [], "hard", "Report a shop that is secretly run by a foreigner under a Saudi citizen's name", "الإبلاغ عن حالة تستر تجاري"),
    (55, [], "hard", "A store refused to refund me. Where do I file a complaint?", "أبي أقدم شكوى على متجر ما رجع لي فلوسي"),
    (30, [], "paraphrase", "Check whether a product I bought has been recalled as defective", "الاستعلام عن المنتجات المعيبة"),
    (5, [], "paraphrase", "How do I check if my business has immediate violations?", "الاستعلام عن المخالفات الفورية على منشأتي"),
    (3, [], "paraphrase", "I want to object to a violation decision issued against my company", "أبغى أعترض على قرار مخالفة صدر على شركتي"),
    (19, [], "paraphrase", "Track the status of a request I submitted to the Ministry of Commerce", "متابعة حالة طلب قدمته لوزارة التجارة"),
    (65, [], "paraphrase", "How do I verify that a consultant is licensed to practise?", "البحث عن المرخص لهم بمزاولة المهن الاستشارية"),
    (47, [49, 48], "paraphrase", "Issue a new license to practise a consulting profession", "إصدار ترخيص مهني جديد"),
    (49, [76], "paraphrase", "My professional license expired. How do I renew it?", "تجديد الترخيص المهني"),
    (60, [], "hard", "I want to open a gold and jewellery shop. What license do I need?", "ترخيص محل ذهب ومجوهرات"),
    (63, [], "paraphrase", "Permit to import non-hazardous chemicals", "تصريح استيراد مواد كيميائية غير خطرة"),
    (35, [], "paraphrase", "Where do I declare the real beneficial owners of my company?", "الإفصاح عن المستفيد الحقيقي للشركة"),
    (32, [], "paraphrase", "How can I vote online in chamber of commerce elections?", "التصويت الإلكتروني في انتخابات الغرف التجارية"),
    (66, [150], "paraphrase", "Update the list of shareholders in the company's records", "تحديث سجل المساهمين في الشركة"),
    (82, [37], "paraphrase", "Amend the company's articles of association", "تعديل عقد تأسيس الشركة"),
    (7, [27, 28], "paraphrase", "Submit the results of the company's general assembly meeting", "رفع نتائج اجتماع الجمعية العامة"),
    (62, [9, 67], "paraphrase", "My establishment's commercial registration was suspended. How do I lift the suspension?", "رفع تعليق السجل التجاري لمؤسسة فردية"),
    (13, [61], "paraphrase", "Update the owner's personal information on the commercial registration", "تحديث بيانات مالك السجل التجاري"),
]

MIXED = [
    (99, [], "كيف أسجل LLC في السعودية؟"),
    (15, [2, 11], "annual confirmation للسجل التجاري حق المؤسسة"),
    (43, [], "تجديد commercial agency"),
    (24, [], "franchise registration وش المطلوب؟"),
    (38, [], "how to قيد سجل تجاري لمؤسسة فردية"),
]

# (query, language, kind)
UNSUPPORTED = [
    ("What's the best pizza in Riyadh?", "en", "out_of_domain"),
    ("Who will win the football match tonight?", "en", "out_of_domain"),
    ("Write my university application essay for me.", "en", "out_of_domain"),
    ("What is 15 times 23?", "en", "out_of_domain"),
    ("Tell me a joke", "en", "out_of_domain"),
    ("Best hotels to stay in AlUla", "en", "out_of_domain"),
    ("Translate this sentence into French", "en", "out_of_domain"),
    ("كيف الطقس في جدة بكرة؟", "ar", "out_of_domain"),
    ("طريقة عمل الكبسة", "ar", "out_of_domain"),
    ("أفضل جوال أشتريه هالسنة", "ar", "out_of_domain"),
    ("من هو أفضل لاعب في الدوري السعودي؟", "ar", "out_of_domain"),
    ("كيف أتعلم البرمجة من الصفر؟", "ar", "out_of_domain"),
    ("How do I renew my passport?", "en", "not_indexed"),
    ("How can I renew my driving licence?", "en", "not_indexed"),
    ("How do I apply for a Hajj permit?", "en", "not_indexed"),
    ("How can I get my foreign university degree recognized in Saudi Arabia?", "en", "not_indexed"),
    ("Book an appointment at a Ministry of Health primary care centre", "en", "not_indexed"),
    ("How do I apply for a tourist e-visa to Saudi Arabia?", "en", "not_indexed"),
    ("How do I register my national address?", "en", "not_indexed"),
    ("University admission requirements for King Saud University", "en", "not_indexed"),
    ("How do I register my company for VAT?", "en", "not_indexed"),
    ("How do I issue work visas for my company's foreign employees?", "en", "not_indexed"),
    ("How do I get a municipal (Balady) license for my shop?", "en", "not_indexed"),
    ("كيف أجدد جواز السفر؟", "ar", "not_indexed"),
    ("تجديد الإقامة للعمالة", "ar", "not_indexed"),
    ("كيف أسدد المخالفات المرورية؟", "ar", "not_indexed"),
    ("كيف أعادل شهادتي الجامعية من خارج السعودية؟", "ar", "not_indexed"),
    ("كيف أسجل في الضمان الاجتماعي؟", "ar", "not_indexed"),
    ("كيف أصدر هوية وطنية لأول مرة؟", "ar", "not_indexed"),
    ("تسجيل المنشأة في التأمينات الاجتماعية", "ar", "not_indexed"),
    ("إصدار رخصة بلدي للمحل", "ar", "not_indexed"),
    ("حجز موعد في مستشفى حكومي", "ar", "not_indexed"),
    # The Ministry of Commerce pages for "Issuing Laboratory License" (sID=56) and
    # "Seasonal permits for catering vehicles" (sID=53) have no description, so
    # validation quarantined them; their questions became not-indexed cases.
    ("Temporary permit for food trucks serving pilgrims during the season", "en", "not_indexed"),
    ("تصاريح موسمية لمركبات الإعاشة", "ar", "not_indexed"),
    ("How do I get a license for a laboratory?", "en", "not_indexed"),
    ("إصدار رخصة مختبر", "ar", "not_indexed"),
]


def build_rows() -> list[dict]:
    rows = []
    for i, (sid, alts, diff, en, ar) in enumerate(SUPPORTED):
        split = "calibration" if i % 2 == 0 else "test"
        for lang, q in (("en", en), ("ar", ar)):
            rows.append(dict(query=q, language=lang, expected_service_id=f"mc-{sid}",
                             acceptable_service_ids=";".join(f"mc-{a}" for a in alts),
                             query_type=diff, supported=1, split=split))
    for i, (sid, alts, q) in enumerate(MIXED):
        rows.append(dict(query=q, language="mixed", expected_service_id=f"mc-{sid}",
                         acceptable_service_ids=";".join(f"mc-{a}" for a in alts),
                         query_type="mixed_language", supported=1,
                         split="calibration" if i % 2 == 0 else "test"))
    counters: dict[tuple, int] = {}
    for q, lang, kind in UNSUPPORTED:
        k = (lang, kind)
        counters[k] = counters.get(k, 0) + 1
        rows.append(dict(query=q, language=lang, expected_service_id="", acceptable_service_ids="",
                         query_type=kind, supported=0,
                         split="calibration" if counters[k] % 2 == 1 else "test"))
    for n, r in enumerate(rows, 1):
        r["query_id"] = f"q{n:03d}"
    return rows


if __name__ == "__main__":
    out = Path(__file__).with_name("benchmark.csv")
    cols = ["query_id", "query", "language", "expected_service_id", "acceptable_service_ids",
            "query_type", "supported", "split"]
    rows = build_rows()
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} queries -> {out}")
