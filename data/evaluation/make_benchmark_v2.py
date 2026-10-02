"""Dalil V2 benchmark: realistic user language across agencies.

    python data/evaluation/make_benchmark_v2.py   ->  data/evaluation/benchmark_v2.csv

How it was written (honest provenance)
--------------------------------------
* Written by the project author with AI assistance (Claude) on 1 Oct 2026,
  from the lists of official service *titles* of each captured agency,
  BEFORE any V2 retrieval setting was tuned. The V1 benchmark is kept
  unchanged (``benchmark.csv``) as the baseline.
* Each **family** is one information need, asked in several ways:
  formal English, conversational English, short/simple, misspelled,
  formal Arabic, colloquial Saudi Arabic, and mixed Arabic/English where
  realistic. The user's own trade-name examples are in family ``mc-trade-name``.
* Expected answers are given as (source, official English title) and
  resolved to service ids by exact title match when the benchmark is loaded
  (``resolve_expected`` in ``src/evaluation/evaluate_v2.py``), so a parsing
  change can't silently shift labels. ``acceptable`` lists sibling services
  that are also correct.
* Splits are by family (all phrasings of one need share a split):
  ``dev`` (tuning), ``val`` (choosing between tuned options) and ``test``
  (held out, reported only). The family containing the user's reported
  failure is in ``dev`` because it was used for diagnosis.
* Unsupported questions: ``out_of_domain`` (not public-service questions)
  and ``not_indexed`` (real Saudi services whose agencies Dalil could not
  capture: MOI/Absher, traffic, MOH appointments, GOSI…). Several are
  deliberately close to indexed topics.
"""
from __future__ import annotations

import csv
from pathlib import Path

# family id, split, expected (source, title), acceptable [(source, title)], [(lang, style, query)]
F = []


def fam(fid, split, expected, acceptable, queries):
    F.append((fid, split, expected, acceptable, queries))


MC = "mc"
# ------------------------------------------------------------------ Commerce
fam("mc-trade-name", "dev", (MC, "Trade Name Reservation"), [], [
    ("en", "formal", "What is the procedure for reserving a trade name?"),
    ("en", "user_reported", "How can I reserve a trade name?"),
    ("en", "user_reported", "I want to book a business name"),
    ("en", "user_reported", "How do I get a commercial name?"),
    ("en", "user_reported", "Where can I register/reserve my company name?"),
    ("en", "simple", "I need a company name, what do I do?"),
    ("en", "misspelled", "how to reserv a trade nme"),
    ("ar", "formal", "ما إجراءات حجز الاسم التجاري؟"),
    ("ar", "colloquial", "ابي احجز اسم لمحلي الجديد"),
    ("mixed", "mixed", "حجز trade name كيف"),
])
fam("mc-llc", "test", (MC, "Establish a limited liability company"), [], [
    ("en", "formal", "What are the steps to incorporate a limited liability company?"),
    ("en", "conversational", "how do i open an LLC in saudi"),
    ("en", "short", "LLC setup"),
    ("ar", "formal", "ما خطوات تأسيس شركة ذات مسؤولية محدودة؟"),
    ("ar", "colloquial", "ابغى افتح شركة ذات مسؤولية محدودة"),
    ("mixed", "mixed", "تأسيس LLC"),
])
fam("mc-annual-cr", "val", (MC, "The annual confirmation of the commercial registry data of an establishment"),
    [(MC, "The annual confirmation of the company's main commercial registry data"),
     (MC, "The annual confirmation of the company's branch commercial registry data")], [
    ("en", "formal", "Is an annual confirmation of commercial registration data required?"),
    ("en", "conversational", "do i have to confirm my CR details every year"),
    ("en", "fees", "how much is the yearly CR confirmation fee"),
    ("ar", "formal", "التأكيد السنوي لبيانات السجل التجاري"),
    ("ar", "colloquial", "لازم أأكد بيانات السجل كل سنة؟"),
])
fam("mc-complaint", "test", (MC, "Receiving a Commercial Complaint"), [], [
    ("en", "formal", "How do I file a complaint against a store?"),
    ("en", "conversational", "a shop won't give me my money back"),
    ("en", "short", "consumer complaint"),
    ("ar", "colloquial", "كيف اشتكي على محل؟"),
    ("ar", "formal", "تقديم بلاغ عن متجر"),
])
fam("mc-cr-extract", "dev", (MC, "Extractor of a commercial registry / commercial letter"),
    [(MC, "Query about commercial registration data")], [
    ("en", "formal", "How do I obtain an extract of a commercial registration?"),
    ("en", "conversational", "i need proof that my business is registered"),
    ("ar", "formal", "مستخرج سجل تجاري"),
])
# ----------------------------------------------------------------------- HRSD
H = "hrsd"
fam("hrsd-maid-visa", "dev", (H, "Request to issue a domestic worker visa electronically"),
    [(H, "Hiring a Domestic Worker from Outside the Kingdom (Through Recruitment Agencies)"), (H, "Visa Application")], [
    ("en", "formal", "How do I apply for a domestic worker visa?"),
    ("en", "conversational", "I want to bring a housemaid from abroad"),
    ("en", "short", "maid visa"),
    ("ar", "colloquial", "كيف اطلع تأشيرة عاملة منزلية؟"),
    ("ar", "colloquial", "ابي استقدم خادمة"),
])
fam("hrsd-transfer", "test", (H, "Transferring Expatriate Worker Service"),
    [(H, "Transfer of services from another employer (Freedom of movement)"),
     (H, "Request for transfer of expatriate workers"), (H, "Employee Transfer"),
     (H, "Transfer of expatriate services by border number")], [
    ("en", "formal", "How can an employer transfer a foreign worker from another company?"),
    ("en", "conversational", "how do i move a worker's sponsorship to my company"),
    ("ar", "colloquial", "نقل كفالة عامل وافد"),
    ("ar", "formal", "كيف أنقل خدمات عامل وافد؟"),
])
fam("hrsd-citizen-account", "val", (H, "Citizen account"), [], [
    ("en", "formal", "How do I register for the Citizen Account program?"),
    ("en", "short", "citizen account registration"),
    ("ar", "formal", "التسجيل في حساب المواطن"),
    ("ar", "colloquial", "ابي اسجل في حساب المواطن"),
])
fam("hrsd-dispute", "test", (H, "Friendly Settlement for Labor Disputes"),
    [(H, "Reporting Violations of Labor Regulations")], [
    ("en", "conversational", "My employer hasn't paid my salary, how can I resolve the dispute?"),
    ("en", "short", "labour dispute settlement"),
    ("ar", "colloquial", "عندي خلاف مع صاحب العمل كيف احله؟"),
    ("ar", "formal", "التسوية الودية للخلافات العمالية"),
])
fam("hrsd-disability-aid", "dev", (H, "Financial subsidy service for people with disability"),
    [(H, "Issue Disability Certificate"), (H, "Disability Assessment")], [
    ("en", "formal", "Is there financial support for people with disabilities?"),
    ("en", "short", "disability allowance"),
    ("ar", "formal", "الإعانة المالية للأشخاص ذوي الإعاقة"),
])
fam("hrsd-freelance", "test", (H, "freelance work document renewal service"), [], [
    ("en", "conversational", "How do I get a freelance work permit as a Saudi?"),
    ("en", "short", "freelancer document"),
    ("ar", "formal", "وثيقة العمل الحر"),
])
fam("hrsd-wps", "val", (H, "Issue Wage Protection Certificate"), [(H, "Uploading the Wage Protection file")], [
    ("en", "formal", "How can an establishment get a wage protection certificate?"),
    ("ar", "formal", "شهادة حماية الأجور"),
])
fam("hrsd-social-security", "test", (H, "Developed Social Security System"),
    [(H, "Social security ineligibility objection"),
     (H, "Complaints, Inquiries, and Suggestions of Social Security Beneficiaries")], [
    ("en", "formal", "How do I apply for social security support?"),
    ("ar", "colloquial", "كيف اسجل في الضمان الاجتماعي المطور؟"),
])
fam("hrsd-violence", "dev", (H, "Reporting domestic violence"), [], [
    ("en", "formal", "How can I report domestic violence?"),
    ("en", "conversational", "someone in my family is being abused, where do I report it"),
    ("ar", "formal", "الإبلاغ عن العنف الأسري"),
])
fam("hrsd-telework", "val", (H, "Create and document a telework contract"), [], [
    ("en", "conversational", "how do I document a remote work contract"),
    ("ar", "formal", "توثيق عقد عمل عن بعد"),
])
fam("hrsd-ajeer", "test", (H, "Ajeer Permit Issuance"), [], [
    ("en", "formal", "How do I issue a temporary work permit through Ajeer?"),
    ("ar", "formal", "إصدار تصاريح أجير"),
])
# ---------------------------------------------------------------------- ZATCA
Z = "zatca"
fam("zatca-vat-reg", "dev", (Z, "VAT Registration for Businesses"),
    [(Z, "VAT Registration for Individuals"), (Z, "VAT Registration for Group")], [
    ("en", "formal", "How do I register my company for VAT?"),
    ("en", "conversational", "I need a VAT number for my business"),
    ("en", "misspelled", "vat registeration"),
    ("ar", "formal", "التسجيل في ضريبة القيمة المضافة"),
    ("ar", "colloquial", "ابي رقم ضريبي لمؤسستي"),
])
fam("zatca-vat-return", "test", (Z, "Submit VAT Return"), [(Z, "VAT Return Amendment")], [
    ("en", "formal", "How do I file my VAT return?"),
    ("en", "short", "submit vat declaration"),
    ("ar", "formal", "تقديم إقرار ضريبة القيمة المضافة"),
])
fam("zatca-zakat-ind", "val", (Z, "Zakat Payment for Individuals"), [(Z, "ZAKATY Calculator (Individuals)")], [
    ("en", "conversational", "How can I pay my zakat online as an individual?"),
    ("en", "short", "calculate my zakat"),
    ("ar", "colloquial", "كيف ادفع زكاتي أونلاين؟"),
])
fam("zatca-car-import", "test", (Z, "Vehicle Import for Individuals"), [(Z, "Estimated Vehicle Import Duty Calculator")], [
    ("en", "formal", "How do I import a car into Saudi Arabia?"),
    ("en", "fees", "how much customs duty do I pay on an imported car"),
    ("ar", "colloquial", "ابي استورد سيارة من برا"),
])
fam("zatca-traveler", "dev", (Z, "Submit Customs Declaration for Travelers"), [], [
    ("en", "conversational", "Do I need to declare cash or goods when I travel?"),
    ("ar", "formal", "الإقرار الجمركي للمسافرين"),
])
fam("zatca-vat-refund", "test", (Z, "VAT Refund"), [(Z, "VAT Refund Registration for Eligible Persons")], [
    ("en", "formal", "How can I get a VAT refund?"),
    ("ar", "formal", "استرداد ضريبة القيمة المضافة"),
])
fam("zatca-tax-residency", "val", (Z, "Tax Residency Certificate for Individuals"),
    [(Z, "Tax Residency & Withholding Tax Certificate")], [
    ("en", "short", "tax residency certificate"),
    ("ar", "formal", "شهادة الإقامة الضريبية"),
])
fam("zatca-importer", "test", (Z, "Register as an Importer or Exporter"), [], [
    ("en", "conversational", "how do I register as an importer"),
    ("ar", "formal", "التسجيل كمستورد أو مصدر"),
])
# ------------------------------------------------------------------------ MOE
E = "moe"
fam("moe-school-fees", "dev", (E, "Reviewing the School Tuition Fees"), [], [
    ("en", "formal", "How can I check private school tuition fees?"),
    ("ar", "colloquial", "كم رسوم المدارس الأهلية؟ وين اشوفها"),
])
fam("moe-admission", "test", (E, "Unified University Admission and Educational Institutions"), [], [
    ("en", "formal", "How do I apply for university admission?"),
    ("ar", "formal", "القبول الموحد في الجامعات"),
])
fam("moe-degree", "val", (E, "Verification of University Qualifications (Qualified)"),
    [(E, "Inquiry about the Recommended Universities")], [
    ("en", "formal", "How can my university degree be verified?"),
    ("ar", "formal", "التحقق من المؤهلات الجامعية"),
])
fam("moe-bus", "test", (E, "School Transportation Service"), [], [
    ("en", "conversational", "is there a school bus for my kids"),
    ("ar", "formal", "خدمة النقل المدرسي"),
])
fam("moe-first-grade", "dev", (E, "Pre-enrollment for public education students in the first grade and the third level of kindergartens in public schools"), [], [
    ("en", "conversational", "How do I register my child for first grade?"),
    ("ar", "colloquial", "ابي اسجل ولدي أول ابتدائي"),
])
fam("moe-scholarship", "test", (E, "Apply for a scholarship in King's External Scholarship Program"),
    [(E, "Request for Studying Abroad on One's Own Account")], [
    ("en", "formal", "How do I apply for a government scholarship to study abroad?"),
    ("ar", "formal", "الابتعاث الخارجي"),
])
# ------------------------------------------------------------------------ Hajj
J = "haj"
fam("haj-umrah-permit", "val", (J, "Umrah permit application"), [], [
    ("en", "formal", "How do I get a permit to perform Umrah?"),
    ("ar", "colloquial", "كيف اطلع تصريح عمرة؟"),
])
fam("haj-rawdah", "test", (J, "Issuance of Rawdah Sharif Permit (Men / Women)"), [], [
    ("en", "conversational", "how do I book a visit to the Rawdah in the Prophet's Mosque"),
    ("ar", "formal", "تصريح زيارة الروضة الشريفة"),
])
fam("haj-umrah-companies", "dev", (J, "Inquire About Licensed Umrah Companies and Establishments"), [], [
    ("en", "conversational", "is this Umrah company licensed?"),
    ("ar", "formal", "الاستعلام عن شركات العمرة المرخصة"),
])
fam("haj-permit-status", "test", (J, "Inquire about the Reservation Status of the Hajj Permit"), [], [
    ("en", "short", "check my hajj permit booking status"),
    ("ar", "formal", "حالة حجز تصريح الحج"),
])
# ------------------------------------------------------------------------ MOJ
M = "moj"
fam("moj-poa", "dev", (M, "Issue Power of Attorney"),
    [(M, "Register power of attorney"), (M, "Real Estate Powers of Attorney"), (M, "Multi-principal power of attorney")], [
    ("en", "formal", "How do I issue a power of attorney online?"),
    ("en", "short", "make a POA"),
    ("ar", "colloquial", "كيف اطلع وكالة إلكترونية؟"),
])
fam("moj-verify-poa", "test", (M, "Verify power of attorney"), [], [
    ("en", "formal", "How can I check whether a power of attorney is valid?"),
    ("ar", "formal", "التحقق من صحة الوكالة"),
])
fam("moj-marriage", "val", (M, "Create marriage contract"),
    [(M, "Certify marriage contract"), (M, "Notarize marriage with non-Saudi spouse"), (M, "Notarize previous marriage")], [
    ("en", "conversational", "how do we get a marriage contract"),
    ("ar", "formal", "توثيق عقد النكاح"),
])
fam("moj-divorce", "test", (M, "Notarize divorce"),
    [(M, "Notarize Khul' (Wife-Initiated Divorce)"), (M, "Notarize Revocation of Divorce")], [
    ("en", "formal", "How is a divorce registered?"),
    ("ar", "formal", "توثيق الطلاق"),
])
fam("moj-custody", "dev", (M, "Apply For Child Custody Order"),
    [(M, "Notarize child custody"), (M, "Apply for Child Visitation Decision")], [
    ("en", "conversational", "who gets custody of the children after divorce, how do I apply"),
    ("ar", "formal", "طلب حضانة الأطفال"),
])
fam("moj-inheritance", "test", (M, "Calculate inheritance"), [(M, "Legal Heirship Certificate")], [
    ("en", "formal", "How do I calculate inheritance shares?"),
    ("ar", "formal", "حساب المواريث"),
])
fam("moj-deed", "val", (M, "Real Estate Title Deed Inquiry"), [(M, "Inquiry on owned properties")], [
    ("en", "short", "check a property title deed"),
    ("ar", "formal", "الاستعلام عن صك عقاري"),
])
fam("moj-enforcement", "test", (M, "File an enforcement application"), [], [
    ("en", "conversational", "someone owes me money and won't pay, how do I file for enforcement?"),
    ("ar", "formal", "تقديم طلب تنفيذ"),
])


# =============================================================================
# FRESH FAMILIES (final pass, 1 Oct 2026)
# Written AFTER the lexicon was frozen (src/retrieval/lexicon.py sha256
# 97c3fe1b1d46...c2ee, unchanged since) and before any V2 setting was tuned on
# the final corpus. They cover agencies that had no questions before (MoMAH,
# MOFA, SFDA, CHI) plus extra MoJ/ZATCA/MOE/Hajj needs, cross-agency and
# ambiguous questions. Flagged fresh=1 so results can be reported separately.
# CHI has no Arabic page text in the index, so its Arabic questions test
# cross-language retrieval (Arabic question -> English official text).
# =============================================================================
FRESH = set()


def fam2(fid, split, expected, acceptable, queries):
    FRESH.add(fid)
    fam(fid, split, expected, acceptable, queries)


O, S, C, B = "mofa", "sfda", "chi", "momah"
fam2("mofa-passport-renew", "test", (O, "Passport Renewal"), [(O, "Passport Issuance")], [
    ("en", "conversational", "My Saudi passport expired while I'm living abroad, how do I renew it?"),
    ("en", "short", "renew passport at embassy"),
    ("ar", "colloquial", "جوازي خلص وانا برا السعودية كيف اجدده"),
    ("mixed", "mixed", "تجديد passport من السفارة"),
])
fam2("mofa-family-visit", "test", (O, "Family Visit Visa"), [(O, "Personal Visit Visa")], [
    ("en", "conversational", "How can I bring my parents to visit me in Saudi Arabia?"),
    ("en", "formal", "What is the procedure for a family visit visa?"),
    ("ar", "formal", "تأشيرة الزيارة العائلية"),
    ("ar", "colloquial", "ابي اجيب اهلي زيارة للسعودية"),
])
fam2("mofa-attestation", "val", (O, "Document Ratification"),
     [(O, "Power of Attorney Ratification"), (O, "Establishment Contract Ratification")], [
    ("en", "formal", "How do I get a document attested by the Ministry of Foreign Affairs?"),
    ("en", "short", "document legalization"),
    ("ar", "formal", "تصديق المستندات من وزارة الخارجية"),
])
fam2("mofa-tourist", "test", (O, "Tourist Visa"), [(O, "Electronic Visa Waiver")], [
    ("en", "conversational", "can a foreigner get a visa just for tourism in saudi"),
    ("ar", "formal", "التأشيرة السياحية"),
])
fam2("mofa-medical", "test", (O, "Medical Visa"), [], [
    ("en", "conversational", "my relative needs to come to Saudi for medical treatment, what visa?"),
    ("ar", "colloquial", "تأشيرة علاج لقريبي"),
])
fam2("mofa-transit", "val", (O, "Land Transit Visa"), [(O, "Transit of Goods Visa")], [
    ("en", "conversational", "I want to drive through Saudi Arabia to reach another country"),
    ("ar", "formal", "تأشيرة العبور البري"),
])
fam2("sfda-drug-reg", "test", (S, "Saudi Drug Registration (SDR)"),
     [(S, "Saudi Drugs information system (SDI)"), (S, "Drugs List")], [
    ("en", "formal", "How is a new pharmaceutical product registered in the Kingdom?"),
    ("en", "short", "medicine registration"),
    ("ar", "formal", "تسجيل الأدوية في هيئة الغذاء والدواء"),
])
fam2("sfda-side-effect", "test", (S, "Saudi Vigilance System"),
     [(S, "Medication Error Report"), (S, "National Center for Medical Devices Reporting")], [
    ("en", "conversational", "I had a bad reaction to a medicine, where do I report it?"),
    ("en", "short", "report adverse drug reaction"),
    ("ar", "colloquial", "صار عندي أعراض جانبية من دواء وين أبلغ؟"),
])
fam2("sfda-food-establishment", "val", (S, "Registration of Local Food Establishments"), [(S, "Customer Journey")], [
    ("en", "conversational", "how do I register my food factory with the food and drug authority"),
    ("ar", "formal", "تسجيل المنشآت الغذائية المحلية"),
])
fam2("sfda-poison", "test", (S, "The National Drug & Poison Information Center (NDPIC)"), [], [
    ("en", "conversational", "who can I ask about a possible poisoning or a drug interaction?"),
    ("ar", "formal", "المركز الوطني لمعلومات الأدوية والسموم"),
])
fam2("sfda-clinical-trial", "test", (S, "Saudi Clinical Trials Registry (SCTR)"), [], [
    ("en", "short", "register a clinical trial"),
    ("ar", "formal", "تسجيل التجارب السريرية"),
])
fam2("chi-provider-complaint", "test", (C, "File complaint against service provider"),
     [(C, "File complaint against CHI"), (C, "Complaint Follow-Up")], [
    ("en", "conversational", "The hospital refused to accept my health insurance, where can I complain?"),
    ("en", "short", "health insurance complaint against clinic"),
    ("ar", "colloquial", "المستشفى رفض تأميني الطبي ابي اشتكي"),
])
fam2("chi-employer-insurance", "test", (C, "File complaint against an employer"), [], [
    ("en", "conversational", "my employer did not give me medical insurance, what can I do?"),
    ("ar", "colloquial", "الشركة ما سوت لي تأمين طبي"),
])
fam2("chi-accreditation", "val", (C, "Issuance of New Accreditation"), [(C, "Accreditation Renewal")], [
    ("en", "formal", "How does a healthcare provider obtain accreditation from the Council of Health Insurance?"),
    ("ar", "formal", "اعتماد مقدم خدمة صحية لدى مجلس الضمان الصحي"),
])
fam2("momah-building-permit", "test", (B, "Issuing a Building Permit"),
     [(B, "Renewal of a Building Permit"), (B, "Building Permit Inquiry")], [
    ("en", "conversational", "I want to build a house on my land, what permit do I need?"),
    ("en", "short", "construction permit"),
    ("ar", "formal", "إصدار رخصة بناء"),
    ("ar", "colloquial", "ابي اطلع رخصة بناء لبيتي"),
])
fam2("momah-shop-license", "test", (B, "Issuing a Commercial License"),
     [(B, "Issuing a Fast-Track Commercial License"), (B, "Renewing a Commercial License")], [
    ("en", "conversational", "I'm opening a shop, which municipal license do I need?"),
    ("en", "formal", "How is a commercial activity license issued by the municipality?"),
    ("ar", "colloquial", "ابي افتح محل وش رخصة البلدية المطلوبة"),
    ("mixed", "mixed", "رخصة Balady لمحل تجاري"),
])
fam2("momah-health-cert", "val", (B, "Issuance of Health Certificate"),
     [(B, "Renewal of Health Certificate"), (B, "Issuance of a Seasonal Health Certificate")], [
    ("en", "conversational", "my restaurant workers need a health certificate, how do they get it"),
    ("ar", "colloquial", "كيف اطلع شهادة صحية للعامل"),
])
fam2("momah-food-truck", "test", (B, "Mobile Cart License Issuance"),
     [(B, "Non-Food Mobile Cart Permit"), (B, "Reserving a Site for a Mobile Cart License")], [
    ("en", "conversational", "I want to start a food truck, what license do I need?"),
    ("ar", "formal", "إصدار رخصة عربة متنقلة"),
    ("mixed", "mixed", "رخصة food truck"),
])
fam2("momah-demolition", "test", (B, "Issuing a Building Demolition Permit"), [], [
    ("en", "short", "permit to demolish an old building"),
    ("ar", "formal", "تصريح هدم مبنى"),
])
fam2("momah-grave", "test", (B, "Inquiring About a Deceased's Grave"), [(B, "Cemetery Inquiry")], [
    ("en", "conversational", "how can I find where my late grandfather is buried?"),
    ("ar", "formal", "الاستعلام عن قبر متوفى"),
])
fam2("momah-sign", "val", (B, "Issuing an Operational License for an Advertising or Promotional Sign"), [], [
    ("en", "conversational", "do I need a license to put an advertising sign on my shop?"),
    ("ar", "formal", "رخصة لوحة إعلانية"),
])
fam2("momah-tobacco", "test", (B, "Permit for Providing Tobacco Products"), [], [
    ("en", "conversational", "can my grocery sell cigarettes, what permit is required?"),
    ("ar", "formal", "تصريح تقديم منتجات التبغ"),
])
fam2("momah-excavation", "test", (B, "Excavation Permits"),
     [(B, "Emergency Excavation Permit"), (B, "Building Connection Excavation Permit Request")], [
    ("en", "short", "digging permit for utility works"),
    ("ar", "formal", "تصاريح الحفريات"),
])
fam2("zatca-due-invoices", "test", (Z, "Due Invoices Payment"), [], [
    ("en", "conversational", "how do I pay my outstanding ZATCA bills"),
    ("ar", "formal", "سداد الفواتير المستحقة للهيئة"),
])
fam2("zatca-withholding", "val", (Z, "Submit Withholding Tax Return"), [], [
    ("en", "formal", "How do I declare withholding tax on payments to a non-resident?"),
    ("ar", "formal", "تقديم إقرار ضريبة الاستقطاع"),
])
fam2("moe-maternity", "test", (E, "Requesting Maternity Leave"), [(E, "Maternity and Childcare Leave")], [
    ("en", "conversational", "I'm a teacher and I'm pregnant, how do I request maternity leave?"),
    ("ar", "formal", "طلب إجازة أمومة للمعلمة"),
])
fam2("moe-early-retirement", "test", (E, "Applying for Early Retirement"), [], [
    ("en", "short", "teacher early retirement request"),
    ("ar", "colloquial", "ابي اتقاعد مبكر وانا معلم"),
])
fam2("moe-student-certificate", "val", (E, "Request for Printing an Introduction Certificate for a Regular Student"),
     [(E, "View and Print Certificates")], [
    ("en", "conversational", "I need a letter proving my son is enrolled at school"),
    ("ar", "formal", "طباعة تعريف طالب منتظم"),
])
fam2("haj-domestic-companies", "test", (J, "Inquire About Licensed Domestic Hajj Companies and Establishments"), [], [
    ("en", "conversational", "how can I check that a domestic Hajj campaign is licensed?"),
    ("ar", "formal", "الاستعلام عن شركات حجاج الداخل المرخصة"),
])
fam2("moj-will", "test", (M, "Notarize will"), [], [
    ("en", "conversational", "how do I officially register my will?"),
    ("ar", "formal", "توثيق الوصية"),
])
fam2("moj-revoke-poa", "val", (M, "Revoke power of attorney"), [], [
    ("en", "conversational", "I want to cancel a power of attorney I gave someone"),
    ("ar", "colloquial", "ابي الغي وكالة سويتها"),
])
fam2("moj-lawsuit", "test", (M, "File statement of claim"), [(M, "e-Litigation (Digital pleading)")], [
    ("en", "conversational", "how do I file a lawsuit online?"),
    ("ar", "formal", "رفع دعوى إلكترونيا"),
])
fam2("moj-bankruptcy", "test", (M, "File Bankruptcy Petition"), [(M, "Bankruptcy filings database")], [
    ("en", "short", "file for bankruptcy"),
    ("ar", "formal", "تقديم طلب إفلاس"),
])
fam2("moj-mobile-notary", "test", (M, "Book mobile notary public appointment"),
     [(M, "Inquiry on mobile notary public appointments")], [
    ("en", "conversational", "can a notary come to my house?"),
    ("ar", "formal", "حجز موعد كاتب عدل متنقل"),
])
# cross-agency / ambiguous: more than one agency has a correct answer
fam2("x-poa-abroad", "test", (O, "Power of Attorney Ratification"),
     [(M, "Issue Power of Attorney"), (M, "Register power of attorney")], [
    ("en", "conversational", "I'm outside Saudi Arabia, how do I get a power of attorney certified?"),
    ("ar", "colloquial", "انا برا المملكة وابي اصدق وكالة"),
])
fam2("x-worker-visa", "test", (H, "Visa Application"),
     [(O, "Permanent Work Visa"), (O, "Temporary Work Visa"), (H, "Request to issue a domestic worker visa electronically")], [
    ("en", "conversational", "how do I get a work visa for a new employee from abroad?"),
    ("ar", "formal", "طلب تأشيرة عمل"),
])

UNSUPPORTED_FRESH = [
    ("en", "not_indexed", "How do I register my national address?"),
    ("en", "not_indexed", "How do I get a birth certificate for my newborn?"),
    ("en", "not_indexed", "How do I report a car accident to Najm?"),
    ("en", "not_indexed", "How do I apply for a fishing licence?"),
    ("en", "not_indexed", "How can I dispute my electricity bill?"),
    ("en", "not_indexed", "Reset my Absher password"),
    ("ar", "not_indexed", "كيف أطلع شهادة ميلاد لمولودي؟"),
    ("ar", "not_indexed", "تحديث العنوان الوطني"),
    ("ar", "not_indexed", "الاعتراض على فاتورة الكهرباء"),
    ("ar", "not_indexed", "طلب دعم حافز للباحثين عن عمل"),
    ("en", "out_of_domain", "Translate 'good morning' into French"),
    ("en", "out_of_domain", "What time is Maghrib prayer in Jeddah today?"),
    ("ar", "out_of_domain", "وش أفضل مطعم مندي في الرياض؟"),
    ("ar", "out_of_domain", "اشرح لي نظرية النسبية"),
    ("mixed", "out_of_domain", "best laptop for programming ميزانيتي ٤ آلاف"),
]

# -------------------------------------------------------------- unsupported
UNSUPPORTED = [
    ("en", "out_of_domain", "What's the best pizza in Riyadh?"),
    ("en", "out_of_domain", "Who will win the football match tonight?"),
    ("en", "out_of_domain", "Write my university application essay for me."),
    ("en", "out_of_domain", "What is the integral of x squared?"),
    ("en", "out_of_domain", "I have a headache and fever, what disease do I have?"),
    ("en", "out_of_domain", "Which stock should I buy this week?"),
    ("en", "out_of_domain", "Tell me a joke"),
    ("en", "out_of_domain", "Best hotels to stay in AlUla"),
    ("en", "out_of_domain", "Write a Python function to sort a list"),
    ("ar", "out_of_domain", "كيف الطقس في جدة بكرة؟"),
    ("ar", "out_of_domain", "طريقة عمل الكبسة"),
    ("ar", "out_of_domain", "أفضل جوال أشتريه هالسنة"),
    ("ar", "out_of_domain", "من هو أفضل لاعب في الدوري السعودي؟"),
    ("ar", "out_of_domain", "اكتب لي قصيدة عن الرياض"),
    ("en", "not_indexed", "How do I renew my driving licence?"),
    ("en", "not_indexed", "How can I pay my traffic violations?"),
    ("en", "not_indexed", "How do I renew my iqama?"),
    ("en", "not_indexed", "How do I issue an exit re-entry visa for my worker?"),
    ("en", "not_indexed", "Book an appointment at a primary health care centre"),
    ("en", "not_indexed", "How do I register my employees in GOSI?"),
    ("en", "not_indexed", "How do I renew my car registration (istimara)?"),
    ("en", "not_indexed", "How do I get a sick leave report?"),
    ("ar", "not_indexed", "كيف أجدد رخصة القيادة؟"),
    ("ar", "not_indexed", "تجديد الإقامة"),
    ("ar", "not_indexed", "كيف أسدد المخالفات المرورية؟"),
    ("ar", "not_indexed", "إصدار هوية وطنية لأول مرة"),
    ("ar", "not_indexed", "حجز موعد في مركز صحي"),
    ("ar", "not_indexed", "تسجيل العاملين في التأمينات الاجتماعية"),
]


def build_rows():
    rows = []
    for fid, split, (src, title), acc, qs in F:
        for lang, style, q in qs:
            rows.append(dict(family=fid, split=split, query=q, language=lang, style=style, supported=1,
                             expected_source=src, expected_title=title,
                             acceptable="||".join(f"{s}::{t}" for s, t in acc), fresh=int(fid in FRESH)))
    for i, (lang, kind, q) in enumerate(UNSUPPORTED):
        split = ("dev", "val", "test", "test", "dev")[i % 5]
        rows.append(dict(family=f"unsupported-{i:02d}", split=split, query=q, language=lang, style=kind,
                         supported=0, expected_source="", expected_title="", acceptable="", fresh=0))
    for i, (lang, kind, q) in enumerate(UNSUPPORTED_FRESH):
        split = ("val", "test", "test")[i % 3]
        rows.append(dict(family=f"unsupported-fresh-{i:02d}", split=split, query=q, language=lang, style=kind,
                         supported=0, expected_source="", expected_title="", acceptable="", fresh=1))
    for n, r in enumerate(rows, 1):
        r["query_id"] = f"v2q{n:03d}"
    return rows


if __name__ == "__main__":
    out = Path(__file__).with_name("benchmark_v2.csv")
    cols = ["query_id", "family", "split", "query", "language", "style", "supported", "expected_source",
            "expected_title", "acceptable", "fresh"]
    rows = build_rows()
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    import collections
    print(len(rows), collections.Counter((r["split"], r["supported"]) for r in rows))
