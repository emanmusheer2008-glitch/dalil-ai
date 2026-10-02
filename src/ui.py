"""Presentation helpers for the Streamlit app: styles, translations, HTML cards.

All dynamic text is HTML-escaped before it is inserted into markup.
"""
from __future__ import annotations

import html

T = {
    "en": {
        "tagline": "Bilingual guide to official Saudi public-service information",
        "hero_sub": "Ask in Arabic or English. Dalil searches its verified official sources and shows you "
                    "the matching service, the official text and the link to the source — or tells you "
                    "when it doesn't know.",
        "ask_label": "Your question",
        "ask_placeholder": "e.g. How do I transform my establishment into a company?",
        "ask_button": "Ask Dalil",
        "examples": "Try an example",
        "answer_heading": "Best match from official sources",
        "agency": "Agency",
        "category": "Category",
        "why": "Why this matched — passages from the official page",
        "official_page": "Open official page",
        "other_lang": "العربية",
        "collected": "Captured from the official page on",
        "modified": "page last modified",
        "related": "Also possibly relevant",
        "similarity": "similarity",
        "threshold_note": "Answer threshold",
        "scope_note": "Dalil currently indexes {n} Ministry of Commerce e-services. Questions outside this "
                      "corpus are declined rather than guessed.",
        "disclaimer": "Dalil AI is an independent educational project and is not an official Saudi government "
                      "service. Always verify important information through the linked official source.",
        "nav_ask": "Ask Dalil", "nav_explore": "Explore services", "nav_kb": "Knowledge base",
        "nav_eval": "Evaluation", "nav_about": "About",
        "insufficient_title": "Not enough verified information",
        "try_instead": "Browse the indexed services in Explore services, or check the unified national platform "
                       "my.gov.sa directly.",
        "latency": "retrieved in {ms:.0f} ms",
    },
    "ar": {
        "tagline": "دليل ثنائي اللغة للمعلومات الرسمية عن الخدمات العامة في السعودية",
        "hero_sub": "اسأل بالعربية أو الإنجليزية. يبحث دليل في مصادره الرسمية الموثقة ويعرض لك الخدمة المطابقة "
                    "ونصها الرسمي ورابط المصدر — أو يخبرك بأنه لا يعرف.",
        "ask_label": "سؤالك",
        "ask_placeholder": "مثال: كيف أحول مؤسستي الفردية إلى شركة؟",
        "ask_button": "اسأل دليل",
        "examples": "جرّب مثالاً",
        "answer_heading": "أفضل تطابق من المصادر الرسمية",
        "agency": "الجهة",
        "category": "التصنيف",
        "why": "سبب التطابق — مقاطع من الصفحة الرسمية",
        "official_page": "فتح الصفحة الرسمية",
        "other_lang": "English",
        "collected": "نُسخ من الصفحة الرسمية بتاريخ",
        "modified": "آخر تعديل للصفحة",
        "related": "خدمات قد تكون ذات صلة",
        "similarity": "درجة التشابه",
        "threshold_note": "حد الإجابة",
        "scope_note": "يفهرس دليل حالياً {n} خدمة إلكترونية لوزارة التجارة. الأسئلة خارج هذا النطاق يُعتذر عنها "
                      "بدلاً من التخمين.",
        "disclaimer": "دليل مشروع تعليمي مستقل وليس خدمة حكومية رسمية. تحقّق دائماً من المعلومات المهمة عبر "
                      "المصدر الرسمي المرفق.",
        "nav_ask": "اسأل دليل", "nav_explore": "استعرض الخدمات", "nav_kb": "قاعدة المعرفة",
        "nav_eval": "التقييم", "nav_about": "عن المشروع",
        "insufficient_title": "لا توجد معلومات موثقة كافية",
        "try_instead": "استعرض الخدمات المفهرسة، أو راجع المنصة الوطنية الموحدة my.gov.sa مباشرة.",
        "latency": "زمن البحث {ms:.0f} ملّي ثانية",
    },
}

# Examples are benchmark questions that the evaluation answered correctly, plus
# one question Dalil is expected to decline (not in the indexed corpus).
EXAMPLES = {
    "en": [
        "Steps to set up a limited liability company in Saudi Arabia",
        "My professional license expired. How do I renew it?",
        "How do I register a franchise?",
        "Permit to import non-hazardous chemicals",
        "How do I renew my passport?",
    ],
    "ar": [
        "خطوات تأسيس شركة ذات مسؤولية محدودة",
        "التأكيد السنوي لبيانات السجل التجاري للمؤسسة كم رسومه؟",
        "تحويل المؤسسة الفردية إلى شركة",
        "الاستعلام عن المنتجات المعيبة",
        "كيف أجدد جواز السفر؟",
    ],
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');
:root{
  --dl-ink:#1C2521; --dl-muted:#5B6660; --dl-green:#0F6B5A; --dl-green-2:#0B4F43;
  --dl-sand:#C8A45A; --dl-card:#FFFFFF; --dl-line:#E3DED2; --dl-soft:#F1EEE6; --dl-warn:#8A5A12;
}
html, body, [class*="css"], .stMarkdown, .stTextInput input, .stTextArea textarea, button {
  font-family: 'IBM Plex Sans Arabic', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
}
.block-container{max-width:1080px; padding-top:3.6rem; padding-bottom:4rem;}
#MainMenu, footer {visibility:hidden;}
.dl-hero{background:linear-gradient(135deg,var(--dl-green-2),var(--dl-green) 60%,#15836E);
  color:#fff;border-radius:20px;padding:34px 34px 30px;margin:6px 0 22px;position:relative;overflow:hidden}
.dl-hero:after{content:"";position:absolute;inset:auto -60px -80px auto;width:260px;height:260px;
  border-radius:50%;border:34px solid rgba(200,164,90,.25)}
.dl-hero h1{font-size:2.3rem;margin:0 0 4px;color:#fff;font-weight:700;letter-spacing:-.5px}
.dl-hero .dl-tag{color:#E9D9AE;font-weight:500;margin-bottom:12px}
.dl-hero p{max-width:720px;opacity:.93;font-size:1.02rem;line-height:1.7;margin:0}
.dl-card{background:var(--dl-card);border:1px solid var(--dl-line);border-radius:16px;padding:22px 24px;
  margin:10px 0 14px;box-shadow:0 1px 2px rgba(20,30,25,.04)}
.dl-card h3{margin:.1rem 0 .4rem;font-size:1.35rem;color:var(--dl-ink)}
.dl-meta{color:var(--dl-muted);font-size:.9rem;margin-bottom:10px}
.dl-pill{display:inline-block;background:var(--dl-soft);border:1px solid var(--dl-line);color:var(--dl-ink);
  border-radius:999px;padding:2px 10px;font-size:.8rem;margin:0 6px 6px 0}
.dl-pill.ok{background:#E4F2EC;border-color:#BFDDCF;color:var(--dl-green-2)}
.dl-pill.maybe{background:#FBF1DD;border-color:#EBD3A2;color:var(--dl-warn)}
.dl-field{margin:12px 0 0}
.dl-field .lbl{font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;color:var(--dl-green);font-weight:600}
.dl-field .val{white-space:pre-wrap;line-height:1.75;color:var(--dl-ink);margin-top:2px}
.dl-ev{border-inline-start:3px solid var(--dl-sand);background:#FCFAF4;padding:8px 12px;margin:8px 0;
  border-radius:6px;white-space:pre-wrap;line-height:1.7;font-size:.93rem}
.dl-ev .sec{font-size:.75rem;color:var(--dl-muted);font-weight:600}
.dl-btn{display:inline-block;background:var(--dl-green);color:#fff !important;text-decoration:none !important;
  padding:8px 16px;border-radius:10px;font-weight:600;margin:12px 8px 0 0;font-size:.92rem}
.dl-btn.ghost{background:transparent;color:var(--dl-green) !important;border:1px solid var(--dl-green)}
.dl-note{background:#FBF1DD;border:1px solid #EBD3A2;color:#5E4210;border-radius:10px;padding:8px 12px;
  font-size:.88rem;margin:8px 0}
.dl-refuse{background:#fff;border:1px dashed #C9B98F;border-radius:16px;padding:22px 24px;margin:10px 0}
.dl-refuse h3{margin:0 0 6px;color:var(--dl-warn)}
.dl-foot{color:var(--dl-muted);font-size:.82rem;border-top:1px solid var(--dl-line);margin-top:34px;padding-top:12px}
.dl-kpi{background:#fff;border:1px solid var(--dl-line);border-radius:14px;padding:14px 16px}
.dl-kpi .n{font-size:1.7rem;font-weight:700;color:var(--dl-green-2)}
.dl-kpi .l{font-size:.82rem;color:var(--dl-muted)}
[dir="rtl"]{text-align:right}
[dir="rtl"] .dl-field .lbl{letter-spacing:0;text-transform:none}
@media (max-width: 640px){
  .dl-hero{padding:24px 20px}
  .dl-hero h1{font-size:1.8rem}
  .dl-card{padding:16px}
}
</style>
"""


def esc(s) -> str:
    return html.escape(str(s)) if s is not None else ""


def dir_of(lang: str) -> str:
    return "rtl" if lang == "ar" else "ltr"


def kpi(n, label) -> str:
    return f'<div class="dl-kpi"><div class="n">{esc(n)}</div><div class="l">{esc(label)}</div></div>'


def service_card(sa, t: dict, show_evidence: bool = True, heading: str | None = None) -> str:
    d = dir_of(sa.shown_lang)
    pill_cls = "ok" if sa.confidence_label in ("Strong match", "تطابق قوي") else "maybe"
    parts = [f'<div class="dl-card" dir="{d}" lang="{sa.shown_lang}">']
    if heading:
        parts.append(f'<div class="dl-meta">{esc(heading)}</div>')
    parts.append(f"<h3>{esc(sa.title)}</h3>")
    meta = [esc(sa.agency)]
    if sa.category:
        meta.append(esc(sa.category))
    parts.append(f'<div class="dl-meta">{" · ".join(meta)}</div>')
    parts.append(f'<span class="dl-pill {pill_cls}">{esc(sa.confidence_label)}</span>'
                 f'<span class="dl-pill">{esc(t["similarity"])} {sa.score:.2f}</span>')
    if sa.language_note:
        parts.append(f'<div class="dl-note">{esc(sa.language_note)}</div>')
    for _, label, text in sa.fields:
        parts.append(f'<div class="dl-field"><div class="lbl">{esc(label)}</div><div class="val">{esc(text)}</div></div>')
    if sa.official_url:
        parts.append(f'<a class="dl-btn" href="{esc(sa.official_url)}" target="_blank" rel="noopener">'
                     f'{esc(t["official_page"])} ↗</a>')
    if sa.other_lang_url:
        parts.append(f'<a class="dl-btn ghost" href="{esc(sa.other_lang_url)}" target="_blank" rel="noopener">'
                     f'{esc(t["other_lang"])} ↗</a>')
    prov = f'{esc(t["collected"])} {esc((sa.date_collected or "")[:10])}'
    if sa.source_last_modified:
        prov += f' · {esc(t["modified"])}: {esc(sa.source_last_modified)}'
    parts.append(f'<div class="dl-meta" style="margin-top:12px">{prov}</div>')
    parts.append("</div>")
    return "".join(parts)


def evidence_block(sa, t: dict, ui_lang: str) -> str:
    rows = []
    for label, text, score, lang in sa.evidence:
        rows.append(f'<div class="dl-ev" dir="{dir_of(lang)}" lang="{lang}"><div class="sec">{esc(label)} · '
                    f'{score:.2f}</div>{esc(text)}</div>')
    return "".join(rows)
