"""Generic parser for official service-detail pages (V2 sources).

Government sites use different CMSs (Drupal, SharePoint, custom), but service
pages share a vocabulary: a title, a description, and labelled blocks such as
"Steps", "Required Documents", "Conditions", "Service Fees", "Duration",
"Target Audience" -- in English and Arabic. This parser:

1. walks the page's main content in document order and splits it into
   ``(label, text)`` sections, where a *label* is a heading, a bold lead-in
   (``<strong>Label</strong> text``), a definition term, a Drupal field label,
   or a tab button (tab labels are matched to their panels by ARIA ids, or by
   order when ids are missing);
2. maps labels to schema fields with a bilingual keyword table;
3. keeps the text **verbatim** (Unicode/whitespace clean-up only).

Anything the page does not label stays ``None``. Unlabelled boilerplate
(navigation, "related services", ratings, share buttons) is dropped.
Per-source differences live in :data:`SOURCES` below.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from src.ingestion.base import BaseLoader
from src.ingestion.normalization import clean_text
from src.schema import ServiceRecord
from src.utils.arabic import normalize_arabic


@dataclass(frozen=True)
class SourceInfo:
    agency_en: str
    agency_ar: str
    domain: str
    channel: str


SOURCES = {
    "hrsd": SourceInfo("Ministry of Human Resources and Social Development", "وزارة الموارد البشرية والتنمية الاجتماعية",
                       "hrsd.gov.sa", "Online (HRSD / Qiwa / related platforms)"),
    "zatca": SourceInfo("Zakat, Tax and Customs Authority", "هيئة الزكاة والضريبة والجمارك", "zatca.gov.sa",
                        "Online (ZATCA portal)"),
    "moe": SourceInfo("Ministry of Education", "وزارة التعليم", "moe.gov.sa", "Online (Ministry of Education)"),
    "haj": SourceInfo("Ministry of Hajj and Umrah", "وزارة الحج والعمرة", "haj.gov.sa", "Online (Ministry of Hajj and Umrah)"),
    "sfda": SourceInfo("Saudi Food and Drug Authority", "الهيئة العامة للغذاء والدواء", "sfda.gov.sa", "Online (SFDA)"),
    "momah": SourceInfo("Ministry of Municipalities and Housing", "وزارة البلديات والإسكان", "momah.gov.sa",
                        "Online (Balady / MOMAH)"),
    "moj": SourceInfo("Ministry of Justice", "وزارة العدل", "moj.gov.sa", "Online (Najiz / Ministry of Justice)"),
    "mofa": SourceInfo("Ministry of Foreign Affairs", "وزارة الخارجية", "mofa.gov.sa", "Online (Ministry of Foreign Affairs)"),
    "chi": SourceInfo("Council of Health Insurance", "مجلس الضمان الصحي", "chi.gov.sa", "Online (Council of Health Insurance)"),
}

# Label keywords (lower-cased English / normalised Arabic) -> schema field.
# Order matters: the first matching rule wins.
LABEL_RULES: list[tuple[str, str]] = [
    ("steps", r"\bsteps?\b|procedure|how to (apply|get)|خطوات|الخطوه|اجراءات الحصول|طريقه الحصول|مراحل"),
    ("required_documents", r"documents?|attachments?|مستندات|الوثاءق|المرفقات|الاوراق المطلوبه"),
    ("eligibility", r"eligib|who can apply|الاهليه|الفءات المستحقه|من يحق"),
    ("requirements", r"conditions?|requirements?|terms of use|controls|prerequisit|\bterms\b|شروط|الشروط|اشتراطات|ضوابط|متطلبات"),
    ("fees", r"\bfees?\b|\bcost\b|charges?|price|رسوم|تكلفه|المقابل المالي"),
    ("processing_time", r"duration|processing time|time to|service level|expected time|مده|زمن|وقت تنفيذ|اتفاقيه مستوي"),
    ("target_audience", r"beneficiar|target (audience|group)|audience|who is it for|المستفيد|الفءه المستهدفه|الفءات المستهدفه"),
    ("service_languages", r"language|لغه"),
    ("notes", r"\bnotes?\b|notice|important|alert|remark|ملاحظ|تنبيه|تنويه|هام"),
    ("description", r"description|about (the )?(e-)?service|overview|service brief|وصف|نبذه|عن الخدمه"),
]
DROP_LABELS = re.compile(
    r"related services|faq|frequently asked|rate|rating|share|mobile app|contact via|useful resources|user manual|"
    r"service link|start (the )?service|release date|payment channels?|sms service|خدمات ذات صله|الاسءله الشاءعه|"
    r"تقييم|شارك|تطبيقات|دليل المستخدم|رابط الخدمه|ابدا الخدمه|تاريخ اصدار",
)
STOP_TEXT = re.compile(r"^(related services|خدمات ذات صلة|الخدمات ذات العلاقة|rate this service|قيّم الخدمة)", re.I)

_BLOCK = {"p", "div", "li", "ul", "ol", "section", "article", "table", "tr", "td", "th", "dd", "dt", "br", "h1",
          "h2", "h3", "h4", "h5", "h6", "header", "footer", "main"}


def label_field(label: str) -> str | None:
    lab = normalize_arabic(label).lower().strip(" :：-")
    if not lab or len(lab) > 60 or DROP_LABELS.search(lab):
        return None
    for fld, rx in LABEL_RULES:
        if re.search(rx, lab):
            return fld
    return None


def _text_of(el: Tag) -> str:
    """Block-aware text: list items become '- item' lines."""
    out: list[str] = []

    def walk(n):
        if isinstance(n, Comment):
            return
        if isinstance(n, NavigableString):
            out.append(str(n))
            return
        if not isinstance(n, Tag):
            return
        if n.name in ("script", "style", "noscript", "svg", "button"):
            return
        block = n.name in _BLOCK
        if block:
            out.append("\n")
        if n.name == "li":
            out.append("- ")
        for c in n.children:
            walk(c)
        if block:
            out.append("\n")

    walk(el)
    return clean_text("".join(out)) or ""


def _is_label_el(el: Tag) -> bool:
    if el.name in ("h2", "h3", "h4", "h5", "h6", "dt"):
        return True
    cls = " ".join(el.get("class", []))
    if "field__label" in cls:
        return True
    if el.name in ("p", "div", "span") and re.search(r"fw-bold|font-bold|\blabel\b|-label|title", cls) \
            and not el.find(["p", "div", "ul", "ol", "table"]):
        # "<p class='fw-bold'>Service Fees</p><p>Free</p>" label/value pairs (MOMAH and similar)
        txt = el.get_text(" ", strip=True)
        nxt = el.find_next_sibling()
        return 0 < len(txt) <= 40 and nxt is not None and label_field(txt) is not None
    if el.name in ("strong", "b"):
        # a bold lead-in at the start of its block
        parent = el.parent
        if parent is None:
            return False
        first = next((c for c in parent.children if not (isinstance(c, NavigableString) and not c.strip())), None)
        return first is el and len(el.get_text(strip=True)) <= 60
    return False


_RELATED = re.compile(r"^\s*(related services|related e-?services|similar services|you can view the affiliated services|"
                      r"خدمات ذات صلة|الخدمات ذات الصلة|الخدمات ذات العلاقة|خدمات مشابهة|يمكنك الاطلاع على الخدمات التابعة)"
                      r"\s*\.?\s*$", re.I)
_ESCAPED_TAG = re.compile(r"<\s*(ol|ul|li|p|div|br|strong|b|span|table)\b", re.I)


def _unescape_rich_text(main: Tag) -> None:
    """Some CMS templates (e.g. MoJ) print rich text HTML-escaped, so the page shows literal
    ``<ol><li>…`` text. Re-parse such text nodes as HTML so lists and paragraphs survive.
    This changes markup only; the words are kept verbatim."""
    for node in list(main.find_all(string=_ESCAPED_TAG)):
        if isinstance(node, NavigableString) and node.parent is not None and \
                node.parent.name not in ("script", "style"):
            frag = BeautifulSoup(str(node), "html.parser")
            for c in list(frag.contents):
                node.insert_before(c)
            node.extract()


def extract_sections(main: Tag) -> tuple[str | None, list[tuple[str, str]]]:
    """Return (title, [(label, text), ...]) in document order."""
    # SharePoint wraps the whole page in one <form>: unwrap forms instead of deleting them,
    # otherwise the entire service content is thrown away (MoJ / CHI captures).
    for frm in main.find_all("form"):
        frm.unwrap()
    for c in main.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()
    # read the title before any clean-up (some themes put the <h1> inside breadcrumb/header wrappers)
    h1 = main.find("h1")
    title = clean_text(h1.get_text(" ", strip=True)) if h1 else None
    # everything after a "Related services" heading belongs to OTHER services (cards with their own
    # titles/descriptions) -- drop it so another service's text can never be attributed to this one
    marker = main.find(lambda t: isinstance(t, Tag) and t.name in ("h2", "h3", "h4", "h5", "p", "div", "span")
                       and not t.find(True) and _RELATED.match(t.get_text(" ", strip=True) or ""))
    if marker is not None:
        for el in list(marker.find_all_next()):
            if el.parent is not None and not el.decomposed:
                el.decompose()
        marker.decompose()
    # small hidden helpers (SharePoint field labels, hidden URLs/tags) are not content
    for el in main.select('[style*="display:none"], [style*="display: none"]'):
        if len(el.get_text(strip=True)) < 200 and not el.find(["ul", "ol", "table"]):
            el.decompose()
    for bad in main.select("script, style, noscript, svg, iframe, nav, footer, .swiper, [class*=related], "
                           "[id*=related], [class*=rating], [class*=share], .ms-hide, [class*=carousel], "
                           "[class*=carousal], [class*=services_card], [class*=service-card]"):
        bad.decompose()
    # breadcrumb trails are navigation, but some themes wrap the <h1> and lead paragraph in a
    # "region-breadcrumb" container -> remove only breadcrumb blocks that do not hold the title
    for bc in main.select("[class*=breadcrumb]"):
        if not bc.decomposed and bc.find("h1") is None:
            bc.decompose()
    _unescape_rich_text(main)

    sections: list[tuple[str, str]] = []

    # 1) tab widgets: tab buttons label their panels. Supported: ARIA (role=tab + aria-controls),
    #    Bootstrap pills/tabs (data-toggle + href="#id") and plain in-page anchors (href="#id")
    #    whose label is a known field name (CHI) -- skip-links etc. are ignored.
    tabs = list(main.select('[role="tab"], [data-toggle="pill"], [data-toggle="tab"], [data-bs-toggle="tab"], '
                            '[data-bs-toggle="pill"]'))
    tabs += [a for a in main.select('a[href^="#"]') if a not in tabs
             and len(a.get_text(strip=True)) <= 40 and label_field(a.get_text(" ", strip=True))]
    for tab in tabs:
        panel = None
        for ref in (tab.get("data-bs-target"), tab.get("data-target"), tab.get("aria-controls"), tab.get("href")):
            ref = (ref or "").lstrip("#").strip()
            if ref and (panel := main.find(id=ref)) is not None:
                break
        if panel is None or panel.find(lambda t: t is tab):
            continue
        label = clean_text(tab.get_text(" ", strip=True))
        # a heading/title repeating the tab label inside the panel is not content
        for h in panel.find_all(True):
            if not h.find(True) and clean_text(h.get_text(" ", strip=True)) == label:
                h.decompose()
        sections.append((label, _text_of(panel)))
        panel.decompose()
    for tl in main.select('[role="tablist"], ul.nav-pills, ul.nav-tabs'):
        tl.decompose()
    # tab panels that no tab label points to have no reliable meaning -> drop rather than guess
    for pane in main.select('[role="tabpanel"], .tab-pane'):
        pane.decompose()
    # fixed-id tabs (Ministry of Commerce style)
    for pid, lab in (("steps", "Steps"), ("terms", "Conditions"), ("document", "Required Documents")):
        el = main.find(id=pid)
        if el is not None:
            sections.append((lab, _text_of(el)))
            el.decompose()

    # 2) walk remaining content: label elements start a new section
    labels = [el for el in main.find_all(True) if _is_label_el(el)]
    for i, lab in enumerate(labels):
        if lab.find_parent(lambda t: t in labels[:i]):
            continue
        label_text = clean_text(lab.get_text(" ", strip=True)) or ""
        if STOP_TEXT.match(label_text):
            break
        parts = []
        # bold lead-in: rest of the same block is the value
        if lab.name in ("strong", "b") and lab.parent is not None:
            full = _text_of(lab.parent)
            parts.append(full[len(label_text):].strip(" :：\n") if full.startswith(label_text) else full)
        # following siblings until next label
        sib = lab.next_sibling if lab.name not in ("strong", "b") else lab.parent.next_sibling
        while sib is not None:
            if isinstance(sib, Tag):
                if _is_label_el(sib) or sib.find(_is_label_el):
                    break
                parts.append(_text_of(sib))
            elif isinstance(sib, NavigableString) and sib.strip():
                parts.append(sib.strip())
            sib = sib.next_sibling
        text = clean_text("\n".join(p for p in parts if p))
        if text:
            sections.append((label_text, text))
    return title, sections


_UI_NOISE = re.compile(r"display settings|dark mode|high brightness|cookies?|إعدادات الرؤية|اعدادات الرؤيه|الوضع الليلي|"
                       r"for more information you may review help|للحصول على معلومات إضافية، يمكنك مراجعة المساعدة|for any inquiries|يمكنك التعرف على الخدمات المتاحة|you can learn about the available services|no feedback has been submitted|لم يتم تقديم|thank you! your response|for any inquiries or comments|شكرا[ً]? لك|لأي استفسار|الوضع القابل للوصول|accessible mode|was this (page|content) useful|هل كانت هذه الصفحة مفيدة|last (update|modified)|آخر تحديث", re.I)


def first_paragraph_after_title(main: Tag, title: str | None) -> str | None:
    """Fallback description: the first substantial paragraph *after* the page's <h1>
    (content before the title is page chrome such as display/accessibility widgets)."""
    h1 = main.find("h1")
    for tags in (["p"], ["p", "div"]):  # real paragraphs first, then leaf divs
        candidates = h1.find_all_next(tags) if h1 is not None else main.find_all(tags)
        for p in candidates:
            if p.find(["p", "div"]) or p.find_parent(["button", "a"]):
                continue
            t = clean_text(p.get_text(" ", strip=True))
            if t and len(t) >= 40 and t != title and not _UI_NOISE.search(t):
                return t
    return None


_PLACEHOLDER = re.compile(
    r"^\s*(none|n/?a|-|no content available|this information is currently unavailable\.?|not available|"
    r"لا يوجد|لا ينطبق|لا ?يوجد محتوى|لا ?يوجد محتوى متاح|غير متوفر|هذه المعلومات غير متوفرة حاليا\.?)\s*$", re.I)
_NUMBERS_ONLY = re.compile(r"^[\d\s|.,:\-–]*$")
_SLA_WORDS = re.compile(r"service level agreement|اتفاقي[ةه] مستو[ىي] الخدم[ةه]", re.I)


def _meaningful(text: str) -> bool:
    """Reject values that are only numbers/punctuation once a bare 'Service Level Agreement'
    link label is removed -- e.g. MoJ durations whose units are icons ('2 | 4'). Showing
    those would invite a guess about the unit, so they are left absent."""
    return not _NUMBERS_ONLY.match(_SLA_WORDS.sub("", text))


def _strip_heading_lines(text: str) -> str:
    """Drop leading lines that are just a section heading repeated inside the value
    ("Description", "Service Steps", "خطوات التقديم على / الخدمة")."""
    lines = (text or "").split("\n")
    while len(lines) > 1 and len(lines[0]) <= 40 and not lines[0].startswith("-") and (
            label_field(lines[0]) or label_field(lines[0] + " " + lines[1]) and len(lines[1]) <= 12
            or lines[0].strip().lower() in ("page content", "محتوى الصفحة")):
        if label_field(lines[0] + " " + lines[1]) and len(lines[1]) <= 12 and not label_field(lines[0]):
            lines = lines[2:]
        else:
            lines = lines[1:]
    return "\n".join(lines).strip()


def parse_page(html: str, page_title: str | None = None) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find(["main", "article"]) or soup
    title, sections = extract_sections(main)
    if not title and page_title:
        # browser <title> usually appends the site name: "Service - Ministry ..." / "Service | Site"
        title = clean_text(re.split(r"\s[|\-–]\s", page_title)[0])
    out: dict = {"title": title, "sections": sections}
    for label, text in sections:
        fld = label_field(label)
        text = _strip_heading_lines(text)
        if not fld or not text or _PLACEHOLDER.match(text) or not _meaningful(text):
            continue
        if out.get(fld):
            if text not in out[fld]:
                out[fld] = out[fld] + "\n" + text
        else:
            out[fld] = text
    if not out.get("description"):
        # a block the CMS itself marks as the description (ZATCA "service-description", CHI "description-txt")
        for el in main.find_all(class_=re.compile(r"(^|[-_])(service-)?desc(ription)?([-_]txt|[-_]text)?$", re.I)):
            if el.find_parent(class_=re.compile(r"card|slide|teaser|views-row")):
                continue  # a description inside a card belongs to another (listed) service
            t = _text_of(el)
            if t and len(t) >= 30 and not _UI_NOISE.search(t):
                out["description"] = t
                break
    if not out.get("description"):
        out["description"] = first_paragraph_after_title(main, title)
    return out


_ARABIC = re.compile(r"[\u0600-\u06FF]")


def _is_arabic(text: str | None) -> bool:
    return bool(text) and len(_ARABIC.findall(text)) >= 0.3 * len(re.sub(r"\W", "", text))


class GenericServiceLoader(BaseLoader):
    """Load one ``dalil_v2_<source>.json`` capture file (+ its ``dalil_v2_<source>_*.json`` supplements).

    A supplement (e.g. ``dalil_v2_sfda_arfix.json``) is a later, separately exported re-capture of
    pages that failed in the original file. It may only *fill* an (id, language) page whose original
    capture was not HTTP 200; it never overwrites a successful original page. Both raw files stay
    unchanged on disk, and each page keeps its own URL, capture time and SHA-256.
    """

    name = "generic_service_pages"

    def __init__(self, path, supplements=None):
        super().__init__(path)
        p = Path(path)
        self.supplements = (sorted(p.parent.glob(p.stem + "_*.json")) if supplements is None
                            else [Path(x) for x in supplements])
        self.page_stats: dict = {}

    def describe(self) -> str:
        sup = "+" + ",".join(x.name for x in self.supplements) if self.supplements else ""
        return f"{self.name}:{self.path.name}{sup}"

    def load(self) -> list[ServiceRecord]:
        cap = json.loads(Path(self.path).read_text(encoding="utf-8"))
        src = cap["source"]
        info = SOURCES[src]
        st = {"pages_in_capture": 0, "pages_http_200": 0, "pages_failed_http": 0,
              "pages_filled_by_supplement": 0, "pages_error_or_empty": 0, "pages_wrong_language": 0}
        pages_all: dict[tuple[str, str], dict] = {}
        for item in cap["records"]:
            st["pages_in_capture"] += 1
            pages_all[(str(item["id"]).lower(), item["lang"])] = item
        for sup in self.supplements:
            for item in json.loads(sup.read_text(encoding="utf-8")).get("records", []):
                k = (str(item["id"]).lower(), item["lang"])
                if k in pages_all and pages_all[k].get("http_status") == 200 and pages_all[k].get("main_html"):
                    continue  # never replace a good original page
                if item.get("http_status") == 200 and item.get("main_html"):
                    pages_all[k] = {**item, "_supplement": sup.name}
                    st["pages_filled_by_supplement"] += 1
        by_id: dict[str, dict[str, dict]] = {}
        for (sid, lang), item in pages_all.items():
            if item.get("http_status") != 200 or not item.get("main_html"):
                st["pages_failed_http"] += 1
                continue
            st["pages_http_200"] += 1
            by_id.setdefault(sid, {})[lang] = item
        records = []
        for sid in sorted(by_id):
            pages = by_id[sid]
            dates = sorted(p["fetched_at"] for p in pages.values())
            rec = ServiceRecord(
                service_id=f"{src}-{hashlib.sha1(sid.encode()).hexdigest()[:10]}",
                source_type="official_web_page",
                source_domain=info.domain,
                verification_status="verified_official_capture",
                date_collected=dates[0],
                last_verified=dates[-1][:10],
                source_record_id=sid,
                service_channel=info.channel,
                provenance_notes=f"Captured from the official {info.agency_en} service pages ({info.domain}); "
                                 "text stored verbatim apart from Unicode/whitespace normalisation.",
            )
            hashes, sups = [], []
            for lang, page in sorted(pages.items()):
                parsed = parse_page(page["main_html"], page.get("page_title"))
                # Guard against error / placeholder pages served with HTTP 200
                if not parsed.get("title") or re.search(r"not found|404|غير متاح|page cannot be found", parsed["title"], re.I):
                    st["pages_error_or_empty"] += 1
                    continue
                # An "Arabic" page must actually be in Arabic (and vice versa) -- never present one
                # language's text as the other's official version.
                body = " ".join(str(parsed.get(f) or "") for f in ("title", "description"))
                if (lang == "ar") != _is_arabic(body):
                    st["pages_wrong_language"] += 1
                    continue
                setattr(rec, f"title_{lang}", parsed["title"])
                setattr(rec, f"agency_{lang}", info.agency_en if lang == "en" else info.agency_ar)
                setattr(rec, f"official_url_{lang}", page["url"])
                for name in ("description", "eligibility", "requirements", "required_documents", "steps", "fees",
                             "processing_time", "target_audience", "service_languages", "notes"):
                    setattr(rec, f"{name}_{lang}", parsed.get(name))
                hashes.append(f"{lang}:{hashlib.sha256(page['main_html'].encode('utf-8')).hexdigest()}")
                if page.get("_supplement"):
                    sups.append(f"{lang}:{page['_supplement']}")
            rec.source_sha256 = ";".join(hashes) or None
            rec.extra = {"source": src, **({"supplement_pages": sups} if sups else {})}
            records.append(rec)
        self.page_stats = st
        return records
