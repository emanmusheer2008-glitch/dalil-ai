"""Generic V2 page parser. Fixtures are synthetic HTML that mimic the *structure*
of the captured CMS layouts (their wording is invented and never indexed)."""
import json

from src.ingestion.generic_service_page import GenericServiceLoader, label_field, parse_page

DRUPAL_TABS = """
<main><article>
<h1 class="title page-title">Test Labour Service</h1>
<div class="field field--name-body field__item"><p>This electronic service lets an establishment test the parser with a long enough description.</p></div>
<div class="field"><div class="field__label">Target Audience</div><div class="field__items"><div class="field__item">Business owners</div></div></div>
<div class="field"><div class="field__label">Duration of service</div><div class="field__item"><div>Immediate</div></div></div>
<div class="field"><div class="field__label">Service Fees</div><div class="field__item"><div>Free</div></div></div>
<div class="field"><div class="field__label">Frequently Asked Questions</div><div class="field__item"><a>FAQ</a></div></div>
<div role="tablist">
  <div role="tab" aria-controls="p1"><span>Steps</span></div>
  <div role="tab" aria-controls="p2"><span>Terms Of Use</span></div>
  <div role="tab" aria-controls="p3"><span>Required Documents</span></div>
</div>
<div id="p1" role="tabpanel"><ul><li>Log in to the platform.</li><li>Submit the request.</li></ul></div>
<div id="p2" role="tabpanel"><p>The establishment must be active.</p></div>
<div id="p3" role="tabpanel"></div>
</article></main>
"""

SHAREPOINT_HEADINGS = """
<div id="contentBox">
<h1>تسجيل تجريبي</h1>
<p>خدمة إلكترونية تجريبية تتيح للمنشآت التسجيل، وهذا النص طويل بما يكفي ليكون وصفاً.</p>
<h4>خطوات الخدمة</h4><p>الدخول إلى البوابة</p><p>تعبئة النموذج</p>
<h3>الفئة المستهدفة</h3><p>الشركات</p>
<h3>تكلفة الخدمة</h3><p>مجانية</p>
<h2>خدمات ذات صلة</h2><p>خدمة أخرى</p>
</div>
"""

BOLD_LEADS = """
<main><h1>Reviewing Test Fees</h1>
<p><strong>Description</strong>The ministry offers an electronic service that allows parents to inquire about test fees.</p>
<p><strong>Controls and Conditions</strong>None</p>
<p><strong>Notice:</strong>All correspondence is archived.</p>
</main>
"""


def test_label_mapping_bilingual():
    assert label_field("Required Documents") == "required_documents"
    assert label_field("المستندات المطلوبة") == "required_documents"
    assert label_field("Terms Of Use") == "requirements"
    assert label_field("شروط الاستخدام") == "requirements"
    assert label_field("Service Fees") == "fees"
    assert label_field("رسوم الخدمة") == "fees"
    assert label_field("Duration of service") == "processing_time"
    assert label_field("الفئة المستهدفة") == "target_audience"
    assert label_field("Related Services") is None
    assert label_field("الأسئلة الشائعة") is None


def test_drupal_fields_and_tabs():
    p = parse_page(DRUPAL_TABS)
    assert p["title"] == "Test Labour Service"
    assert p["description"].startswith("This electronic service lets an establishment")
    assert p["steps"] == "- Log in to the platform.\n- Submit the request."
    assert p["requirements"] == "The establishment must be active."
    assert p.get("required_documents") is None            # empty panel -> not invented
    assert p["fees"] == "Free" and p["processing_time"] == "Immediate"
    assert p["target_audience"] == "Business owners"


def test_sharepoint_headings_arabic_and_stop_at_related():
    p = parse_page(SHAREPOINT_HEADINGS)
    assert p["title"] == "تسجيل تجريبي"
    assert "الدخول إلى البوابة" in p["steps"] and "تعبئة النموذج" in p["steps"]
    assert p["fees"] == "مجانية" and p["target_audience"] == "الشركات"
    assert "خدمة أخرى" not in json.dumps(p, ensure_ascii=False)
    assert p["description"].startswith("خدمة إلكترونية تجريبية")


def test_bold_lead_ins_and_none_values():
    p = parse_page(BOLD_LEADS)
    assert p["description"].startswith("The ministry offers")
    assert p.get("requirements") is None                    # "None" is not a requirement
    assert p["notes"] == "All correspondence is archived."


def test_loader_merges_languages_and_skips_error_pages(tmp_path):
    cap = {"source": "hrsd", "records": [
        {"id": "X1", "lang": "en", "url": "https://www.hrsd.gov.sa/en/ministry-services/services/x1",
         "http_status": 200, "fetched_at": "2026-10-01T00:00:00Z", "page_title": "t", "main_html": DRUPAL_TABS},
        {"id": "x1", "lang": "ar", "url": "https://www.hrsd.gov.sa/ministry-services/services/x1",
         "http_status": 200, "fetched_at": "2026-10-01T00:00:05Z", "page_title": "t",
         "main_html": SHAREPOINT_HEADINGS},
        {"id": "x2", "lang": "en", "url": "https://www.hrsd.gov.sa/en/x2", "http_status": 200,
         "fetched_at": "2026-10-01T00:00:09Z", "main_html": "<main><h1>Page not found</h1></main>"},
    ]}
    path = tmp_path / "dalil_v2_hrsd.json"
    path.write_text(json.dumps(cap, ensure_ascii=False), encoding="utf-8")
    recs = {r.source_record_id: r for r in GenericServiceLoader(path).load()}
    r = recs["x1"]
    assert r.title_en == "Test Labour Service" and r.title_ar == "تسجيل تجريبي"
    assert r.agency_en.startswith("Ministry of Human Resources") and r.source_domain == "hrsd.gov.sa"
    assert r.source_sha256.startswith("ar:") and ";en:" in r.source_sha256
    assert recs["x2"].title_en is None                      # error page not used
