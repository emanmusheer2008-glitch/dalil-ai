import json

from src.ingestion import pipeline
from src.ingestion.csv_loader import LegacySeedCSVLoader
from src.ingestion.document_loader import SavedPageLoader
from src.ingestion.mc_catalog import MCCatalogLoader, parse_article
from src.ingestion.official_open_data import OpenDataFileLoader

# Synthetic HTML that mimics the *structure* of an mc.gov.sa service page.
# It is a test fixture only -- its wording is invented and never indexed.
FIXTURE_EN = """
<article>
 <div>Ministry of Commerce E-Services Test Service</div>
 <h2>Test Service</h2><div>Test Service</div>
 <a>Start</a><span>Merchant</span><span>Commercial Register</span><span>Business sector</span>
 <p>An electronic service provided for testing the parser, long enough to count as a description sentence.</p>
 <div>Service level agreement</div><div>Steps</div><div>Conditions</div><div>Required Documents</div>
 <div id="steps"><ul><li>Log in to the platform.</li><li>Submit the application.</li></ul></div>
 <div id="terms"><ul><li>The registration must be active.</li></ul></div>
 <div id="document"></div>
 <div>Beneficiary Group</div><div>Merchant</div>
 <div>Duration of service</div><div>Immediate</div>
 <div>The service is provided in</div><div>Arabic and English</div>
 <div>Service Fees</div><div>100 riyals</div>
 <div>Pay Methods</div>
 <div>You can call the unified number</div><div>1900</div>
 <div>To contact via e-mail</div><div>cs@example.gov.sa</div>
 <div>Rate: 3 / 5</div><div>Related Services</div><div>Other service</div>
 <div>Last Modified 16 Sep 2026</div>
</article>
"""

FIXTURE_AR = """
<article>
 <div>وزارة التجارة</div><h2>خدمة اختبار</h2><div>خدمة اختبار</div>
 <a>ابدأ الخدمة</a><span>التاجر</span><span>السجل التجاري</span><span>قطاع الأعمال</span>
 <p>خدمة إلكترونية لاختبار المحلل البرمجي، وهذا النص طويل بما يكفي ليعتبر وصفاً للخدمة المقدمة.</p>
 <div>إتفاقية مستوى الخدمة</div>
 <div id="steps"><ul><li>الدخول على المنصة.</li></ul></div>
 <div id="terms"></div><div id="document"></div>
 <div>الفئة المستفيدة</div><div>التاجر</div>
 <div>رسوم الخدمة</div><div>100 ريال</div>
 <div>التقييم: 3 / 5</div>
 <div>آخر تعديل 04 ربيع الثاني 1448</div>
</article>
"""


def test_parse_article_extracts_only_stated_fields():
    p = parse_article(FIXTURE_EN, "en", "Test Service")
    assert p["title"] == "Test Service"
    assert p["description"].startswith("An electronic service provided for testing")
    assert p["steps"] == "- Log in to the platform.\n- Submit the application."
    assert p["requirements"] == "- The registration must be active."
    assert p["required_documents"] is None          # empty tab -> None, not invented
    assert p["fees"] == "100 riyals"
    assert p["processing_time"] == "Immediate"
    assert p["target_audience"] == "Merchant"
    assert p["tags"] == ["Merchant", "Commercial Register", "Business sector"]
    assert p["contact"] == "1900, cs@example.gov.sa"
    assert p["source_last_modified"] == "16 Sep 2026"


def test_parse_article_arabic():
    p = parse_article(FIXTURE_AR, "ar", None)
    assert p["title"] == "خدمة اختبار"
    assert p["fees"] == "100 ريال"
    assert p.get("processing_time") is None
    assert p["source_last_modified"] == "04 ربيع الثاني 1448"


def test_mc_loader_merges_languages_and_keeps_provenance(tmp_path):
    cap = {"records": [
        {"sid": 7, "lang": "en", "url": "https://mc.gov.sa/en/eservices/Pages/ServiceDetails.aspx?sID=7",
         "http_status": 200, "fetched_at": "2026-09-30T10:00:00Z", "page_title": "Test Service",
         "article_html": FIXTURE_EN},
        {"sid": 7, "lang": "ar", "url": "https://mc.gov.sa/ar/eservices/Pages/ServiceDetails.aspx?sID=7",
         "http_status": 200, "fetched_at": "2026-09-30T10:00:05Z", "page_title": None,
         "article_html": FIXTURE_AR},
        {"sid": 8, "lang": "en", "url": "https://mc.gov.sa/x", "http_status": 500,
         "fetched_at": "2026-09-30T10:00:09Z", "article_html": None},
    ]}
    path = tmp_path / "cap.json"
    path.write_text(json.dumps(cap, ensure_ascii=False), encoding="utf-8")
    recs = MCCatalogLoader(path).load()
    assert len(recs) == 1                              # failed page skipped
    r = recs[0]
    assert r.service_id == "mc-7" and r.title_ar == "خدمة اختبار" and r.title_en == "Test Service"
    assert r.official_url_ar.endswith("sID=7") and r.date_collected == "2026-09-30T10:00:00Z"
    assert r.source_sha256.startswith("ar:") and ";en:" in r.source_sha256
    assert r.category_en == "Merchant · Commercial Register · Business sector"
    assert r.extra["page_tags"]["ar"] == ["التاجر", "السجل التجاري", "قطاع الأعمال"]


def test_seed_is_loaded_but_quarantined(tmp_path, monkeypatch):
    seed = tmp_path / "seed.csv"
    seed.write_text(
        "service_id,title,agency,category,description,official_url,source,language\n"
        "1,Some Service,Ministry X,Cat,A description.,https://my.gov.sa/en/services/1,GOV.SA,English\n",
        encoding="utf-8",
    )
    recs = LegacySeedCSVLoader(seed).load()
    assert recs[0].verification_status == "unverified"
    report = pipeline.run([LegacySeedCSVLoader(seed)], write=False)
    assert report["services_indexed"] == 0 and report["services_quarantined"] == 1


def test_saved_page_loader(tmp_path):
    (tmp_path / "p.html").write_text(
        "<html><head><title>T</title></head><body><nav>menu</nav><main><h1>Equivalency</h1>"
        "<p>Official requirement text.</p></main></body></html>", encoding="utf-8")
    (tmp_path / "p.meta.json").write_text(json.dumps({
        "service_key": "moe-eq", "lang": "en", "official_url": "https://www.moe.gov.sa/en/x",
        "agency": "Ministry of Education", "date_saved": "2026-09-30T10:00:00Z"}), encoding="utf-8")
    [rec] = SavedPageLoader(tmp_path).load()
    assert rec.title_en == "Equivalency"
    assert "Official requirement text." in rec.description_en and "menu" not in rec.description_en
    assert rec.source_domain == "moe.gov.sa" and rec.source_sha256.startswith("en:")
    assert rec.fees_en is None


def test_open_data_loader_maps_only_declared_columns(tmp_path):
    (tmp_path / "ds.csv").write_text("ID,NameEn,DescEn,Link,Other\n5,Svc,Desc,https://x.gov.sa/5,ignored\n",
                                     encoding="utf-8")
    (tmp_path / "ds.map.json").write_text(json.dumps({
        "dataset_url": "https://open.data.gov.sa/en/datasets/view/abc", "id_column": "ID",
        "downloaded_at": "2026-10-01T00:00:00Z",
        "columns": {"title_en": "NameEn", "description_en": "DescEn", "official_url_en": "Link"}}),
        encoding="utf-8")
    [rec] = OpenDataFileLoader(tmp_path).load()
    assert rec.title_en == "Svc" and rec.official_url_en == "https://x.gov.sa/5"
    assert rec.fees_en is None and rec.source_type == "official_open_data"
