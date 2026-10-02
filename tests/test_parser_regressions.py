"""Regression tests for parser bugs found while integrating the real V2 captures (final pass).

Each fixture is a minimal reconstruction of the markup pattern that broke, not a copy of a page.
"""
import json

from src.ingestion.generic_service_page import GenericServiceLoader, parse_page


def test_sharepoint_form_wrapper_is_not_deleted():
    # MoJ / CHI: SharePoint wraps the whole page in <form>; deleting forms deleted everything.
    html = """<body><form id="aspnetForm"><h1>Notarize divorce</h1>
      <p class="lead">An electronic service that enables users to obtain certification of a divorce.</p>
      <h3>Steps</h3><ol><li>Log in to Najiz.</li><li>Submit the application.</li></ol></form></body>"""
    p = parse_page(html)
    assert p["title"] == "Notarize divorce"
    assert "certification of a divorce" in p["description"]
    assert "Log in to Najiz." in p["steps"]


def test_html_escaped_rich_text_is_reparsed():
    # MoJ prints rich text escaped: the page literally shows "<ol><li>..."
    html = """<main><h1>Service</h1><p>A service that lets the user do a specific official thing online.</p>
      <h3>Steps</h3><div class="render-me">&lt;ol&gt;&lt;li style='direction: ltr'&gt;First step&lt;/li&gt;&lt;li&gt;Second step&lt;/li&gt;&lt;/ol&gt;</div></main>"""
    steps = parse_page(html)["steps"]
    assert "<" not in steps and "First step" in steps and "Second step" in steps


def test_bootstrap_tabs_use_data_bs_target_not_aria_label():
    # MOFA: aria-controls holds the label text; the panel id is in data-bs-target.
    html = """<main><h1>Company ID Verification</h1><p>This service allows a company representative to verify data.</p>
      <ul role="tablist"><li><a role="tab" aria-controls="الخطوات" data-bs-target="#tab1Content" href="#">الخطوات</a></li>
      <li><a role="tab" aria-controls="المستندات المطلوبة" data-bs-target="#tab3Content" href="#">المستندات المطلوبة</a></li></ul>
      <div id="tab1Content" role="tabpanel"><ul><li>Register in the service</li><li>Book an appointment</li></ul></div>
      <div id="tab3Content" role="tabpanel"><ul><li>Passport or national ID</li></ul></div></main>"""
    p = parse_page(html)
    assert "Book an appointment" in p["steps"]
    assert "Passport or national ID" in p["required_documents"]


def test_related_service_cards_never_become_this_services_description():
    # MoMAH: "Related services" carousel cards carry other services' descriptions.
    html = """<main><div class="region-breadcrumb"><h1>Transferring a Request</h1></div>
      <p>An electronic service on Balady that allows the account manager to transfer a request.</p>
      <p class="fw-bold">Service Fees</p><p>Free</p>
      <p class="fw-bold">Related services</p>
      <div class="carousal-all"><div class="card"><p class="service-description">The Roads Work Permit service is
      one of the Infrastructure Works Coordination Services.</p></div></div></main>"""
    p = parse_page(html)
    assert p["title"] == "Transferring a Request"            # h1 inside a breadcrumb wrapper is kept
    assert p["description"].startswith("An electronic service on Balady")
    assert "Roads Work Permit" not in json.dumps(p, ensure_ascii=False)
    assert p["fees"] == "Free"                                 # <p class=fw-bold>label</p><p>value</p>


def test_placeholders_and_unitless_numbers_are_left_absent():
    html = """<main><h1>X</h1><p>A long enough description of an official electronic service here.</p>
      <h3>Required Documents</h3><p>No content available</p>
      <h3>المتطلبات</h3><p>لايوجد محتوى متاح</p>
      <h3>Execution Duration</h3><p>2</p><p>4</p></main>"""
    p = parse_page(html)
    assert not p.get("required_documents")
    assert not p.get("requirements")
    assert not p.get("processing_time")      # "2 4" without units would invite a guess


def test_page_chrome_before_title_is_not_a_description():
    html = """<main><div>Display Settings Normal Mode Dark Mode Gray Mode High brightness and more</div>
      <h1>Service</h1><p>The real description of this official electronic service is here.</p></main>"""
    assert parse_page(html)["description"].startswith("The real description")


def _write(path, source, records):
    path.write_text(json.dumps({"source": source, "records": records}, ensure_ascii=False), encoding="utf-8")


def _page(sid, lang, status, title, body="A sufficiently long official description of this electronic service."):
    html = f"<main><h1>{title}</h1><p>{body}</p><h3>Steps</h3><ol><li>Step one of the service.</li></ol></main>"
    return {"id": sid, "lang": lang, "url": f"https://www.sfda.gov.sa/{lang}/eservices/{sid}", "http_status": status,
            "fetched_at": "2026-10-01T06:00:00Z", "page_title": title, "main_html": html if status == 200 else "<p>404</p>"}


def test_supplement_fills_only_failed_pages(tmp_path):
    main = tmp_path / "dalil_v2_sfda.json"
    sup = tmp_path / "dalil_v2_sfda_arfix.json"
    _write(main, "sfda", [_page("a", "en", 200, "Calories"), _page("a", "ar", 404, "x"),
                          _page("b", "en", 200, "Drugs List"), _page("b", "ar", 200, "قائمة الأدوية", "وصف رسمي كاف لهذه الخدمة الإلكترونية الرسمية")])
    _write(sup, "sfda", [_page("a", "ar", 200, "نظام سعرات", "وصف رسمي كاف لهذه الخدمة الإلكترونية الرسمية"),
                         _page("b", "ar", 200, "SHOULD NOT REPLACE", "نص")])
    loader = GenericServiceLoader(main)
    recs = {r.source_record_id: r for r in loader.load()}
    assert recs["a"].title_ar == "نظام سعرات"           # filled from the supplement
    assert recs["b"].title_ar == "قائمة الأدوية"        # good original never overwritten
    assert loader.page_stats["pages_filled_by_supplement"] == 1
    assert recs["a"].extra.get("supplement_pages") == ["ar:dalil_v2_sfda_arfix.json"]


def test_arabic_slot_must_contain_arabic(tmp_path):
    main = tmp_path / "dalil_v2_sfda.json"
    _write(main, "sfda", [_page("a", "en", 200, "Calories"), _page("a", "ar", 200, "Calories (English text)")])
    loader = GenericServiceLoader(main, supplements=[])
    rec = loader.load()[0]
    assert rec.title_en == "Calories" and rec.title_ar is None
    assert loader.page_stats["pages_wrong_language"] == 1
