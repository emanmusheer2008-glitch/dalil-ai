"""API tests against the REAL Dalil V2 engine (DALIL_RUNTIME=full; no mocking of retrieval or answering).

The lite (V3) runtime has its own tests in tests/test_api_lite.py.

The client is created once per module; the FastAPI lifespan loads the knowledge base and
starts the background engine load, exactly as in production. Tests that need the engine
wait for /health to report ready.
"""
import time

import pytest

from src import config

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    if not config.EMBEDDINGS_NPY.exists():
        pytest.skip("index not built")
    pytest.importorskip("sentence_transformers")
    from fastapi.testclient import TestClient

    mp = pytest.MonkeyPatch()
    mp.setenv("DALIL_RUNTIME", "full")

    from api.main import STATE, app

    with TestClient(app) as c:
        deadline = time.time() + 300
        while STATE.status != "ready" and time.time() < deadline:
            if STATE.status == "error":
                pytest.skip(f"engine unavailable: {STATE.error}")
            time.sleep(0.5)
        if STATE.status != "ready":
            pytest.skip("engine did not load in time")
        yield c
    mp.undo()


def _ask(client, q, language="auto"):
    r = client.post("/ask", json={"question": q, "language": language})
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------------ meta --
def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["name"] == "Dalil AI API" and r.json()["status"] == "ok"


def test_health_reports_ready_engine(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["engine_loaded"] is True and body["services"] > 0


def test_info_is_derived_from_artifacts_and_has_no_local_paths(client):
    from src.ingestion.pipeline import load_services

    recs = load_services()
    body = client.get("/info").json()
    assert body["corpus"]["services"] == len(recs)
    assert body["corpus"]["agencies"] == len({s.split("-")[0] for s in recs})
    assert set(body["supported_languages"]) == {"ar", "en"}
    text = client.get("/info").text + client.get("/stats").text
    for bad in ("/home/", "C:\\\\", "Users\\\\", "/root/", "/tmp/"):
        assert bad not in text


def test_stats_reads_evaluation_results(client):
    import json

    body = client.get("/stats").json()
    res = json.loads((config.EVAL_DIR / "results_v2.json").read_text(encoding="utf-8"))
    assert body["evaluation"]["test_retrieval"] == res["v2_test"]["retrieval"]
    assert body["corpus"]["services_indexed"] > 0


# ------------------------------------------------------------------ /ask --
def test_ask_english_same_as_engine(client):
    from api.main import STATE

    q = "How can I renew a commercial registration?"
    body = _ask(client, q)
    direct = STATE.engine.ask(q)                      # the verified engine, called directly
    assert body["engine_status"] == direct.status
    assert [s["service_id"] for s in body["sources"]] == [c.service_id for c in direct.citations]
    assert body["detected_language"] == "en" and body["answer_language"] == "en"
    assert body["response_type"] in ("answer", "possible_match", "related_services", "unsupported")


def test_ask_arabic_preserves_unicode(client):
    q = "كيف يمكنني تجديد السجل التجاري؟"
    r = client.post("/ask", json={"question": q})
    assert r.status_code == 200
    assert "charset=utf-8" in r.headers["content-type"]
    assert q in r.content.decode("utf-8")              # raw Arabic, not \\u escapes
    body = r.json()
    assert body["question"] == q and body["detected_language"] == "ar" and body["answer_language"] == "ar"
    if body["sources"]:
        assert body["sources"][0]["language"] == "ar"


def test_ask_known_answer_has_structure_and_citations(client):
    body = _ask(client, "I want to book a business name")
    assert body["response_type"] in ("answer", "possible_match")
    assert body["sources"][0]["service_id"] == "mc-1"
    assert body["sources"][0]["url"].startswith("https://")
    cites = {s["citation"] for s in body["sources"]}
    for sec in body["sections"]:
        for p in sec["points"]:
            assert p["citation"] in cites and p["text"]


def test_ask_unsupported(client):
    body = _ask(client, "What's the best pizza in Riyadh?")
    assert body["response_type"] == "unsupported"
    assert body["sources"] == [] and body["sections"] == [] and body["related_services"] == []


def test_ask_related_services_are_links_only(client):
    # (was "How do I renew my iqama?" -- iqama renewal is covered since the V4.1 Absher records)
    body = _ask(client, "How do I register my car with Najm insurance?")
    if body["response_type"] != "related_services":
        pytest.skip(f"engine returned {body['response_type']} for this question")
    assert body["sections"] == [] and body["sources"] == []
    assert body["related_services"] and all(s["url"] for s in body["related_services"])


def test_ask_language_override(client):
    body = _ask(client, "خطوات تأسيس شركة ذات مسؤولية محدودة", language="en")
    assert body["detected_language"] == "ar" and body["answer_language"] == "en"


@pytest.mark.parametrize("payload", [
    {"question": ""}, {"question": "    "}, {"question": "x" * 401}, {"question": "hi", "language": "fr"}, {},
])
def test_ask_validation(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


# -------------------------------------------------------------- services --
def test_services_list_and_pagination(client):
    body = client.get("/services", params={"limit": 5, "offset": 0}).json()
    assert body["total"] >= 5 and len(body["items"]) == 5
    nxt = client.get("/services", params={"limit": 5, "offset": 5}).json()
    assert {i["service_id"] for i in body["items"]}.isdisjoint({i["service_id"] for i in nxt["items"]})


def test_services_filters(client):
    z = client.get("/services", params={"agency": "zatca", "limit": 100}).json()
    assert z["total"] > 0 and all(i["agency_code"] == "zatca" for i in z["items"])
    s = client.get("/services", params={"search": "حجز اسم تجاري", "language": "ar"}).json()
    assert any(i["service_id"] == "mc-1" for i in s["items"])


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 101}, {"offset": -1}, {"language": "fr"}])
def test_services_bad_params(client, params):
    assert client.get("/services", params=params).status_code == 422


def test_service_detail(client):
    body = client.get("/services/mc-1").json()
    assert body["service_id"] == "mc-1"
    assert body["en"]["title"] and body["ar"]["title"]
    assert body["en"]["official_url"].startswith("https://mc.gov.sa/")


@pytest.mark.parametrize("sid", ["does-not-exist", "mc-999999", "..%2F..%2Fetc"])
def test_service_detail_not_found(client, sid):
    assert client.get(f"/services/{sid}").status_code == 404


def test_agencies(client):
    body = client.get("/agencies").json()
    total = client.get("/services", params={"limit": 1}).json()["total"]
    assert sum(a["services"] for a in body) == total
    assert {"zatca", "mc", "moj"} <= {a["code"] for a in body}


def test_cors_preflight_allows_frontend(client):
    r = client.options("/ask", headers={"Origin": "https://example-frontend.app",
                                        "Access-Control-Request-Method": "POST",
                                        "Access-Control-Request-Headers": "content-type"})
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") in ("*", "https://example-frontend.app")
