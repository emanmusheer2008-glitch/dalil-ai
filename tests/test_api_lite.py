"""API + engine tests for the V3 lite runtime (DALIL_RUNTIME=lite; real engine, no mocking).

* the production process must not import torch / sentence-transformers / transformers;
* answers, refusals and Arabic Unicode behave as in the V2 contract;
* stateless follow-ups via ``context_service_id``.
"""
import json
import subprocess
import sys
import time

import pytest

from src import config

pytestmark = pytest.mark.integration

CFG_V3 = config.PROCESSED_DIR / "retrieval_config_v3.json"


def _need_artifacts():
    for p in (CFG_V3, config.EMBEDDINGS_NPY, config.PROCESSED_DIR / "static_vectors.npy"):
        if not p.exists():
            pytest.skip(f"lite artifacts missing: {p.name}")


@pytest.fixture(scope="module")
def client():
    _need_artifacts()
    from fastapi.testclient import TestClient

    mp = pytest.MonkeyPatch()
    mp.setenv("DALIL_RUNTIME", "lite")
    from api.main import STATE, app

    with TestClient(app) as c:
        deadline = time.time() + 120
        while STATE.status != "ready" and time.time() < deadline:
            if STATE.status == "error":
                pytest.fail(f"lite engine failed: {STATE.error}")
            time.sleep(0.2)
        assert STATE.status == "ready"
        assert STATE.runtime == "lite"
        yield c
    mp.undo()


def _ask(client, q, **kw):
    r = client.post("/ask", json={"question": q, **kw})
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------ no torch ---
def test_lite_process_does_not_import_torch():
    """Fresh interpreter: load the API in lite mode, answer a question, check heavy ML libs are absent."""
    _need_artifacts()
    code = (
        "import os, sys, json\n"
        "os.environ['DALIL_RUNTIME']='lite'\n"
        "from fastapi.testclient import TestClient\n"
        "from api.main import app, STATE\n"
        "with TestClient(app) as c:\n"
        "    STATE.wait_ready(120)\n"
        "    r = c.post('/ask', json={'question': 'How can I reserve a trade name?'})\n"
        "    assert r.status_code == 200\n"
        "print(json.dumps({m: (m in sys.modules) for m in ('torch','sentence_transformers','transformers')}))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=config.ROOT,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    loaded = json.loads(out.stdout.strip().splitlines()[-1])
    assert loaded == {"torch": False, "sentence_transformers": False, "transformers": False}


# ------------------------------------------------------------ metadata ---
def test_health_info_identify_lite(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["runtime"] == "lite" and h["version"].startswith("3.")
    info = client.get("/info").json()
    assert info["runtime"] == "lite"
    assert info["retrieval"]["transformer_loaded_at_runtime"] is False
    assert info["retrieval"]["thresholds"]["answer"] == json.loads(CFG_V3.read_text())["t_answer"]
    assert info["generative_model"] is None


def test_stats_report_v3_and_v2_reference(client):
    ev = client.get("/stats").json()["evaluation"]
    assert "results_v3.json" in ev["source"]
    assert ev["test_retrieval"]["n"] > 0
    assert ev["v2_full_runtime_reference_test"]["retrieval"]["n"] == ev["test_retrieval"]["n"]


def test_no_paths_or_secrets_leak(client):
    blob = json.dumps([client.get(p).json() for p in ("/", "/health", "/info", "/stats")])
    for bad in ("/home/", "C:\\\\", "Users\\\\", "/root", "site-packages"):
        assert bad not in blob


# ----------------------------------------------------------------- ask ---
def test_english_answer_has_citations(client):
    d = _ask(client, "How can I apply for a family visit visa?")
    assert d["runtime"] == "lite"
    assert d["response_type"] in ("answer", "possible_match")
    assert d["sources"] and all(s["url"] for s in d["sources"])
    cites = {s["citation"] for s in d["sources"]}
    assert all(p["citation"] in cites for sec in d["sections"] for p in sec["points"])


def test_arabic_answer_unicode(client):
    r = client.post("/ask", json={"question": "ابي احجز اسم تجاري"})
    assert r.status_code == 200
    assert "charset=utf-8" in r.headers["content-type"]
    assert "اسم تجاري" in r.content.decode("utf-8")          # raw UTF-8, not \\u escapes only
    d = r.json()
    assert d["detected_language"] == "ar" and d["answer_language"] == "ar"
    assert d["response_type"] in ("answer", "possible_match")
    assert d["sources"][0]["service_id"] == "mc-1"


@pytest.mark.parametrize("q", ["What is the best pizza in Riyadh?", "Who won the football match yesterday?",
                               "اكتب لي قصيدة عن البحر"])
def test_out_of_domain_not_answered(client, q):
    d = _ask(client, q)
    assert d["response_type"] in ("unsupported", "related_services")
    assert d["sections"] == [] and d["sources"] == []


def test_followup_uses_context(client):
    first = _ask(client, "ابي احجز اسم تجاري")
    sid = first["sources"][0]["service_id"]
    d = _ask(client, "كم الرسوم؟", context_service_id=sid)
    assert d["context_service_id"] == sid
    assert d["sources"] and d["sources"][0]["service_id"] == sid
    assert d["intent"] == "fees"


def test_followup_english_documents(client):
    first = _ask(client, "How can I apply for a family visit visa?")
    sid = first["sources"][0]["service_id"]
    d = _ask(client, "What documents do I need?", context_service_id=sid)
    assert d["context_service_id"] == sid and d["sources"][0]["service_id"] == sid


def test_new_topic_ignores_context(client):
    """A long, self-contained question is not treated as a follow-up."""
    d = _ask(client, "How do I apply for a family visit visa for my parents to come to Saudi Arabia?",
             context_service_id="mc-1")
    assert d["context_service_id"] is None


def test_context_without_sources_is_ignored(client):
    d = _ask(client, "how much does it cost?", context_service_id="xx-999999")
    assert d["context_service_id"] is None


def test_bad_context_id_rejected(client):
    r = client.post("/ask", json={"question": "fees?", "context_service_id": "../../etc/passwd"})
    assert r.status_code == 422


def test_followup_alone_without_context_does_not_invent(client):
    d = _ask(client, "how much are the fees?")
    # no service named: must not be a confident answer
    assert d["response_type"] != "answer"


# ------------------------------------------------------ engine helpers ---
@pytest.mark.parametrize("q,expected", [
    ("كم الرسوم؟", True), ("What documents do I need?", True), ("how long does it take?", True),
    ("and is it available online?", True),
    ("How do I renew my passport at the passports department in Riyadh city today?", False),
    ("pizza", False),
])
def test_looks_like_followup(q, expected):
    from src.lite.engine import looks_like_followup
    assert looks_like_followup(q) is expected
