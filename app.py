"""Dalil AI — Streamlit application.

    streamlit run app.py

The app only *reads* prebuilt artefacts (knowledge base, cached embeddings,
evaluation results). Build them first with:

    python -m src.indexing.build_index
    python -m src.evaluation.evaluate
"""
from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from src import config
from src.answering.composer import compose, query_language
from src.indexing.chunking import SECTION_LABELS
from src.ingestion.pipeline import load_services
from src.retrieval.retriever import Retriever, load_retrieval_config
from src.ui import CSS, EXAMPLES, T, dir_of, esc, evidence_block, kpi, service_card
from src.utils.arabic import normalize_for_matching

MAX_QUERY_CHARS = 400

st.set_page_config(page_title="Dalil AI — دليل", page_icon="🧭", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ data ----
@st.cache_resource(show_spinner="Loading Dalil's multilingual model…")
def get_retriever() -> Retriever:
    r = Retriever.from_disk()
    r.chunk_scores("warm-up")          # load the model once, at start-up
    return r


@st.cache_resource
def get_records():
    return load_services()


@st.cache_data
def read_json(path_str: str):
    from pathlib import Path

    p = Path(path_str)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ui_lang() -> str:
    return st.session_state.get("ui_lang", "en")


def header():
    c1, c2 = st.columns([5, 1.3])
    with c2:
        choice = st.segmented_control("Language", ["English", "العربية"],
                                      default="العربية" if ui_lang() == "ar" else "English",
                                      label_visibility="collapsed", key="lang_toggle")
        st.session_state["ui_lang"] = "ar" if choice == "العربية" else "en"
    return T[ui_lang()]


def footer(t):
    st.markdown(f'<div class="dl-foot" dir="{dir_of(ui_lang())}">⚠️ {esc(t["disclaimer"])}</div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------------ pages ---
def page_ask():
    t = header()
    lang = ui_lang()
    records = get_records()
    st.markdown(
        f'<div class="dl-hero" dir="{dir_of(lang)}"><h1>Dalil AI · دليل</h1>'
        f'<div class="dl-tag">{esc(t["tagline"])}</div><p>{esc(t["hero_sub"])}</p></div>',
        unsafe_allow_html=True,
    )

    ex = st.pills(t["examples"], EXAMPLES[lang], key=f"ex_{lang}")
    if ex and st.session_state.get("_last_ex") != ex:
        st.session_state["query"] = ex
        st.session_state["_last_ex"] = ex

    with st.form("ask", border=False):
        q = st.text_input(t["ask_label"], key="query", placeholder=t["ask_placeholder"],
                          max_chars=MAX_QUERY_CHARS)
        submitted = st.form_submit_button(t["ask_button"], type="primary")

    q = (q or "").strip()
    if not q:
        st.caption(t["scope_note"].format(n=len(records)))
        footer(t)
        return

    cfg = load_retrieval_config()
    retriever = get_retriever()
    result = retriever.search(q, method=cfg["method"], alpha=cfg["alpha"], top_k=5)
    threshold = cfg["threshold"] if cfg.get("threshold") is not None else 0.5
    answer = compose(result, records, threshold=threshold, ui_lang=query_language(q))
    at = T[answer.ui_lang]

    if answer.status == "insufficient":
        st.markdown(
            f'<div class="dl-refuse" dir="{dir_of(answer.ui_lang)}"><h3>{esc(at["insufficient_title"])}</h3>'
            f'<div>{esc(answer.message)}</div><div class="dl-meta" style="margin-top:8px">'
            f'{esc(at["try_instead"])}</div></div>', unsafe_allow_html=True)
    elif answer.status == "answered":
        p = answer.primary
        st.markdown(service_card(p, at, heading=at["answer_heading"]), unsafe_allow_html=True)
        with st.expander(at["why"], expanded=True):
            st.markdown(evidence_block(p, at, answer.ui_lang), unsafe_allow_html=True)
        if answer.related:
            st.markdown(f"**{esc(at['related'])}**")
            for r in answer.related:
                with st.expander(f"{r.title} · {r.score:.2f}"):
                    st.markdown(service_card(r, at), unsafe_allow_html=True)

    top = f"{answer.top_score:.3f}" if answer.top_score is not None else "—"
    st.caption(f"{at['latency'].format(ms=answer.latency_ms or 0)} · top score {top} · "
               f"{at['threshold_note']} {threshold:.3f} ({cfg['method']}"
               f"{', α=' + str(cfg['alpha']) if cfg['method'] == 'hybrid' else ''})")
    footer(t)


def page_explore():
    t = header()
    lang = ui_lang()
    records = list(get_records().values())
    st.title(t["nav_explore"])
    c1, c2 = st.columns([2, 1])
    kw = c1.text_input("Keyword / كلمة", placeholder="e.g. franchise, سجل تجاري", max_chars=100)
    def tags(r):
        pt = r.extra.get("page_tags", {}) if isinstance(r.extra, dict) else {}
        return pt.get(lang) or pt.get("en") or []

    cats = sorted({tg for r in records for tg in tags(r)})
    chosen = c2.multiselect("Label / التصنيف", cats)
    nkw = normalize_for_matching(kw) if kw else ""

    def matches(r):
        if chosen and not set(chosen) & set(tags(r)):
            return False
        if nkw:
            hay = normalize_for_matching(" ".join(filter(None, [r.title_en, r.title_ar, r.description_en,
                                                                r.description_ar])))
            return all(tok in hay for tok in nkw.split())
        return True

    shown = [r for r in records if matches(r)]
    st.caption(f"{len(shown)} / {len(records)}")
    for r in sorted(shown, key=lambda r: (r.get("title", lang) or r.title_en or "")):
        show = lang if r.get("title", lang) else r.languages()[0]
        with st.expander(r.get("title", show)):
            parts = [f'<div dir="{dir_of(show)}">',
                     f'<div class="dl-meta">{esc(r.get("agency", show))} · {esc(r.get("category", show) or "")}</div>']
            for name in ("description", "requirements", "required_documents", "steps", "fees",
                         "processing_time", "target_audience"):
                val = r.get(name, show)
                if val:
                    parts.append(f'<div class="dl-field"><div class="lbl">{esc(SECTION_LABELS[show].get(name, name))}'
                                 f'</div><div class="val">{esc(val)}</div></div>')
            url = r.get("official_url", show)
            parts.append(f'<a class="dl-btn" href="{esc(url)}" target="_blank" rel="noopener">'
                         f'{esc(T[show]["official_page"])} ↗</a></div>')
            st.markdown("".join(parts), unsafe_allow_html=True)
    footer(t)


def page_kb():
    t = header()
    rep = read_json(str(config.BUILD_REPORT_JSON)) or {}
    meta = read_json(str(config.INDEX_META_JSON)) or {}
    st.title(t["nav_kb"])
    st.write("Dalil answers **only** from the services below. Each was captured from an official Saudi "
             "government page, stored with its URL, capture date and a SHA-256 hash of the captured HTML, "
             "and validated before indexing. Records that could not be verified are quarantined, not indexed.")
    cols = st.columns(4)
    for col, (n, l) in zip(cols, [
        (rep.get("services_indexed", 0), "services indexed"),
        (meta.get("n_chunks", 0), "evidence chunks"),
        (f"{rep.get('with_arabic', 0)} / {rep.get('with_english', 0)}", "with official Arabic / English text"),
        (rep.get("services_quarantined", 0), "records quarantined"),
    ]):
        col.markdown(kpi(n, l), unsafe_allow_html=True)

    st.subheader("Sources")
    st.write(f"**Agencies:** {', '.join(rep.get('agencies', []))}  \n"
             f"**Source domains:** {', '.join(rep.get('source_domains', []))}  \n"
             f"**Captured:** {str(rep.get('date_collected_range', ['', ''])[0])[:10]} · "
             f"**Knowledge base built:** {rep.get('built_at', '')[:10]} · "
             f"**Embedding model:** `{meta.get('model_name', '')}` ({meta.get('dim', '')}-d)")

    if rep.get("field_coverage"):
        cov = pd.DataFrame([{"field": k[:-3], "language": k[-2:], "services": v}
                            for k, v in rep["field_coverage"].items()])
        cov = cov[~cov.field.isin(["source_last_modified", "eligibility"])]
        st.subheader("Field coverage")
        st.caption("How many services state each field on their official page. Missing fields are left empty — never filled in.")
        chart = alt.Chart(cov).mark_bar().encode(
            x=alt.X("services:Q", title="services"), y=alt.Y("field:N", sort="-x", title=None),
            color=alt.Color("language:N", scale=alt.Scale(range=["#0F6B5A", "#C8A45A"])),
            yOffset="language:N").properties(height=320)
        st.altair_chart(chart, use_container_width=True)

    if rep.get("page_labels_en"):
        st.subheader("Official page labels")
        st.caption("Short labels shown on each official service page (number of services).")
        st.write(" ".join(f'<span class="dl-pill">{esc(c)} · {n}</span>'
                          for c, n in sorted(rep["page_labels_en"].items(), key=lambda x: -x[1])),
                 unsafe_allow_html=True)

    st.subheader("Quarantined records")
    qpath = config.QUARANTINE_JSONL
    if qpath.exists():
        rows = [json.loads(line) for line in qpath.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows:
            st.dataframe(pd.DataFrame([{"id": r["service_id"], "title": r.get("title_en") or r.get("title_ar"),
                                        "url": r.get("official_url_en") or r.get("official_url_ar"),
                                        "reason": "; ".join(r["_errors"])} for r in rows]),
                         hide_index=True, use_container_width=True)
    footer(t)


def page_eval():
    t = header()
    res = read_json(str(config.EVAL_RESULTS_JSON))
    st.title(t["nav_eval"])
    if not res:
        st.info("Run `python -m src.evaluation.evaluate` to generate results.")
        return
    b = res["benchmark"]
    sel = res["selected"]
    rt = res["retrieval_test"]["overall"]
    rf = res["refusal"]["test"]
    st.write(f"Benchmark: **{b['n_queries']} questions** ({b['n_supported']} answerable, "
             f"{b['n_unsupported']} unanswerable) in English, Arabic and mixed language. Method and refusal "
             f"threshold were chosen on the calibration half; numbers below are on the **held-out test half** "
             f"unless stated. All values are computed by `src/evaluation/evaluate.py`.")
    cols = st.columns(5)
    for col, (n, l) in zip(cols, [
        (f"{rt['top1']:.0%}", f"Top-1 retrieval accuracy (n={rt['n']})"),
        (f"{rt['top3']:.0%}", "Top-3 retrieval accuracy"),
        (f"{rt['mrr']:.2f}", "MRR"),
        (f"{rf['unsupported_refused']:.0%}", "unanswerable questions declined"),
        (f"{rf['supported_answered_correct_top1']:.0%}", "answerable questions answered correctly "
                                                           "(the price of cautious refusal)"),
    ]):
        col.markdown(kpi(n, l), unsafe_allow_html=True)
    st.caption(f"Selected configuration: **{sel['config']}** · refusal threshold "
               f"{res['refusal']['threshold']:.3f} · mean latency {res['latency_ms']['mean']:.0f} ms "
               f"(p95 {res['latency_ms']['p95']:.0f} ms) on {res['latency_ms']['hardware']}")

    st.subheader("Retrieval methods compared (test split)")
    comp = pd.DataFrame([{"method": k, "metric": m, "value": v["test"][m]}
                         for k, v in res["method_comparison"].items() for m in ("top1", "top3", "mrr")])
    st.altair_chart(alt.Chart(comp).mark_bar().encode(
        x=alt.X("method:N", sort=None, title=None), y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("metric:N", scale=alt.Scale(range=["#0F6B5A", "#7FB7A6", "#C8A45A"])),
        xOffset="metric:N", tooltip=["method", "metric", alt.Tooltip("value:Q", format=".3f")]
    ).properties(height=300), use_container_width=True)

    st.subheader("By language (test split)")
    lang_rows = [{"language": k, **{m: v[m] for m in ("n", "top1", "top3", "mrr")}}
                 for k, v in res["retrieval_test"]["by_language"].items()]
    st.dataframe(pd.DataFrame(lang_rows), hide_index=True, use_container_width=True)

    st.subheader("Cross-lingual retrieval (all answerable questions)")
    st.caption("Arabic questions searched against the English page text only, and vice-versa. Lexical matching "
               "cannot cross languages; multilingual embeddings can.")
    cl = pd.DataFrame([{"setting": s, "method": m, "top1": v["top1"], "top3": v["top3"], "mrr": v["mrr"], "n": v["n"]}
                       for s, d in res["cross_lingual"].items() for m, v in d.items()])
    st.dataframe(cl, hide_index=True, use_container_width=True)

    st.subheader("Declining unanswerable questions")
    c1, c2 = st.columns(2)
    c1.markdown(kpi(f"{rf['refusal_precision']:.0%}", "refusal precision (test)"), unsafe_allow_html=True)
    c2.markdown(kpi(f"{rf['supported_answered_correct_top1']:.0%}", "answerable questions answered correctly (test)"),
                unsafe_allow_html=True)
    curve = pd.DataFrame(res["refusal"]["curve_all_queries"]).melt("threshold", var_name="rate", value_name="value")
    rule = alt.Chart(pd.DataFrame({"t": [res["refusal"]["threshold"]]})).mark_rule(strokeDash=[4, 4]).encode(x="t:Q")
    line = alt.Chart(curve).mark_line().encode(
        x=alt.X("threshold:Q"), y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("rate:N", scale=alt.Scale(range=["#0F6B5A", "#C8A45A"])))
    st.altair_chart((line + rule).properties(height=280), use_container_width=True)
    st.caption("Trade-off across thresholds (all queries). Dashed line = threshold chosen on the calibration split.")

    st.subheader("Every question")
    pq = config.EVAL_PER_QUERY_CSV
    if pq.exists():
        df = pd.read_csv(pq)
        st.dataframe(df[["query_id", "query", "language", "query_type", "split", "expected_service_id",
                         "top1_service_id", "rank", "top_score", "refused"]], hide_index=True,
                     use_container_width=True, height=380)
    st.info("Limitations: the benchmark was written by the project author (AI-assisted) and is small; the corpus "
            "covers one ministry. Results describe this corpus and question set, not Saudi government services "
            "in general.")
    footer(t)


def page_about():
    t = header()
    st.title(t["nav_about"])
    p = config.ROOT / "docs" / "methodology.md"
    st.markdown(p.read_text(encoding="utf-8") if p.exists() else "See README.md")
    st.graphviz_chart("""
digraph { rankdir=TB; node [shape=box style="rounded,filled" fillcolor="#F1EEE6" color="#0F6B5A" fontname="Helvetica"];
 A [label="Official Saudi pages (mc.gov.sa, EN + AR)"]; B [label="Capture (browser, low rate)\\n+ SHA-256"];
 C [label="Parse + normalise + validate\\n(quarantine unverified)"]; D [label="Knowledge base\\nservices.jsonl"];
 E [label="Section chunks\\n(overview, conditions, documents, steps, fees)"];
 F [label="Multilingual embeddings\\n(MiniLM-L12, cached .npy)"]; G [label="Arabic / English question"];
 H [label="Hybrid retrieval\\n(dense + char n-gram TF-IDF)"]; I [label="Calibrated threshold\\nanswer or decline"];
 J [label="Official text + evidence\\n+ source link"];
 A->B->C->D->E->F->H; G->H->I->J; }""")
    footer(t)


pages = [
    st.Page(page_ask, title="Ask Dalil", icon=":material/chat:", default=True),
    st.Page(page_explore, title="Explore services", icon=":material/search:", url_path="explore"),
    st.Page(page_kb, title="Knowledge base", icon=":material/database:", url_path="knowledge-base"),
    st.Page(page_eval, title="Evaluation", icon=":material/monitoring:", url_path="evaluation"),
    st.Page(page_about, title="About", icon=":material/info:", url_path="about"),
]
st.navigation(pages, position="top").run()
