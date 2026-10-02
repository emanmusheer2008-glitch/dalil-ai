"""Dalil AI — Streamlit application (V2).

    streamlit run app.py

The app only *reads* prebuilt artefacts (knowledge base, cached embeddings,
calibrated config, evaluation results). Build them first with:

    python -m src.indexing.build_index
    python -m src.evaluation.evaluate_v2
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src import config
from src.answering.synthesizer import UI as SUI
from src.indexing.chunking import SECTION_LABELS
from src.engine import Dalil
from src.retrieval.retriever import Retriever
from src.ui import CSS, CSS_V2, EXAMPLES, T, dir_of, esc, evidence_v2_html, kpi, synth_answer_html
from src.utils.arabic import normalize_for_matching

MAX_QUERY_CHARS = 400

st.set_page_config(page_title="Dalil AI — دليل", page_icon="🧭", layout="wide", initial_sidebar_state="collapsed")
st.markdown(CSS + CSS_V2, unsafe_allow_html=True)


# ------------------------------------------------------------------ data ----
@st.cache_resource(show_spinner="Loading Dalil's multilingual model and index (first start only)…")
def get_engine() -> Dalil:
    d = Dalil()                         # reads the index, knowledge base and calibrated config once
    d.warm_up()                         # loads the model once per server process
    return d


def get_retriever() -> Retriever:
    return get_engine().retriever


def get_records():
    return get_engine().records


@st.cache_data
def read_json(path_str: str):
    p = Path(path_str)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data
def coverage():
    recs = get_records()
    agencies = Counter(r.agency_en or r.agency_ar for r in recs.values())
    return {"n": len(recs), "agencies": agencies,
            "ar": sum(bool(r.title_ar) for r in recs.values()), "en": sum(bool(r.title_en) for r in recs.values())}


def ui_lang() -> str:
    return st.session_state.get("ui_lang", "en")


def header():
    _, c2 = st.columns([5, 1.3])
    with c2:
        choice = st.segmented_control("Language", ["English", "العربية"],
                                      default="العربية" if ui_lang() == "ar" else "English",
                                      label_visibility="collapsed", key="lang_toggle")
        st.session_state["ui_lang"] = "ar" if choice == "العربية" else "en"
    return T[ui_lang()]


def footer(t):
    st.markdown(f'<div class="dl-foot" dir="{dir_of(ui_lang())}">⚠️ {esc(t["disclaimer"])}</div>',
                unsafe_allow_html=True)


def scope_line(lang: str) -> str:
    c = coverage()
    n_ag = len(c["agencies"])
    if lang == "ar":
        return (f"يجيب دليل حالياً من {c['n']} خدمة رسمية لدى {n_ag} جهات حكومية سعودية "
                f"(نص عربي رسمي لـ{c['ar']} خدمة، وإنجليزي لـ{c['en']}). لا يغطي كل الخدمات الحكومية، "
                f"ويعتذر عن الأسئلة خارج مصادره بدلاً من التخمين.")
    return (f"Dalil currently answers from {c['n']} official services of {n_ag} Saudi government agencies "
            f"(official Arabic text for {c['ar']}, English for {c['en']}). It does not cover every government "
            f"service, and declines questions outside its sources instead of guessing.")


# ------------------------------------------------------------------ pages ---
def page_ask():
    t = header()
    lang = ui_lang()
    st.markdown(
        f'<div class="dl-hero" dir="{dir_of(lang)}"><h1>Dalil AI · دليل</h1>'
        f'<div class="dl-tag">{esc(t["tagline"])}</div><p>{esc(t["hero_sub"])}</p></div>',
        unsafe_allow_html=True,
    )
    get_engine()                         # warm the model before the first question

    ex = st.pills(t["examples"], EXAMPLES[lang], key=f"ex_{lang}")
    if ex and st.session_state.get("_last_ex") != ex:
        st.session_state["query"] = ex
        st.session_state["_last_ex"] = ex

    with st.form("ask", border=False):
        q = st.text_input(t["ask_label"], key="query", placeholder=t["ask_placeholder"], max_chars=MAX_QUERY_CHARS)
        st.form_submit_button(t["ask_button"], type="primary")

    q = (q or "").strip()
    if not q:
        st.caption(scope_line(lang))
        footer(t)
        return

    engine = get_engine()
    with st.spinner(t["loading"]):
        t0 = time.perf_counter()
        ans = engine.ask(q)
        total_ms = (time.perf_counter() - t0) * 1000
    at = T[ans.ui_lang]
    s = SUI[ans.ui_lang]

    if ans.status == "insufficient":
        rel = ""
        if ans.related:
            items = "".join(
                f'<li><a href="{esc(c.url)}" target="_blank" rel="noopener">{esc(c.title)}</a>'
                f' <span class="dl-meta">· {esc(c.agency or "")}</span></li>' for c in ans.related)
            head = "خدمات رسمية ذات صلة" if ans.ui_lang == "ar" else "Related official services"
            rel = f'<div style="margin-top:10px"><b>{head}</b><ul style="margin:6px 0 0">{items}</ul></div>'
        title = at["insufficient_title"] if not ans.related else (
            "إجابة غير مؤكدة" if ans.ui_lang == "ar" else "No exact answer — related services")
        st.markdown(
            f'<div class="dl-refuse" dir="{dir_of(ans.ui_lang)}"><h3>{esc(title)}</h3>'
            f'<div>{esc(ans.message)}</div>{rel}<div class="dl-meta" style="margin-top:8px">'
            f'{esc(at["try_instead"])}</div></div>', unsafe_allow_html=True)
    elif ans.status in ("answered", "tentative"):
        if ans.status == "tentative":
            st.markdown(f'<span class="dl-pill maybe">{esc(at["tentative_title"])}</span>', unsafe_allow_html=True)
        st.markdown(synth_answer_html(ans, at, s), unsafe_allow_html=True)
        with st.expander(at["evidence_title"]):
            st.markdown(evidence_v2_html(ans, SECTION_LABELS[ans.ui_lang]), unsafe_allow_html=True)

    top = f"{ans.top_score:.2f}" if ans.top_score is not None else "—"
    st.caption(f"{at['latency'].format(ms=total_ms)} · {at.get('score', 'match score')} {top}")
    footer(t)


def page_explore():
    t = header()
    lang = ui_lang()
    records = list(get_records().values())
    st.title(t["nav_explore"])
    c1, c2 = st.columns([2, 1])
    kw = c1.text_input("Keyword / كلمة", placeholder="e.g. visa, VAT, سجل تجاري", max_chars=100)
    agencies = sorted({(r.get("agency", lang) or r.agency_en or "") for r in records})
    chosen = c2.multiselect("Agency / الجهة", agencies)
    nkw = normalize_for_matching(kw) if kw else ""

    def matches(r):
        if chosen and (r.get("agency", lang) or r.agency_en) not in chosen:
            return False
        if nkw:
            hay = normalize_for_matching(" ".join(filter(None, [r.title_en, r.title_ar, r.description_en,
                                                                r.description_ar])))
            return all(tok in hay for tok in nkw.split())
        return True

    shown = sorted([r for r in records if matches(r)], key=lambda r: (r.get("title", lang) or r.title_en or ""))
    st.caption(f"{len(shown)} / {len(records)}")
    page_size = 30
    pages = max(1, (len(shown) + page_size - 1) // page_size)
    page = st.number_input("Page", 1, pages, 1, label_visibility="collapsed") if pages > 1 else 1
    for r in shown[(page - 1) * page_size: page * page_size]:
        show = lang if r.get("title", lang) else r.languages()[0]
        with st.expander(f"{r.get('title', show)} — {r.get('agency', show)}"):
            parts = [f'<div dir="{dir_of(show)}">']
            for name in ("description", "requirements", "required_documents", "steps", "fees",
                         "processing_time", "target_audience", "notes"):
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
             "and validated before indexing. Records that could not be verified are quarantined, not indexed. "
             "These figures are computed from the knowledge base each time it is rebuilt.")
    c = coverage()
    cols = st.columns(4)
    for col, (n, l) in zip(cols, [
        (c["n"], "services indexed"),
        (len(c["agencies"]), "government agencies"),
        (f"{c['ar']} / {c['en']}", "with official Arabic / English text"),
        (meta.get("n_chunks", 0), "evidence chunks"),
    ]):
        col.markdown(kpi(n, l), unsafe_allow_html=True)

    st.subheader("Agencies and services")
    ag = pd.DataFrame(sorted(c["agencies"].items(), key=lambda x: -x[1]), columns=["agency", "services"])
    st.altair_chart(alt.Chart(ag).mark_bar(color="#0F6B5A").encode(
        x=alt.X("services:Q"), y=alt.Y("agency:N", sort="-x", title=None),
        tooltip=["agency", "services"]).properties(height=34 * len(ag) + 20), use_container_width=True)
    st.write(f"**Source domains:** {', '.join(rep.get('source_domains', []))}  \n"
             f"**Captured:** {str(rep.get('date_collected_range', ['', ''])[0])[:10]} – "
             f"{str(rep.get('date_collected_range', ['', ''])[1])[:10]} · "
             f"**Knowledge base built:** {rep.get('built_at', '')[:10]} · "
             f"**Embedding model:** `{meta.get('model_name', '')}` ({meta.get('dim', '')}-d)")

    if rep.get("field_coverage"):
        cov = pd.DataFrame([{"field": k[:-3], "language": k[-2:], "services": v}
                            for k, v in rep["field_coverage"].items()])
        cov = cov[~cov.field.isin(["source_last_modified", "eligibility"])]
        st.subheader("Field coverage")
        st.caption("How many services state each field on their official page. Missing fields are left empty — "
                   "never filled in.")
        st.altair_chart(alt.Chart(cov).mark_bar().encode(
            x=alt.X("services:Q"), y=alt.Y("field:N", sort="-x", title=None),
            color=alt.Color("language:N", scale=alt.Scale(range=["#0F6B5A", "#C8A45A"])),
            yOffset="language:N").properties(height=340), use_container_width=True)

    st.subheader("Known coverage gaps")
    st.markdown(
        "- **Not captured** (never bypassed): Ministry of Interior / Absher (passports inside the Kingdom, iqama, "
        "national ID, traffic), Ministry of Health (bot check), GOSI (firewall), Transport General Authority, "
        "my.gov.sa (HTTP 403), MISA (services page 404).\n"
        "- **Council of Health Insurance:** English text only — the original Arabic URLs returned 404; the Arabic "
        "pages were re-captured but could not be exported from the browser in this pass.\n"
        "- **SFDA:** 7 of 33 pages had no content beyond placeholders and are quarantined; 1 Arabic page returned 403.\n"
        "- Official pages change: every answer shows its capture date; data must be re-captured periodically.")

    st.subheader("Quarantined records")
    qpath = config.QUARANTINE_JSONL
    if qpath.exists():
        rows = [json.loads(line) for line in qpath.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows:
            st.caption(f"{len(rows)} records were loaded but excluded from answers.")
            st.dataframe(pd.DataFrame([{"id": r["service_id"], "title": r.get("title_en") or r.get("title_ar"),
                                        "url": r.get("official_url_en") or r.get("official_url_ar"),
                                        "reason": "; ".join(r["_errors"])} for r in rows]),
                         hide_index=True, use_container_width=True, height=300)
    footer(t)


def page_eval():
    t = header()
    st.title(t["nav_eval"])
    res = read_json(str(config.EVAL_DIR / "results_v2.json"))
    if not res:
        st.info("Run `python -m src.evaluation.evaluate_v2` to generate results.")
        return
    v2 = res["v2_test"]
    base = res["v1_setting_on_v2_corpus_test"]
    d = v2["decision"]
    b = res["benchmark"]
    st.write(f"V2 benchmark: **{b['n']} questions** in **{b['families']} families** — each information need is "
             f"asked several ways (formal, conversational, short, misspelled, formal and colloquial Arabic, mixed). "
             f"Settings and thresholds were chosen on the *dev* and *val* splits only; every number below is on the "
             f"**held-out test split**. All values are produced by `src/evaluation/evaluate_v2.py`.")
    cols = st.columns(4)
    for col, (n, l) in zip(cols, [
        (f"{v2['retrieval']['top1']:.0%}", f"Top-1 retrieval (n={v2['retrieval']['n']})"),
        (f"{v2['retrieval']['top3']:.0%}", "Top-3 retrieval"),
        (f"{d['answerable_success_top1']:.0%}", "answerable questions answered with the right service"),
        (f"{d['unsupported_refusal_rate']:.0%}", "unanswerable questions declined"),
    ]):
        col.markdown(kpi(n, l), unsafe_allow_html=True)
    lat = res["latency"]
    st.caption(f"False-refusal rate {d['false_refusal_rate']:.0%} · confident answers correct "
               f"{(d['confident_answer_precision'] or 0):.0%} · warm query "
               f"{lat.get('warm_full_answer_ms', lat['warm_retrieval_ms'])['p50']:.0f} ms median for a full answer, "
               f"cold start {lat['cold_start_s']['total']:.1f} s, on {lat['hardware']}")

    st.subheader("V1 setting vs V2 (same corpus, same test questions)")
    rows = []
    for name, blk in (("V1 setting (hybrid α=0.4, no expansion)", base), ("V2 (selected)", v2)):
        rows.append({"system": name, "Top-1": blk["retrieval"]["top1"], "Top-3": blk["retrieval"]["top3"],
                     "MRR": blk["retrieval"]["mrr"],
                     "answerable success": blk["decision"]["answerable_success_top1"],
                     "false refusals": blk["decision"]["false_refusal_rate"],
                     "unanswerable declined": blk["decision"]["unsupported_refusal_rate"]})
    st.dataframe(pd.DataFrame(rows).style.format({c: "{:.0%}" for c in rows[0] if c not in ("system", "MRR")} |
                                                 {"MRR": "{:.3f}"}), hide_index=True, use_container_width=True)

    st.subheader("By language and phrasing (test)")
    lr = [{"slice": f"language: {k}", **{m: v[m] for m in ("n", "top1", "top3", "mrr")}}
          for k, v in v2["retrieval_by_language"].items()]
    lr += [{"slice": f"style: {k}", **{m: v[m] for m in ("n", "top1", "top3", "mrr")}}
           for k, v in v2["retrieval_by_style"].items()]
    st.dataframe(pd.DataFrame(lr), hide_index=True, use_container_width=True)
    pr = res["paraphrase_robustness_test"]["v2"]
    st.caption(f"Paraphrase robustness: in {pr['all_phrasings_top3']:.0%} of test families every phrasing "
               f"retrieves the right service in the top 3 ({pr['all_phrasings_top1']:.0%} at top 1).")

    st.subheader("Is the result real? Leakage checks (test)")
    fr, nf = v2.get("retrieval_fresh_families"), v2.get("retrieval_lexicon_not_fired")
    ab = res.get("lexicon_ablation", {}).get("selected_without_lexicon_expansion", {}).get("test")
    checks = []
    if fr:
        checks.append({"check": "Families written after the vocabulary list was frozen", "n": fr["n"],
                       "Top-1": fr["top1"], "Top-3": fr["top3"]})
    if nf:
        checks.append({"check": "Questions where the vocabulary list did not fire at all", "n": nf["n"],
                       "Top-1": nf["top1"], "Top-3": nf["top3"]})
    if ab:
        checks.append({"check": "Selected system with query expansion switched off", "n": ab["n"],
                       "Top-1": ab["top1"], "Top-3": ab["top3"]})
    if checks:
        st.dataframe(pd.DataFrame(checks).style.format({"Top-1": "{:.0%}", "Top-3": "{:.0%}"}),
                     hide_index=True, use_container_width=True)
    sh = res.get("selection_history")
    if sh:
        st.caption(f"Selection rule: {sh['final_rule']}. Disclosure: the first run of the final pass used "
                   f"'{sh['first_run_rule']}', which picked {sh['first_run_choice']} on a 45-question split by a "
                   f"0.002 MRR margin; the rule was changed after that run (see DEVELOPMENT_LOG.md).")

    st.subheader("Every question (test split)")
    pq = config.EVAL_DIR / "per_query_results_v2.csv"
    if pq.exists():
        df = pd.read_csv(pq)
        st.dataframe(df[df.split == "test"][["query", "language", "style", "expected_service_id", "top1_service_id",
                                            "rank", "top_score", "decision"]], hide_index=True,
                     use_container_width=True, height=380)
    st.info("Limitations: the benchmark was written by the project author (AI-assisted); coverage is limited to the "
            "indexed agencies. Results describe this corpus and question set, not Saudi government services in "
            "general. V1's original benchmark and results are kept in data/evaluation/v1_baseline/.")
    footer(t)


def page_about():
    t = header()
    st.title(t["nav_about"])
    p = config.ROOT / "docs" / "methodology.md"
    st.markdown(p.read_text(encoding="utf-8") if p.exists() else "See README.md")
    st.graphviz_chart("""
digraph { rankdir=TB; node [shape=box style="rounded,filled" fillcolor="#F1EEE6" color="#0F6B5A" fontname="Helvetica"];
 A [label="Official Saudi service pages\\n(10 agencies, EN + AR)"]; B [label="Capture (browser, low rate)\\n+ SHA-256"];
 C [label="Parse + normalise + validate\\n(quarantine unverified)"]; D [label="Knowledge base\\nservices.jsonl"];
 E [label="Section chunks + title chunks"]; F [label="Multilingual embeddings\\n(MiniLM-L12, cached)"];
 G [label="Arabic / English question"]; X [label="Terminology expansion\\n(lay → official terms)"];
 H [label="Hybrid retrieval\\n(dense + char n-grams + BM25\\n+ title coverage)"];
 I [label="Calibrated decision\\nanswer · tentative · decline"];
 J [label="Grounded synthesis\\n(official passages, [n] citations)"];
 A->B->C->D->E->F->H; G->X->H->I->J; }""")
    footer(t)


pages = [
    st.Page(page_ask, title="Ask Dalil", icon=":material/chat:", default=True),
    st.Page(page_explore, title="Explore services", icon=":material/search:", url_path="explore"),
    st.Page(page_kb, title="Knowledge base", icon=":material/database:", url_path="knowledge-base"),
    st.Page(page_eval, title="Evaluation", icon=":material/monitoring:", url_path="evaluation"),
    st.Page(page_about, title="About", icon=":material/info:", url_path="about"),
]
st.navigation(pages, position="top").run()
