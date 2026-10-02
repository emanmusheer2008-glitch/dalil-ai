"""Evaluate the V3 lite runtime on the SAME V2 benchmark and protocol.

    python -m src.evaluation.evaluate_v3

Same questions, same dev/val/test split, same rules as V2 (src/evaluation/evaluate_v2.py):
* configuration chosen by best MRR on dev+val pooled; nothing is tuned on test;
* thresholds fitted on dev+val (>=80 % unsupported refusal; >=90 % confident precision);
* related-links floor just above the highest out-of-domain dev+val score.

Candidate configurations (all torch-free at runtime):
* lexical only: char-n-gram TF-IDF (+ BM25) (+ title coverage) (+ lexicon expansion)
* + static word embeddings, document side "static" (static chunk vectors)
* + static word embeddings, document side "model" (V2 transformer chunk vectors, static queries)

Writes results_v3.json, per_query_results_v3.csv and data/processed/retrieval_config_v3.json.
"""
from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import config
from src.evaluation import evaluate_v2 as ev
from src.ingestion.pipeline import load_services
from src.lite.retriever import LiteRetriever

RESULTS_V3 = config.EVAL_DIR / "results_v3.json"
PER_QUERY_V3 = config.EVAL_DIR / "per_query_results_v3.csv"
CONFIG_V3 = config.PROCESSED_DIR / "retrieval_config_v3.json"


def grid(dense_options):
    for wd, wc, wb, exp, tc in itertools.product(dense_options, (0.05, 0.15, 0.3, 0.45), (0.0, 0.15, 0.3),
                                                 (False, True), (0.0, 0.1, 0.2)):
        if wc == 0 and wb == 0 and wd == 0:
            continue
        yield {"method": "v2", "alpha": 0.0, "title": True,
               "params": {"w_dense": wd, "w_char": wc, "w_bm25": wb, "expand": exp, "expand_dense": False,
                          "w_tcov": tc}}


def main() -> dict:
    recs = load_services()
    bench = pd.read_csv(ev.BENCH_V2, dtype=str, keep_default_na=False)
    bench["supported"] = bench["supported"].astype(int)
    bench["fresh"] = bench["fresh"].replace("", "0").astype(int)
    bench = ev.resolve_expected(bench, recs)

    variants = {}
    lex = LiteRetriever.from_disk(use_static=False)
    variants["lexical"] = (lex, [0.0])
    st = LiteRetriever.from_disk(use_static=True)
    if st.static is not None:
        variants["static_docs"] = (st, [0.15, 0.3, 0.45, 0.6, 0.8])
        st_model = LiteRetriever(st.chunks, st.static, st.lexical, st.bm25)
        st_model.embeddings = np.load(config.EMBEDDINGS_NPY).astype(np.float32)    # V2 doc vectors
        variants["model_docs"] = (st_model, [0.15, 0.3, 0.45, 0.6, 0.8])

    frames, cfgs = {}, {}
    for vname, (retr, dense_opts) in variants.items():
        configs = []
        for c in grid(dense_opts):
            p = c["params"]
            c["name"] = (f"{vname}_d{p['w_dense']}_c{p['w_char']}_b{p['w_bm25']}_x{int(p['expand'])}"
                         f"_tc{p['w_tcov']}")
            c["variant"] = vname
            configs.append(c)
        frames.update(ev.score_queries(retr, bench, configs))
        cfgs.update({c["name"]: c for c in configs})

    def pooled(name):
        f = frames[name]
        m = ev.summ(f[f.split.isin(["dev", "val"])])
        return (round(m["mrr"], 4), round(m["top1"], 4))

    ranked = sorted(cfgs, key=pooled, reverse=True)
    best = {v: next(n for n in ranked if cfgs[n]["variant"] == v) for v in variants}
    chosen = ranked[0]
    df = frames[chosen]
    fit = df[df.split.isin(["dev", "val"])]
    t_ans, t_tent = ev.fit_thresholds(fit)
    ood = fit[(fit.supported == 0) & (fit["style"] == "out_of_domain")]["top_score"]
    t_rel = float(min(t_tent, (ood.max() + 0.005) if len(ood) else t_tent))
    test = df[df.split == "test"]

    dec_ = ev.decide(test["top_score"].to_numpy(), t_ans, t_tent)
    declined = test[dec_ == "declined"]
    rel = declined["top_score"] >= t_rel
    v2 = json.loads((config.EVAL_DIR / "results_v2.json").read_text(encoding="utf-8"))

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "same benchmark/splits/rules as V2; selection = best pooled dev+val MRR; report on test",
        "candidates": len(cfgs),
        "selected": {"name": chosen, **{k: cfgs[chosen][k] for k in ("variant", "params")},
                     "dev_val_pooled": dict(zip(("mrr", "top1"), pooled(chosen)))},
        "static_table": (json.loads((config.PROCESSED_DIR / "static_vocab.json").read_text(encoding="utf-8"))
                         .get("remove_pc") if (config.PROCESSED_DIR / "static_vocab.json").exists() else None),
        "best_per_variant_dev_val_test": {
            v: {"name": n, **{s: ev.summ(frames[n][frames[n].split == s]) for s in ("dev", "val", "test")}}
            for v, n in best.items()},
        "thresholds": {"t_answer": t_ans, "t_tentative": t_tent, "t_related": t_rel},
        "v3_test": {
            "retrieval": ev.summ(test),
            "retrieval_by_language": {k: ev.summ(g) for k, g in test.groupby("language")},
            "retrieval_by_style": {k: ev.summ(g) for k, g in test[test.supported == 1].groupby("style")},
            "decision": ev.decision_metrics(test, t_ans, t_tent),
            "retrieval_fresh_families": ev.summ(test[test.fresh == 1]),
            "related_links": {
                "declined_answerable": int((declined.supported == 1).sum()),
                "declined_answerable_right_service_in_related_top3":
                    int(((declined.supported == 1) & rel & declined["rank"].le(3)).sum()),
                "declined_unanswerable_shown_related": int(((declined.supported == 0) & rel).sum()),
            },
        },
        "v2_reference_test": {"retrieval": v2["v2_test"]["retrieval"], "decision": v2["v2_test"]["decision"],
                              "retrieval_by_language": v2["v2_test"]["retrieval_by_language"]},
    }
    df.assign(decision=ev.decide(df["top_score"].to_numpy(), t_ans, t_tent)).to_csv(PER_QUERY_V3, index=False)
    CONFIG_V3.write_text(json.dumps({
        "runtime": "lite", "variant": cfgs[chosen]["variant"], "method": "v2", "alpha": 0.0,
        "params": cfgs[chosen]["params"], "title_chunks": True, "t_answer": round(t_ans, 4),
        "t_tentative": round(t_tent, 4), "t_related": round(t_rel, 4), "threshold": round(t_tent, 4),
        "calibrated": True, "calibrated_at": results["generated_at"], "source": "src/evaluation/evaluate_v3.py",
    }, indent=2), encoding="utf-8")
    RESULTS_V3.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return results


if __name__ == "__main__":
    r = main()
    print("selected:", r["selected"]["name"], "| dev+val pooled:", r["selected"]["dev_val_pooled"],
          "| static remove_pc:", r["static_table"])
    for v, b in r["best_per_variant_dev_val_test"].items():
        print(f"  {v:12s} {b['name']:45s} dev MRR {b['dev']['mrr']:.3f} val {b['val']['mrr']:.3f} "
              f"test top1 {b['test']['top1']:.3f} top3 {b['test']['top3']:.3f} mrr {b['test']['mrr']:.3f}")
    print("V3 test:", r["v3_test"]["retrieval"])
    print("V3 decision:", {k: round(v, 3) if isinstance(v, float) else v for k, v in r["v3_test"]["decision"].items()})
    print("V2 test:", r["v2_reference_test"]["retrieval"])
