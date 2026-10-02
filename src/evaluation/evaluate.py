"""Run the Dalil benchmark.

    python -m src.evaluation.evaluate

1. Scores every benchmark question against the cached index with three
   methods (lexical, dense, hybrid with several weights).
2. On the *calibration* split only, picks the best method (by MRR, then Top-1)
   and the refusal threshold (by balanced accuracy).
3. Reports all metrics on the held-out *test* split (and on all queries, for
   reference), broken down by language and query type.
4. Runs a cross-lingual experiment: Arabic questions searched against the
   English text only, and English questions against the Arabic text only.
5. Measures end-to-end retrieval latency.

Writes ``data/evaluation/results.json``, ``per_query_results.csv`` and the
calibrated ``data/processed/retrieval_config.json`` used by the app.
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import config
from src.evaluation.metrics import (
    choose_threshold,
    first_relevant_rank,
    refusal_summary,
    retrieval_summary,
)
from src.retrieval.retriever import Retriever

ALPHAS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def configs() -> list[tuple[str, str, float]]:
    out = [("lexical", "lexical", 0.0), ("dense", "dense", 1.0)]
    out += [(f"hybrid_a{a:.1f}", "hybrid", a) for a in ALPHAS]
    return out


def service_ranking(retriever: Retriever, combined: np.ndarray) -> tuple[list[str], np.ndarray]:
    best = np.full(len(retriever._service_ids), -np.inf, dtype=np.float32)
    np.maximum.at(best, retriever._service_codes, combined)
    order = np.argsort(-best)
    order = order[np.isfinite(best[order])]
    return [retriever._service_ids[i] for i in order], best[order]


def score_all(retriever: Retriever, bench: pd.DataFrame, langs=None) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {name: [] for name, _, _ in configs()}
    for q in bench.itertuples():
        dense, lex = retriever.chunk_scores(q.query, langs)
        relevant = {q.expected_service_id} | set(filter(None, str(q.acceptable_service_ids).split(";")))
        relevant.discard("")
        for name, method, alpha in configs():
            comb = dense if method == "dense" else lex if method == "lexical" else alpha * dense + (1 - alpha) * lex
            ranked, scores = service_ranking(retriever, comb)
            rank = first_relevant_rank(ranked, relevant) if q.supported else None
            strict = first_relevant_rank(ranked, {q.expected_service_id}) if q.supported else None
            rows[name].append({
                "query_id": q.query_id, "query": q.query, "language": q.language,
                "query_type": q.query_type, "supported": int(q.supported), "split": q.split,
                "expected_service_id": q.expected_service_id,
                "top1_service_id": ranked[0], "top3_service_ids": ";".join(ranked[:3]),
                "top_score": float(scores[0]), "second_score": float(scores[1]) if len(scores) > 1 else None,
                "rank": rank, "strict_rank": strict,
            })
    return rows


def summarise(df: pd.DataFrame) -> dict:
    sup = df[df.supported == 1]
    ranks = [None if pd.isna(r) else int(r) for r in sup["rank"]]
    strict = [None if pd.isna(r) else int(r) for r in sup["strict_rank"]]
    out = retrieval_summary(ranks)
    out["top1_strict"] = retrieval_summary(strict).get("top1")
    return out


def breakdown(df: pd.DataFrame) -> dict:
    sup = df[df.supported == 1]
    res = {"overall": summarise(df)}
    res["by_language"] = {lang: summarise(g) for lang, g in sup.groupby("language")}
    res["by_query_type"] = {t: summarise(g) for t, g in sup.groupby("query_type")}
    return res


def refusal_block(df: pd.DataFrame, threshold: float) -> dict:
    scores = df["top_score"].to_numpy()
    supported = df["supported"].to_numpy().astype(bool)
    correct = (df["rank"] == 1).to_numpy()
    out = refusal_summary(scores, supported, correct, threshold)
    uns = df[df.supported == 0]
    out["unsupported_refused_by_kind"] = {
        kind: float(np.mean(g["top_score"] < threshold)) for kind, g in uns.groupby("query_type")
    }
    out["unsupported_refused_by_language"] = {
        lang: float(np.mean(g["top_score"] < threshold)) for lang, g in uns.groupby("language")
    }
    sup = df[df.supported == 1]
    out["supported_answered_by_language"] = {
        lang: float(np.mean(g["top_score"] >= threshold)) for lang, g in sup.groupby("language")
    }
    return out


def main() -> dict:
    bench = pd.read_csv(config.BENCHMARK_CSV, dtype={"acceptable_service_ids": str}, keep_default_na=False)
    bench["supported"] = bench["supported"].astype(int)
    retriever = Retriever.from_disk()
    known = set(retriever._service_ids)
    missing = sorted(set(bench.loc[bench.supported == 1, "expected_service_id"]) - known)
    if missing:
        raise SystemExit(f"benchmark references services not in the index: {missing}")

    retriever.chunk_scores("warm-up")  # load the model before timing anything
    all_rows = score_all(retriever, bench)
    frames = {name: pd.DataFrame(rows) for name, rows in all_rows.items()}

    # ---- 1. model selection on calibration split ---------------------------
    comparison = {}
    for name, df in frames.items():
        comparison[name] = {
            "calibration": summarise(df[df.split == "calibration"]),
            "test": summarise(df[df.split == "test"]),
            "all": summarise(df),
        }
    chosen = max(comparison, key=lambda n: (comparison[n]["calibration"]["mrr"],
                                            comparison[n]["calibration"]["top1"]))
    method, alpha = next((m, a) for n, m, a in configs() if n == chosen)
    df = frames[chosen]

    # ---- 2. refusal threshold on calibration split -------------------------
    cal = df[df.split == "calibration"]
    threshold, cal_metrics = choose_threshold(
        cal["top_score"].to_numpy(), cal["supported"].to_numpy().astype(bool), (cal["rank"] == 1).to_numpy()
    )
    test = df[df.split == "test"]

    # threshold sensitivity curve (all queries) for the dashboard
    curve = []
    for t in np.round(np.arange(0.20, 0.80, 0.01), 2):
        m = refusal_summary(df["top_score"].to_numpy(), df["supported"].to_numpy().astype(bool),
                            (df["rank"] == 1).to_numpy(), t)
        curve.append({"threshold": float(t), "supported_answered_correct_top1": m["supported_answered_correct_top1"],
                      "unsupported_refused": m["unsupported_refused"]})

    # ---- 3. cross-lingual experiment ---------------------------------------
    cross = {}
    for label, qlang, index_langs in (("ar_query_en_text", "ar", ("en",)), ("en_query_ar_text", "en", ("ar",)),
                                      ("ar_query_ar_text", "ar", ("ar",)), ("en_query_en_text", "en", ("en",))):
        sub = bench[(bench.language == qlang) & (bench.supported == 1)]
        rows = score_all(retriever, sub, langs=index_langs)
        cross[label] = {name: summarise(pd.DataFrame(rows[name])) for name in ("lexical", "dense", chosen)}

    # ---- 4. latency (full search call, chosen config) -----------------------
    lat = []
    for q in bench["query"]:
        t0 = time.perf_counter()
        retriever.search(q, method=method, alpha=alpha)
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmark": {
            "n_queries": int(len(bench)),
            "n_supported": int(bench.supported.sum()),
            "n_unsupported": int((bench.supported == 0).sum()),
            "by_language": bench.language.value_counts().to_dict(),
            "by_split": bench.split.value_counts().to_dict(),
        },
        "index": {"n_services": int(len(known)), "n_chunks": int(len(retriever.chunks))},
        "method_comparison": comparison,
        "selected": {"config": chosen, "method": method, "alpha": alpha,
                     "selection_rule": "highest calibration MRR, then Top-1"},
        "retrieval_test": breakdown(test),
        "retrieval_all": breakdown(df),
        "refusal": {
            "threshold": threshold,
            "threshold_rule": "maximise balanced accuracy on calibration split",
            "calibration": cal_metrics,
            "test": refusal_block(test, threshold),
            "all": refusal_block(df, threshold),
            "curve_all_queries": curve,
        },
        "cross_lingual": cross,
        "latency_ms": {"mean": float(lat.mean()), "p50": float(np.percentile(lat, 50)),
                       "p95": float(np.percentile(lat, 95)), "n": int(len(lat)),
                       "hardware": f"{platform.processor() or platform.machine()} CPU, {platform.system()}"},
    }
    config.EVAL_RESULTS_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    df.assign(refused=df["top_score"] < threshold).to_csv(config.EVAL_PER_QUERY_CSV, index=False, encoding="utf-8")
    config.RETRIEVAL_CONFIG_JSON.write_text(json.dumps({
        "method": method, "alpha": alpha, "threshold": round(threshold, 4), "calibrated": True,
        "calibrated_at": results["generated_at"], "source": "src/evaluation/evaluate.py",
    }, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    r = main()
    sel = r["selected"]["config"]
    print(f"Selected: {sel}  threshold={r['refusal']['threshold']:.3f}")
    for name, v in r["method_comparison"].items():
        c, t = v["calibration"], v["test"]
        print(f"  {name:14s} cal MRR={c['mrr']:.3f} top1={c['top1']:.3f} | test MRR={t['mrr']:.3f} "
              f"top1={t['top1']:.3f} top3={t['top3']:.3f}")
    print("Test refusal:", {k: round(v, 3) for k, v in r["refusal"]["test"].items() if isinstance(v, float)})
    print("Latency ms:", {k: round(v, 1) if isinstance(v, float) else v for k, v in r["latency_ms"].items()})
