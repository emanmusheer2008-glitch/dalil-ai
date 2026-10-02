"""Evaluate V4 deterministic retrieval (V3 + action-aware rerank) on the SAME benchmark/protocol.

    python -m src.evaluation.evaluate_v4

* candidates = V3 lite search top-10 for every benchmark question (unchanged V3 config);
* the action rerank weights (bonus, penalty) are chosen by pooled dev+val MRR (ties -> smaller);
* decisions use V3's calibrated thresholds on the RAW score of the new top-1 (rerank can't add confidence);
* the Gemini query-rewrite path is NOT part of this deterministic evaluation (see docs/V4.md).
Writes data/evaluation/results_v4.json and data/processed/v4_config.json.
"""
from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone

import pandas as pd

from src import config
from src.evaluation import evaluate_v2 as ev
from src.evaluation.metrics import first_relevant_rank
from src.lite.engine import DalilLite
from src.v4.actions import query_actions, rerank_hits

RESULTS_V4 = config.EVAL_DIR / "results_v4.json"
CONFIG_V4 = config.PROCESSED_DIR / "v4_config.json"
GRID = list(itertools.product((0.0, 0.02, 0.04, 0.06, 0.08), (0.0, 0.03, 0.06, 0.1, 0.15)))


def main() -> dict:
    eng = DalilLite()
    bench = pd.read_csv(ev.BENCH_V2, dtype=str, keep_default_na=False)
    bench["supported"] = bench["supported"].astype(int)
    bench["fresh"] = bench["fresh"].replace("", "0").astype(int)
    bench = ev.resolve_expected(bench, eng.records)
    cands = {q.query_id: eng.search(q.query, top_k=10).hits for q in bench.itertuples()}

    def frame(b, p):
        rows = []
        for q in bench.itertuples():
            hits = rerank_hits(cands[q.query_id], eng.records, q.query, b, p)
            ranked = [h.service_id for h in hits]
            rel = ({q.expected_service_id} | set(filter(None, q.acceptable_service_ids.split(";")))) - {""}
            rows.append({"query_id": q.query_id, "split": q.split, "query": q.query, "language": q.language,
                         "style": q.style, "supported": q.supported, "fresh": q.fresh,
                         "expected_service_id": q.expected_service_id, "top1_service_id": ranked[0],
                         "top_score": hits[0].score, "action_detected": bool(query_actions(q.query)[0]),
                         "rank": first_relevant_rank(ranked, rel) if q.supported else None})
        return pd.DataFrame(rows)

    frames = {(b, p): frame(b, p) for b, p in GRID}

    def pooled(k):
        f = frames[k]
        m = ev.summ(f[f.split.isin(["dev", "val"])])
        return (round(m["mrr"], 4), round(m["top1"], 4), -k[0] - k[1])

    best = max(frames, key=pooled)
    t_a, t_t = eng.t_answer, eng.t_tentative
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "protocol": "V3 top-10 candidates + action rerank; weights by pooled dev+val MRR; thresholds = V3",
           "selected": {"bonus": best[0], "penalty": best[1], "dev_val_pooled_mrr": pooled(best)[0]}}
    for name, key in (("v3", (0.0, 0.0)), ("v4", best)):
        f = frames[key]
        test = f[f.split == "test"]
        out[name] = {"dev_val": ev.summ(f[f.split.isin(["dev", "val"])]), "test": ev.summ(test),
                     "test_by_language": {k: ev.summ(g) for k, g in test.groupby("language")},
                     "test_action_questions": ev.summ(test[test.action_detected]),
                     "test_decision": ev.decision_metrics(test, t_a, t_t)}
    v3t, v4t = frames[(0.0, 0.0)], frames[best]
    t3, t4 = v3t[v3t.split == "test"], v4t[v4t.split == "test"]
    out["test_changed_top1"] = [
        {"query": a.query, "v3_top1": a.top1_service_id, "v4_top1": b.top1_service_id,
         "expected": a.expected_service_id, "v3_correct": a.rank == 1, "v4_correct": b.rank == 1}
        for a, b in zip(t3.itertuples(), t4.itertuples()) if a.top1_service_id != b.top1_service_id]
    RESULTS_V4.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    CONFIG_V4.write_text(json.dumps({"action_bonus": best[0], "action_penalty": best[1],
                                     "source": "src/evaluation/evaluate_v4.py",
                                     "calibrated_at": out["generated_at"]}, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    r = main()
    print("selected", r["selected"])
    for n in ("v3", "v4"):
        d = r[n]["test_decision"]
        print(n, "test", {k: round(v, 3) for k, v in r[n]["test"].items() if k != "n"}, "| action-qs",
              {k: round(v, 3) for k, v in r[n]["test_action_questions"].items()}, "| unsup refusal",
              round(d["unsupported_refusal_rate"], 3), "conf prec", round(d["confident_answer_precision"], 3),
              "unsup conf", round(d["unsupported_confident_answer_rate"], 3), "false refusal",
              round(d["false_refusal_rate"], 3))
    for c in r["test_changed_top1"]:
        print("  changed:", c)
