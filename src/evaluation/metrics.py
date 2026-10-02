"""Retrieval and refusal metrics (pure functions, unit-tested)."""
from __future__ import annotations

import numpy as np


def first_relevant_rank(ranked_ids: list[str], relevant: set[str]) -> int | None:
    """1-based rank of the first relevant id, or None if absent."""
    for i, sid in enumerate(ranked_ids, 1):
        if sid in relevant:
            return i
    return None


def hit_at_k(rank: int | None, k: int) -> float:
    return float(rank is not None and rank <= k)


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def retrieval_summary(ranks: list[int | None]) -> dict:
    n = len(ranks)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "top1": float(np.mean([hit_at_k(r, 1) for r in ranks])),
        "top3": float(np.mean([hit_at_k(r, 3) for r in ranks])),
        "top5": float(np.mean([hit_at_k(r, 5) for r in ranks])),
        "mrr": float(np.mean([reciprocal_rank(r) for r in ranks])),
    }


def refusal_summary(scores: np.ndarray, supported: np.ndarray, top1_correct: np.ndarray, threshold: float) -> dict:
    """Metrics for the answer/refuse decision at a given threshold.

    "positive" = Dalil refuses. A supported question that is answered with the
    wrong service counts as an error in ``end_to_end_accuracy``.
    """
    refuse = scores < threshold
    unsup = ~supported
    tp = int(np.sum(refuse & unsup))
    fp = int(np.sum(refuse & supported))
    fn = int(np.sum(~refuse & unsup))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    sup_correct = supported & ~refuse & top1_correct
    correct = sup_correct | (unsup & refuse)
    return {
        "threshold": float(threshold),
        "refusal_precision": precision,
        "refusal_recall": recall,
        "refusal_f1": f1,
        "unsupported_refused": float(np.mean(refuse[unsup])) if unsup.any() else None,
        "supported_answered": float(np.mean(~refuse[supported])) if supported.any() else None,
        "supported_answered_correct_top1": float(np.mean(sup_correct[supported])) if supported.any() else None,
        "end_to_end_accuracy": float(np.mean(correct)),
        "balanced_accuracy": float(
            0.5 * (np.mean(sup_correct[supported]) + np.mean(refuse[unsup]))
        ) if supported.any() and unsup.any() else None,
    }


def choose_threshold(scores: np.ndarray, supported: np.ndarray, top1_correct: np.ndarray) -> tuple[float, dict]:
    """Pick the threshold maximising balanced accuracy on a calibration set.

    Candidates are midpoints between consecutive observed top scores. Ties are
    broken towards the widest gap (a more robust separation), then the lower
    threshold (answer more often).
    """
    s = np.sort(np.unique(scores))
    cands = np.concatenate([[s[0] - 1e-3], (s[:-1] + s[1:]) / 2, [s[-1] + 1e-3]])
    gaps = np.concatenate([[0.0], s[1:] - s[:-1], [0.0]])
    best, best_key = None, None
    for t, g in zip(cands, gaps):
        m = refusal_summary(scores, supported, top1_correct, t)
        key = (round(m["balanced_accuracy"], 9), g, -t)
        if best_key is None or key > best_key:
            best, best_key = (float(t), m), key
    return best
