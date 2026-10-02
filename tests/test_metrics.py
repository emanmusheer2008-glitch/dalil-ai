import numpy as np
import pytest

from src.evaluation.metrics import (
    choose_threshold,
    first_relevant_rank,
    refusal_summary,
    retrieval_summary,
)


def test_rank_and_summary():
    assert first_relevant_rank(["a", "b", "c"], {"c", "x"}) == 3
    assert first_relevant_rank(["a"], {"z"}) is None
    s = retrieval_summary([1, 2, None, 4])
    assert s["top1"] == 0.25 and s["top3"] == 0.5
    assert s["mrr"] == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)


def test_refusal_summary_counts():
    scores = np.array([0.9, 0.8, 0.2, 0.6])
    supported = np.array([True, True, False, False])
    correct = np.array([True, False, False, False])
    m = refusal_summary(scores, supported, correct, threshold=0.5)
    assert m["refusal_recall"] == 0.5 and m["refusal_precision"] == 1.0
    assert m["supported_answered"] == 1.0
    assert m["end_to_end_accuracy"] == 0.5          # q1 right, q3 refused right


def test_choose_threshold_separates_clean_data():
    scores = np.array([0.8, 0.7, 0.75, 0.3, 0.35])
    supported = np.array([True, True, True, False, False])
    correct = np.ones(5, dtype=bool)
    t, m = choose_threshold(scores, supported, correct)
    assert 0.35 < t < 0.7
    assert m["balanced_accuracy"] == 1.0
