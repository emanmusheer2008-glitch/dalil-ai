"""Lexical (keyword) retrieval baseline: TF-IDF over character n-grams.

Character n-grams (3-5, within word boundaries) are used instead of whole
words because Arabic attaches prefixes/suffixes to words (e.g. "والسجل",
"بالسجل", "السجلات"); n-grams still overlap across these forms. Text is passed
through light Arabic orthographic normalisation first.

Limitation (by design, and measured in the evaluation): lexical matching cannot
connect an Arabic question to English text or vice-versa.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.arabic import normalize_for_matching


class LexicalIndex:
    def __init__(self, texts: list[str]):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            preprocessor=normalize_for_matching,
            sublinear_tf=True,
            min_df=1,
        )
        self.matrix = self.vectorizer.fit_transform(texts)  # rows are L2-normalised

    def scores(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        return (self.matrix @ q.T).toarray().ravel().astype(np.float32)
