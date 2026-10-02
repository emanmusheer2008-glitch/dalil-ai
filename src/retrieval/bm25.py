"""Word-level BM25 with light bilingual normalisation (pure NumPy / SciPy).

Complements character n-gram TF-IDF: BM25 rewards documents containing the
*words* of the query, weighted by rarity, and saturates repeated terms.
Tokens are light-stemmed:

* Arabic: common proclitics (و، ف، ب، ل، ك، ال and combinations) and a few
  suffixes (ات، ين، ون، ها، هم) are stripped, after orthographic normalisation.
* English: lower-cased, possessive/plural "s", "es", "ing", "ed" stripped.
"""
from __future__ import annotations

import re

import numpy as np
from scipy import sparse

from src.utils.arabic import normalize_arabic

_TOKEN = re.compile(r"[a-z0-9]+|[ء-ي]+")
_AR_PREFIX = re.compile(r"^(?:وال|فال|بال|كال|لل|ال|و|ف|ب|ل|ك)(?=[ء-ي]{3,})")
_AR_SUFFIX = re.compile(r"(?:ات|ين|ون|ها|هم|ه)$")
_EN_STOP = {
    "the", "a", "an", "of", "to", "for", "in", "on", "and", "or", "is", "are", "be", "my", "i", "me", "do", "does",
    "how", "can", "what", "where", "who", "which", "with", "by", "at", "it", "this", "that", "from", "as", "your",
    "you", "want", "need", "get", "please", "about", "should", "would", "will", "am", "if", "there", "any", "we",
}
_AR_STOP = {"كيف", "ما", "ماذا", "هل", "من", "في", "على", "الى", "عن", "او", "ان", "انا", "ابغي", "ابي", "ودي", "وش",
            "ايش", "هذا", "هذه", "التي", "الذي", "مع", "لي", "كم", "متي", "اين", "وين"}


def tokenize(text: str) -> list[str]:
    text = normalize_arabic(text).lower()
    out = []
    for tok in _TOKEN.findall(text):
        if tok[0] < "؀":                    # latin / digits
            if tok in _EN_STOP:
                continue
            for suf in ("ing", "ed", "es", "s"):
                if len(tok) > len(suf) + 2 and tok.endswith(suf):
                    tok = tok[: -len(suf)]
                    break
        else:
            if tok in _AR_STOP:
                continue
            tok = _AR_PREFIX.sub("", tok)
            if len(tok) > 4:
                tok = _AR_SUFFIX.sub("", tok)
        out.append(tok)
    return out


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75):
        docs = [tokenize(t) for t in texts]
        vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        for i, toks in enumerate(docs):
            counts: dict[int, int] = {}
            for t in toks:
                j = vocab.setdefault(t, len(vocab))
                counts[j] = counts.get(j, 0) + 1
            for j, c in counts.items():
                rows.append(i)
                cols.append(j)
                vals.append(c)
        n = len(docs)
        tf = sparse.csr_matrix((vals, (rows, cols)), shape=(n, len(vocab)), dtype=np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = dl.mean() if n else 1.0
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        # precompute the saturated tf component per (doc, term)
        tf = tf.tocoo()
        denom = tf.data + k1 * (1 - b + b * dl[tf.row] / avgdl)
        w = tf.data * (k1 + 1) / denom
        self.weights = sparse.csc_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)
        self.vocab = vocab
        self.n = n

    def scores(self, query: str) -> np.ndarray:
        ids = [self.vocab[t] for t in set(tokenize(query)) if t in self.vocab]
        if not ids:
            return np.zeros(self.n, dtype=np.float32)
        s = self.weights[:, ids] @ self.idf[ids]
        return np.asarray(s).ravel().astype(np.float32)
