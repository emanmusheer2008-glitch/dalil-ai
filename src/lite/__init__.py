"""Dalil V3 'lite' public runtime: the V2 pipeline without PyTorch / sentence-transformers.

V2 (src/engine.py, src/retrieval/retriever.py) stays the research baseline. V3 reuses its
chunks, character-n-gram TF-IDF, BM25, bilingual lexicon, title coverage and answer
synthesis, and replaces the transformer encoder with an optional *static* embedding table
(numpy only). See docs/V3_LITE.md.
"""
