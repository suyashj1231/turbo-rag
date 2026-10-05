"""BM25 and RRF: hand-checkable cases, no models loaded."""

import math

import numpy as np

from turbo_rag.chunking import Chunk
from turbo_rag.index.flat import FlatIndex
from turbo_rag.retrieval.bm25 import BM25Index, tokenize
from turbo_rag.retrieval.fusion import reciprocal_rank_fusion
from turbo_rag.retrieval.retriever import Retriever

DOCS = [
    "the hobbit by tolkien a dragon and a burglar",
    "dragon dragon dragon dragon dragon",
    "a quiet novel about a preacher in iowa",
]


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("The Hobbit, by J.R.R. Tolkien!") == ["hobbit", "j", "r", "r", "tolkien"]


def test_bm25_rare_term_wins():
    idx = BM25Index()
    idx.add(DOCS)
    ids, _ = idx.search("tolkien", k=3)
    assert list(ids) == [0]  # only doc containing the term is returned


def test_bm25_idf_matches_formula():
    idx = BM25Index()
    idx.add(DOCS)
    n, df = 3, 1
    assert math.isclose(idx.idf["tolkien"], math.log(1 + (n - df + 0.5) / (df + 0.5)), rel_tol=1e-6)


def test_bm25_term_frequency_saturates():
    idx = BM25Index()
    idx.add(DOCS)
    scores = idx.scores("dragon")
    # doc 1 has 5x the tf of doc 0 but scores far less than 5x
    assert scores[1] > scores[0]
    assert scores[1] < 2.5 * scores[0]


def test_bm25_no_match_returns_empty():
    idx = BM25Index()
    idx.add(DOCS)
    ids, scores = idx.search("submarine", k=5)
    assert len(ids) == 0 and len(scores) == 0


def test_rrf_rewards_agreement():
    # doc 7 is #2 in both lists; 1 and 2 are #1 in only one list each
    fused = reciprocal_rank_fusion([[1, 7, 3], [2, 7, 4]], k=60)
    assert fused[0][0] == 7
    assert math.isclose(fused[0][1], 2 / 62)


def test_rrf_handles_disjoint_lists():
    fused = dict(reciprocal_rank_fusion([[1], [2]], k=60))
    assert fused == {1: 1 / 61, 2: 1 / 61}


class FakeReranker:
    name = "fake"

    def rerank(self, query, docs, top_n):
        # prefer docs mentioning "iowa", like a cross-encoder that "understands" the query
        order = sorted(range(len(docs)), key=lambda i: ("iowa" not in docs[i], i))
        return [(i, 1.0 if "iowa" in docs[i] else 0.0) for i in order[:top_n]]


def test_retriever_hybrid_rerank_uses_reranker_order(monkeypatch):
    chunks = [Chunk(f"d{i}:0000", f"d{i}", 1, 1, t) for i, t in enumerate(DOCS)]
    vectors = np.eye(3, dtype=np.float32)
    dense = FlatIndex()
    dense.add(vectors)
    bm25 = BM25Index()
    bm25.add(DOCS)
    monkeypatch.setattr("turbo_rag.retrieval.retriever.embed_query", lambda q, m: vectors[1])
    r = Retriever(chunks, dense, bm25, "unused", reranker=FakeReranker(), candidate_k=3)

    assert r.retrieve("dragon", k=1, mode="dense")[0].chunk.chunk_id == "d1:0000"
    assert r.retrieve("preacher iowa", k=1, mode="bm25")[0].chunk.chunk_id == "d2:0000"
    assert r.retrieve("dragon", k=1, mode="hybrid_rerank")[0].chunk.chunk_id == "d2:0000"
