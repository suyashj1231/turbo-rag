"""The retrieval funnel. One class, four modes, so the eval can ablate them:

    dense          query -> embed -> FlatIndex top-k
    bm25           query -> BM25 top-k
    hybrid         dense top-50 + bm25 top-50 -> RRF -> top-k
    hybrid_rerank  dense top-50 + bm25 top-50 -> RRF -> cross-encoder -> top-k

Each step trades latency for precision. Measuring every mode on the same
questions is what turns "I added reranking" into "reranking moved MRR from
X to Y" — the sentence interviewers actually want.
"""

from dataclasses import dataclass

from ..chunking import Chunk
from ..embed import embed_query
from ..index.base import VectorIndex
from .bm25 import BM25Index
from .fusion import reciprocal_rank_fusion
from .rerank import Reranker

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        dense: VectorIndex,
        bm25: BM25Index,
        embedding_model: str,
        reranker: Reranker | None = None,
        candidate_k: int = 50,
        rrf_k: int = 60,
    ) -> None:
        self.chunks = chunks
        self.dense = dense
        self.bm25 = bm25
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def _dense_ids(self, query: str, k: int) -> list[int]:
        ids, _ = self.dense.search(embed_query(query, self.embedding_model), k)
        return [int(i) for i in ids]

    def _bm25_ids(self, query: str, k: int) -> list[int]:
        ids, _ = self.bm25.search(query, k)
        return [int(i) for i in ids]

    def retrieve(self, query: str, k: int, mode: str = "hybrid_rerank") -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"unknown retrieval mode {mode!r}; choose from {MODES}")

        if mode == "dense":
            ids, scores = self.dense.search(embed_query(query, self.embedding_model), k)
            return [Hit(self.chunks[i], float(s)) for i, s in zip(ids, scores)]
        if mode == "bm25":
            ids, scores = self.bm25.search(query, k)
            return [Hit(self.chunks[i], float(s)) for i, s in zip(ids, scores)]

        fused = reciprocal_rank_fusion(
            [self._dense_ids(query, self.candidate_k), self._bm25_ids(query, self.candidate_k)],
            k=self.rrf_k,
        )
        if mode == "hybrid" or self.reranker is None:
            return [Hit(self.chunks[i], s) for i, s in fused[:k]]

        candidates = [i for i, _ in fused[: self.candidate_k]]
        ranked = self.reranker.rerank(query, [self.chunks[i].text for i in candidates], top_n=k)
        return [Hit(self.chunks[candidates[j]], float(s)) for j, s in ranked]

