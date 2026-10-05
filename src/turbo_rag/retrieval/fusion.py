"""Reciprocal Rank Fusion (Cormack et al., 2009).

Problem: BM25 scores are unbounded (0..30+), cosine scores live in [-1, 1].
Adding them is meaningless, and min-max normalizing is fragile (one outlier
squashes everything). RRF throws scores away and uses only RANKS:

    RRF(d) = sum over rankers r of  1 / (k + rank_r(d))      (rank is 1-based)

k=60 damps the top: rank 1 vs rank 2 differ a little, not 2x. A doc ranked
decently by BOTH retrievers beats one ranked #1 by only one of them — which is
exactly the agreement signal you want.
"""

from collections import defaultdict


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse several best-first id lists -> [(id, rrf_score)] best first."""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
