"""Stage 2 — YOUR CODE: retrieval metrics.

Each eval question has "gold" chunk_ids: chunks that contain the answer
(you label these by hand when writing the questions). Retrieval metrics ask:
did the search bring the gold back?

- hit@k:      1.0 if ANY gold chunk is in the top k, else 0.0
- recall@k:   fraction of gold chunks that appear in the top k
- MRR:        1 / rank of the FIRST gold chunk found (1-based); 0 if none.
              Rewards putting gold at position 1 over position 5.

These are ~5 lines each. Write tests/test_metrics.py cases first (tiny
hand-computed examples), then implement — hand-checking a metric once on
paper is how you trust every number the scoreboard ever prints.
"""


def hit_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    raise NotImplementedError("TODO(suyash)")


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    raise NotImplementedError("TODO(suyash)")


def mrr(retrieved: list[str], gold: set[str]) -> float:
    raise NotImplementedError("TODO(suyash)")
