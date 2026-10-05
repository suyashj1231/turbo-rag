"""Stage 2: retrieval metrics.

Each eval question has "gold" chunk_ids: chunks that contain the answer.
Retrieval metrics ask: did the search bring the gold back?

- hit@k:      1.0 if ANY gold chunk is in the top k, else 0.0
- recall@k:   fraction of gold chunks that appear in the top k
- MRR:        1 / rank of the FIRST gold chunk found (1-based); 0 if none.
              Rewards putting gold at position 1 over position 5.

Averaged over questions these become the scoreboard. They need no LLM, so
they're deterministic, free, and the right thing to gate CI on.
"""


def hit_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    return 1.0 if any(r in gold for r in retrieved[:k]) else 0.0


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(gold & set(retrieved[:k])) / len(gold)


def mrr(retrieved: list[str], gold: set[str]) -> float:
    for rank, r in enumerate(retrieved, start=1):
        if r in gold:
            return 1.0 / rank
    return 0.0
