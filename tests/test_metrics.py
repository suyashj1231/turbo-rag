"""Executable spec for eval/metrics.py — hand-computed tiny cases."""

from turbo_rag.eval.metrics import hit_at_k, mrr, recall_at_k

RETRIEVED = ["a", "b", "c", "d", "e"]


def test_hit_at_k():
    assert hit_at_k(RETRIEVED, gold={"c"}, k=5) == 1.0
    assert hit_at_k(RETRIEVED, gold={"c"}, k=2) == 0.0
    assert hit_at_k(RETRIEVED, gold={"z"}, k=5) == 0.0


def test_recall_at_k():
    assert recall_at_k(RETRIEVED, gold={"a", "e", "z", "y"}, k=5) == 0.5
    assert recall_at_k(RETRIEVED, gold={"a", "b"}, k=1) == 0.5
    assert recall_at_k(RETRIEVED, gold={"z"}, k=5) == 0.0


def test_mrr():
    assert mrr(RETRIEVED, gold={"a"}) == 1.0
    assert mrr(RETRIEVED, gold={"c", "e"}) == 1 / 3  # first gold found is rank 3
    assert mrr(RETRIEVED, gold={"z"}) == 0.0
