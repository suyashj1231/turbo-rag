"""Stage 1, step 4 — YOUR CODE: exact brute-force search. Ground truth.

This is ~15 lines of numpy, but understand each line — every fancier index in
this project is an approximation of THIS, and recall@k is always measured
against its results.

Hints:
- Vectors are unit-normalized, so cosine similarity == inner product ==
  one matrix-vector product: `scores = self.vectors @ query`  # (n,)
- Top-k: `np.argsort` works; `np.argpartition` then sort the k winners is
  O(n) instead of O(n log n) — nice, not required. Either way, return ids
  sorted best-first.
- memory_bytes: `self.vectors.nbytes`. For n chunks at d=384 float32 that's
  n * 1536 bytes — the number TurboQuant will divide by ~32.
"""

import numpy as np


class FlatIndex:
    """Exact inner-product search over float32 vectors."""

    def __init__(self) -> None:
        self.vectors: np.ndarray | None = None

    def add(self, vectors: np.ndarray) -> None:
        raise NotImplementedError("TODO(suyash)")

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("TODO(suyash)")

    def memory_bytes(self) -> int:
        raise NotImplementedError("TODO(suyash)")
