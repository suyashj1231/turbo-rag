"""Stage 1, step 4: exact brute-force search. Ground truth.

Every fancier index in this project is an approximation of THIS, and recall@k
is always measured against its results.

- Vectors are unit-normalized, so cosine == inner product == one mat-vec.
- Top-k via argpartition (O(n)) then sort only the k winners (O(k log k)),
  instead of a full O(n log n) argsort.
- At 7k books x 384 dims that's ~11 MB and ~1 ms per query. Brute force is
  the right call until ~1M vectors; then you reach for HNSW/IVF/quantization.
"""

import numpy as np


class FlatIndex:
    """Exact inner-product search over float32 vectors."""

    def __init__(self) -> None:
        self.vectors: np.ndarray | None = None

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = self.vectors @ query.astype(np.float32)  # (n,)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return top, scores[top]

    def memory_bytes(self) -> int:
        return 0 if self.vectors is None else self.vectors.nbytes
