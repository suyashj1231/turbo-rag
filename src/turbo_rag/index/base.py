"""The one interface every index implements. Provided — this is the contract
that lets Stage 4 swap TurboQuant in without touching the rest of the pipeline.

Every index (flat float32, naive binary, TurboQuant at 1/2/4 bits) speaks this
protocol, so the benchmark is a for-loop over index implementations.
"""

from typing import Protocol

import numpy as np


class VectorIndex(Protocol):
    def add(self, vectors: np.ndarray) -> None:
        """Index (n, d) unit-normalized float32 vectors. Row i keeps id i."""
        ...

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (ids, scores), each shape (k,), best first.

        Scores are (estimated) inner products with the query.
        """
        ...

    def memory_bytes(self) -> int:
        """Bytes actually used to store the vectors — the compression headline."""
        ...
