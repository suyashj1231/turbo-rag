"""Stage 1, step 3: chunks -> vectors.

The embedding model maps text to a point in R^d such that semantically similar
texts land close together. That single property is what makes dense retrieval
work. We call a frozen pre-trained model; we never train it.

- normalize_embeddings=True makes every vector unit-length, so inner product
  == cosine similarity, and search is one matrix-vector product.
- The model is loaded once per process (lru_cache): loading costs ~1s,
  encoding one query costs ~5ms. Reloading per query is a classic latency bug.
- Queries embed with the same model as documents. all-MiniLM-L6-v2 needs no
  "query: " prefix; e5/bge models do, so check when swapping.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=2)
def _model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    """Embed texts -> (n, d) float32, unit-normalized rows."""
    vectors = _model(model_name).encode(
        texts, normalize_embeddings=True, show_progress_bar=len(texts) > 100, batch_size=64
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(query: str, model_name: str) -> np.ndarray:
    """Embed one query -> (d,) float32, unit-normalized."""
    return embed_texts([query], model_name)[0]
