"""Stage 1, step 3 — YOUR CODE: chunks -> vectors.

The embedding model maps text to a point in R^d such that semantically similar
texts land close together. That single property is what makes all of RAG work.
You are not training it — you're calling a frozen pre-trained model.

Hints:
- `from sentence_transformers import SentenceTransformer`
- `model.encode(list_of_texts, normalize_embeddings=True, show_progress_bar=True)`
- normalize_embeddings=True makes every vector unit-length, so inner product
  == cosine similarity. Do this. It simplifies everything downstream, and
  TurboQuant's inner-product estimator (Stage 3) assumes it too.
- Return float32, shape (n_chunks, dim). Save with np.save; loading embeddings
  must not require re-running the model.
- Queries embed with the same model. Some models want a prefix for queries
  (e.g. "query: ...") — all-MiniLM-L6-v2 does not, but check when you swap models.

First self-check (do this before wiring anything else): embed
["the cat sat on the mat", "a feline rested on the rug", "quarterly revenue grew 4%"]
and print the pairwise similarity matrix. If sentence 1 and 2 aren't clearly
closer to each other than to 3, something is wrong.
"""

import numpy as np


def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    """Embed texts -> (n, d) float32, unit-normalized rows."""
    raise NotImplementedError("TODO(suyash)")


def embed_query(query: str, model_name: str) -> np.ndarray:
    """Embed one query -> (d,) float32, unit-normalized."""
    raise NotImplementedError("TODO(suyash)")
