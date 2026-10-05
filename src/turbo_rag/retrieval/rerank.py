"""Cross-encoder reranking: retrieve wide and cheap, then rerank narrow and expensive.

Bi-encoder (our embedder): query and doc are embedded SEPARATELY, compared by
dot product. Doc vectors are precomputed, so it scales to millions, but the
model never sees query and doc together, so it misses fine interactions
("books NOT about war").

Cross-encoder (reranker): feeds "[query] [SEP] [doc]" through one transformer
and outputs a relevance score. Full attention between query and doc tokens =
much more accurate, but O(candidates) forward passes per query, so it can't
search the corpus. Hence the two-stage funnel: hybrid retrieves ~50-100,
reranker picks the top 5.

Two backends behind one interface:
- CohereReranker: hosted Rerank API (rerank-v3.5). Needs COHERE_API_KEY.
- LocalCrossEncoderReranker: ms-marco-MiniLM-L-6-v2 on CPU, free, offline.
"auto" picks Cohere when a key is present, so CI and laptops without a key
still run the identical pipeline.
"""

import os
import time
from functools import lru_cache
from typing import Protocol


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]:
        """Return [(index into docs, relevance)] best first, length <= top_n."""
        ...


class CohereReranker:
    def __init__(self, model: str) -> None:
        import cohere

        self.client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
        self.model = model
        self.name = f"cohere:{model}"

    def rerank(self, query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]:
        for attempt in range(4):
            try:
                resp = self.client.rerank(model=self.model, query=query, documents=docs, top_n=top_n)
                return [(r.index, r.relevance_score) for r in resp.results]
            except Exception as e:  # trial keys are rate-limited to ~10 calls/min
                if "429" not in str(e) or attempt == 3:
                    raise
                time.sleep(2 ** attempt * 5)
        raise RuntimeError("unreachable")


class LocalCrossEncoderReranker:
    def __init__(self, model: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model)
        self.name = f"local:{model}"

    def rerank(self, query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]:
        scores = self.model.predict([(query, d) for d in docs], show_progress_bar=False)
        order = sorted(range(len(docs)), key=lambda i: -scores[i])[:top_n]
        return [(i, float(scores[i])) for i in order]


@lru_cache(maxsize=4)
def get_reranker(kind: str, cohere_model: str, local_model: str) -> Reranker:
    if kind == "cohere" or (kind == "auto" and os.getenv("COHERE_API_KEY")):
        return CohereReranker(cohere_model)
    return LocalCrossEncoderReranker(local_model)
