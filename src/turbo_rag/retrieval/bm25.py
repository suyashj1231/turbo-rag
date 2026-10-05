"""Sparse retrieval: Okapi BM25, from scratch.

Why have keyword search next to embeddings? Dense vectors are great at
*meaning* ("a story about grief" ~ "a novel about losing a parent") and bad at
*exact tokens*: names, ISBNs, rare words, typos-free proper nouns. MiniLM has
never seen most author names as meaningful units. BM25 nails those. Hybrid =
both strengths.

BM25 score of document D for query Q:

    sum over terms t in Q of
        IDF(t) * tf(t,D) * (k1 + 1) / (tf(t,D) + k1 * (1 - b + b * |D| / avgdl))

    IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

- tf saturation (k1 ~ 1.2-2.0): the 10th "dragon" adds far less than the 1st.
- length normalization (b ~ 0.75): a long description isn't relevant just
  because it contains more words.
- IDF: rare terms ("Tolkien") matter more than common ones ("book").

Implementation: an inverted index (term -> postings of (doc, tf)), so a query
only touches docs containing its terms, not all N docs.
"""

import math
import re
from collections import Counter, defaultdict

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have he her his i in is it its me my of on or "
    "she that the their them they this to was were what when where which who whom why "
    "will with you your about any book books".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: np.ndarray = np.zeros(0)
        self.idf: dict[str, float] = {}

    def add(self, texts: list[str]) -> None:
        lengths = []
        for doc_id, text in enumerate(texts):
            counts = Counter(tokenize(text))
            lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                self.postings[term].append((doc_id, tf))
        self.doc_len = np.array(lengths, dtype=np.float32)
        n = len(texts)
        self.idf = {
            term: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        """BM25 score for every doc (0 for docs sharing no terms)."""
        out = np.zeros(len(self.doc_len), dtype=np.float32)
        avgdl = self.doc_len.mean() if len(self.doc_len) else 1.0
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            ids, tfs = zip(*self.postings[term])
            ids = np.fromiter(ids, dtype=np.int64)
            tfs = np.fromiter(tfs, dtype=np.float32)
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[ids] / avgdl)
            out[ids] += self.idf[term] * tfs * (self.k1 + 1) / (tfs + norm)
        return out

    def search(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = self.scores(query)
        nonzero = np.flatnonzero(scores)
        if len(nonzero) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        k = min(k, len(nonzero))
        top = nonzero[np.argpartition(-scores[nonzero], k - 1)[:k]]
        top = top[np.argsort(-scores[top])]
        return top, scores[top]
