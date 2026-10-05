"""Wires the pieces together. Read it top to bottom — it IS the RAG architecture:

    offline (`rag ingest`):
        books.csv / PDFs -> ingest -> chunk -> embed -> data/processed/
    online (`rag ask`, `rag chat`, POST /chat):
        message + session history -> condense -> hybrid retrieve (dense + BM25,
        RRF) -> rerank -> cited generation -> validate -> persist turn
"""

import json
from dataclasses import asdict
from functools import lru_cache

import numpy as np

from .chunking import Chunk, chunk_corpus
from .config import CONFIG
from .embed import embed_texts
from .generate import Answer, answer, condense_question
from .index.flat import FlatIndex
from .ingest import load_corpus
from .retrieval.bm25 import BM25Index
from .retrieval.rerank import get_reranker
from .retrieval.retriever import Hit, Retriever
from .sessions import SessionStore


def run_ingest() -> None:
    """Offline path: raw corpus -> chunks.jsonl + embeddings.npy on disk."""
    CONFIG.ensure_dirs()
    docs = load_corpus(CONFIG.raw_dir)
    chunks = chunk_corpus(docs, CONFIG.chunk_size, CONFIG.chunk_overlap)
    vectors = embed_texts([c.text for c in chunks], CONFIG.embedding_model)

    with open(CONFIG.processed_dir / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")
    np.save(CONFIG.processed_dir / "embeddings.npy", vectors)
    print(f"ingested {len(docs)} docs -> {len(chunks)} chunks, dim={vectors.shape[1]}")


def load_chunks() -> list[Chunk]:
    with open(CONFIG.processed_dir / "chunks.jsonl") as f:
        return [Chunk(**json.loads(line)) for line in f]


@lru_cache(maxsize=1)
def build_retriever(use_reranker: bool = True) -> Retriever:
    """Load artifacts and build both indexes. Cached: the API builds it once
    at startup (the "cold start"), then every request reuses it."""
    chunks = load_chunks()
    dense = FlatIndex()
    dense.add(np.load(CONFIG.processed_dir / "embeddings.npy"))
    bm25 = BM25Index()
    bm25.add([c.text for c in chunks])  # ~1s for 7k books; cheaper than pickling
    reranker = (
        get_reranker(CONFIG.reranker, CONFIG.cohere_rerank_model, CONFIG.local_rerank_model)
        if use_reranker else None
    )
    return Retriever(chunks, dense, bm25, CONFIG.embedding_model, reranker, CONFIG.candidate_k, CONFIG.rrf_k)


def rag_answer(question: str, mode: str | None = None) -> tuple[Answer, list[Hit]]:
    """Single-turn: retrieve + cited generation."""
    mode = mode or CONFIG.retrieval_mode
    hits = build_retriever(use_reranker=mode == "hybrid_rerank").retrieve(question, CONFIG.top_k, mode)
    return answer(question, [h.chunk for h in hits], CONFIG.llm_model), hits


class ChatService:
    """Multi-turn chat: the unit the HTTP API (or a Lambda handler) calls."""

    def __init__(self, store: SessionStore) -> None:
        self.store = store

    def chat(self, session_id: str, message: str) -> dict:
        history = self.store.history(session_id, last_n=CONFIG.history_turns)
        query = condense_question(message, history, CONFIG.llm_model)
        ans, hits = rag_answer(query)
        sources = [
            {"n": n, "chunk_id": h.chunk.chunk_id, "title": h.chunk.metadata.get("title", h.chunk.doc_id),
             "authors": h.chunk.metadata.get("authors"), "score": round(h.score, 4)}
            for n, h in enumerate(hits, start=1)
        ]
        self.store.append(session_id, "user", message)
        self.store.append(session_id, "assistant", ans.text, {
            "standalone_query": query,
            "citations": [asdict(c) for c in ans.citations],
            "grounded": ans.grounded,
        })
        return {
            "session_id": session_id,
            "answer": ans.text,
            "standalone_query": query,
            "citations": [asdict(c) for c in ans.citations],
            "sources": sources,
            "grounded": ans.grounded,
            "refused": ans.refused,
        }


def print_answer(ans: Answer, hits: list[Hit]) -> None:
    for n, h in enumerate(hits, start=1):
        print(f"  [{n}] {h.score:7.3f}  {h.chunk.chunk_id}  {h.chunk.metadata.get('title', '')}")
    print()
    print(ans.text)
    if ans.attempts > 1:
        print(f"\n(citations repaired on retry; grounded={ans.grounded})")


def run_ask(question: str, mode: str | None = None) -> None:
    ans, hits = rag_answer(question, mode)
    print_answer(ans, hits)


def run_chat() -> None:
    """Interactive multi-turn REPL backed by the same session store as the API."""
    store = SessionStore(CONFIG.sessions_db)
    service = ChatService(store)
    session_id = store.new_session_id()
    print(f"session {session_id} — ask about books, Ctrl-D to quit")
    while True:
        try:
            message = input("\nyou> ").strip()
        except EOFError:
            return
        if not message:
            continue
        result = service.chat(session_id, message)
        if result["standalone_query"] != message:
            print(f"  (searching for: {result['standalone_query']})")
        print(f"\nbot> {result['answer']}")
        for c in result["citations"]:
            print(f"     [{c['n']}] {c['title']}")
