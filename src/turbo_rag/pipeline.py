"""Wires the pieces together. Skeleton provided; it will run once your
modules are implemented. Read it top to bottom — it IS the RAG architecture:

    ingest -> chunk -> embed -> index          (offline, `rag ingest`)
    query -> embed -> search -> prompt -> LLM  (online,  `rag ask`)
"""

import json

import numpy as np

from .chunking import Chunk, chunk_corpus
from .config import CONFIG
from .embed import embed_query, embed_texts
from .generate import answer
from .index.flat import FlatIndex
from .ingest import load_corpus


def run_ingest() -> None:
    """Offline path: PDFs -> chunks.jsonl + embeddings.npy on disk."""
    CONFIG.ensure_dirs()
    docs = load_corpus(CONFIG.raw_dir)
    chunks = chunk_corpus(docs, CONFIG.chunk_size, CONFIG.chunk_overlap)
    vectors = embed_texts([c.text for c in chunks], CONFIG.embedding_model)

    with open(CONFIG.processed_dir / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__) + "\n")
    np.save(CONFIG.processed_dir / "embeddings.npy", vectors)
    print(f"ingested {len(docs)} docs -> {len(chunks)} chunks, dim={vectors.shape[1]}")


def load_chunks() -> list[Chunk]:
    with open(CONFIG.processed_dir / "chunks.jsonl") as f:
        return [Chunk(**json.loads(line)) for line in f]


def build_index() -> tuple[FlatIndex, list[Chunk]]:
    chunks = load_chunks()
    vectors = np.load(CONFIG.processed_dir / "embeddings.npy")
    index = FlatIndex()
    index.add(vectors)
    return index, chunks


def run_ask(question: str) -> str:
    """Online path: one question -> answer, printing retrieved sources."""
    index, chunks = build_index()
    q = embed_query(question, CONFIG.embedding_model)
    ids, scores = index.search(q, CONFIG.top_k)
    hits = [chunks[i] for i in ids]
    for c, s in zip(hits, scores):
        print(f"  [{s:.3f}] {c.chunk_id} ({c.doc_id} p.{c.page_start})")
    return answer(question, hits, CONFIG.llm_model)
