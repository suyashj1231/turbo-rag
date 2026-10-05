"""Central config. Plumbing — provided for you, tweak values as you go."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")  # makes GEMINI_API_KEY / COHERE_API_KEY visible
DATA_DIR = REPO_ROOT / "data"


@dataclass
class Config:
    # paths
    raw_dir: Path = DATA_DIR / "raw"              # books.csv (Kaggle 7k books) and/or PDFs
    processed_dir: Path = DATA_DIR / "processed"  # chunks.jsonl, embeddings.npy, bm25
    eval_dir: Path = DATA_DIR / "eval"            # golden.jsonl + dev.jsonl (committed!)
    cache_dir: Path = DATA_DIR / "cache"          # cached LLM calls
    sessions_db: Path = DATA_DIR / "sessions.db"  # multi-turn chat history (DynamoDB stand-in)

    # chunking (Stage 1: fixed-size; revisit in Stage 5)
    chunk_size: int = 800      # characters
    chunk_overlap: int = 150   # characters

    # models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast, free
    # free-tier Gemini quotas are per model per DAY (as low as 20), so both are env-overridable
    llm_model: str = os.getenv("RAG_LLM_MODEL", "gemini-2.5-flash")        # answering
    judge_model: str = os.getenv("RAG_JUDGE_MODEL", "gemini-3.6-flash")    # judge + Ragas: stronger model
    cohere_rerank_model: str = "rerank-v3.5"
    local_rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # retrieval
    top_k: int = 5             # chunks that reach the LLM
    candidate_k: int = 50      # per-retriever candidates fed into fusion + rerank
    rrf_k: int = 60            # RRF damping constant (the paper's default)
    # "dense" | "bm25" | "hybrid" | "hybrid_rerank" — the eval ablates over these
    retrieval_mode: str = os.getenv("RAG_RETRIEVAL_MODE", "hybrid_rerank")
    # "auto" uses Cohere when COHERE_API_KEY is set, else the local cross-encoder
    reranker: str = os.getenv("RAG_RERANKER", "auto")

    # multi-turn
    history_turns: int = 6     # past messages used for query condensation

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.eval_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
