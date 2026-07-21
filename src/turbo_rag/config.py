"""Central config. Plumbing — provided for you, tweak values as you go."""

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")  # makes GEMINI_API_KEY visible to genai.Client()
DATA_DIR = REPO_ROOT / "data"


@dataclass
class Config:
    # paths
    raw_dir: Path = DATA_DIR / "raw"              # put your PDFs here
    processed_dir: Path = DATA_DIR / "processed"  # chunks.jsonl, embeddings.npy
    eval_dir: Path = DATA_DIR / "eval"            # questions.jsonl (committed!)
    cache_dir: Path = DATA_DIR / "cache"          # cached LLM calls

    # chunking (Stage 1: fixed-size; revisit in Stage 5)
    chunk_size: int = 800      # characters
    chunk_overlap: int = 150   # characters

    # models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast, free
    llm_model: str = "gemini-2.5-flash"       # answering
    judge_model: str = "gemini-2.5-pro"       # judging — always judge with a stronger model

    # retrieval
    top_k: int = 5

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.eval_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
