# turbo-rag

From-scratch RAG pipeline whose vector index is a from-scratch implementation of
**TurboQuant** ([arXiv 2504.19874](https://arxiv.org/abs/2504.19874)) online
vector quantization. No LangChain, no LlamaIndex — numpy, one embedding model,
one LLM API. See [PLAN.md](PLAN.md) for the full roadmap.

## Setup

```bash
uv sync --extra dev        # creates .venv with Python 3.12 and all deps
cp .env.example .env       # add your GEMINI_API_KEY (free: aistudio.google.com/apikey)
```

## Usage

```bash
# drop PDFs into data/raw/, then:
uv run rag ingest          # PDFs -> chunks + embeddings (offline)
uv run rag ask "..."       # one question through the pipeline
uv run rag eval            # the scoreboard (Stage 2)
uv run pytest              # the executable spec — make these pass
```

## Architecture

```
offline:  data/raw/*.pdf -> ingest -> chunking -> embed -> data/processed/
online:   question -> embed -> index.search (top-k) -> generate (LLM) -> answer
eval:     data/eval/questions.jsonl -> harness -> retrieval metrics + LLM judge
```

Every index implements `index/base.py::VectorIndex`, so quantized indexes swap
in without touching the pipeline.

## Benchmarks

(Stage 4 — tables and plots land here, with design decisions and where results
agreed/disagreed with the paper.)
