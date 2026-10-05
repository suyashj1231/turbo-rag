# turbo-rag

A RAG system built from scratch (no LangChain or LlamaIndex) over the
[Kaggle 7k Books](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) dataset:

- **Hybrid retrieval**: BM25 written from scratch plus dense vectors (MiniLM), fused with Reciprocal Rank Fusion
- **Cross-encoder reranking**: Cohere Rerank, or a local ms-marco cross-encoder when no key is set
- **Citation-enforced answers**: numbered sources, a citation validator, one repair retry, then it fails closed to "I don't know"
- **Multi-turn chat**: questions rewritten with conversation history, sessions persisted in SQLite (same key design as DynamoDB)
- **HTTP API**: FastAPI (stands in for API Gateway + Lambda)
- **CI-gated evaluation**: a retrieval-metrics gate plus a Ragas gate (faithfulness, answer relevancy, context precision)

Next on the roadmap: a from-scratch **TurboQuant** quantized index (see [PLAN.md](PLAN.md)).
Interview study guide: [docs/INTERVIEW.md](docs/INTERVIEW.md).

## Setup

```bash
uv sync                    # Python deps (dev + eval groups included)
cp .env.example .env       # GEMINI_API_KEY (required), COHERE_API_KEY (optional)

mkdir -p data/raw && curl -sL -o /tmp/books.zip \
  https://www.kaggle.com/api/v1/datasets/download/dylanjcastillo/7k-books-with-metadata \
  && unzip -o /tmp/books.zip -d data/raw
uv run rag ingest          # 6,810 books -> 8,047 chunks -> embeddings (~1 min on a laptop)
```

## Usage

```bash
uv run rag ask "that sci-fi book where a modern town gets sent back to the 1600s"
uv run rag ask "books by Marilynne Robinson" --mode bm25   # dense | bm25 | hybrid | hybrid_rerank
uv run rag chat                                            # multi-turn REPL
uv run rag serve                                           # API on :8000

curl -X POST localhost:8000/chat -H 'content-type: application/json' \
     -d '{"message": "a funny novel about a simple soldier in WW1"}'
# -> {session_id, answer, citations, sources, standalone_query, grounded, refused}
curl -X POST localhost:8000/chat -H 'content-type: application/json' \
     -d '{"session_id": "<id>", "message": "which American novel did it inspire?"}'

uv run pytest                          # 44 unit tests, no keys needed
uv run rag validate-golden             # check golden labels against the corpus
uv run rag eval --no-llm               # retrieval ablation on the golden set (free, deterministic)
uv run rag eval                        # + multi-turn, LLM-as-judge grading, refusal check
uv run rag eval --set dev              # same, on the dev set you iterate on
uv run rag ragas --gate                # Ragas metrics on golden, exit 1 below thresholds
```

## Architecture

```
offline   books.csv ─► ingest ─► chunk (+title/author header) ─► embed (MiniLM, 384-d)
                                                     └─► data/processed/{chunks.jsonl, embeddings.npy}

online    POST /chat {session_id, message}
            │
            ├─ SessionStore.history(session_id)           SQLite (DynamoDB-shaped: PK session, SK turn)
            ├─ condense_question(message, history)         "who wrote it?" ─► "who wrote Gilead?"
            ├─ dense top-50 (FlatIndex) ┐
            │                           ├─ RRF fusion ─► rerank top-50 ─► top-5
            ├─ BM25 top-50 (from scratch)┘               (Cohere | local cross-encoder)
            ├─ generate with numbered sources ─► validate citations ─► repair once ─► fail closed
            └─ SessionStore.append(user, assistant + citations)

CI        pytest ─► retrieval gate (hit/MRR/recall vs thresholds.json) ─► Ragas gate (needs secret)
```

## Eval sets

| set | file | size | how it's made | used for |
|---|---|---|---|---|
| **golden** | `data/eval/golden.jsonl` | 35 | hand-written, checked against the corpus, frozen | reported numbers + CI gate |
| dev | `data/eval/dev.jsonl` | 55 | 40 LLM-generated + 15 hand-written | iterating and tuning |

Golden labels name **books (every edition) plus an evidence quote**, not chunk
ids, so they still hold if chunking changes. `rag validate-golden` checks every
label: the doc exists, the evidence appears in it, no unlabeled editions, no
overlap with the dev set. Categories: vague, lookup, metadata, author_list,
multi_book, multi_turn, unanswerable (including near-misses: real books that
aren't in the corpus).

## Results: golden set (k=5, local cross-encoder reranker)

24 single-turn answerable items (the 4 multi-turn items need the LLM, and the 7 unanswerable items are scored by refusal):

| mode            | hit@5 | recall@5 | MRR   | vague MRR | multi_book recall | p50 latency |
|-----------------|------:|---------:|------:|----------:|------------------:|------------:|
| dense           | 0.958 | 0.805 | 0.821 | 0.650 | 0.533 | 17 ms |
| bm25            | 0.917 | 0.761 | 0.819 | 0.667 | 0.467 | 0.4 ms |
| hybrid (RRF)    | 0.917 | 0.822 | 0.833 | 0.562 | **0.700** | 7 ms |
| hybrid + rerank | **1.000** | **0.912** | **0.958** | **0.938** | 0.533 | 123 ms |

Multi-turn (hybrid + rerank): follow-ups retrieved with the raw message hit
2/4; with LLM query condensation, 4/4.

Small-n caveat: one question is about 4% of the "all" row. Treat differences
under about 0.05 as noise.

## Results: dev set (55 questions)

| mode            | hit@5 | recall@5 | MRR   | p50 latency |
|-----------------|------:|---------:|------:|------------:|
| dense           | 0.855 | 0.768    | 0.779 | 14 ms       |
| bm25            | 0.945 | 0.820    | 0.918 | 0.5 ms      |
| hybrid (RRF)    | 0.945 | 0.864    | 0.885 | 9 ms        |
| hybrid + rerank | **0.982** | **0.885** | **0.964** | 145 ms |

On the 15 hand-written **vague** questions ("a sci-fi book where a mining town
gets thrown back in time"), the hardest split:

| mode            | hit@5 | MRR   |
|-----------------|------:|------:|
| dense           | 0.667 | 0.469 |
| bm25            | 0.800 | 0.700 |
| hybrid          | 0.800 | 0.647 |
| hybrid + rerank | **0.933** | **0.867** |

Ragas smoke test (2 questions; full run pending API quota): faithfulness 1.00,
answer relevancy 0.92, context precision 1.00.

## Design decisions

- **Every chunk gets a title/author header.** A chunk from the middle of a long description doesn't otherwise say which book it belongs to.
- **The first eval set was saturated.** LLM-generated questions copied rare names from the descriptions, so BM25 scored MRR 1.000. The hand-written "vague" split fixed that and is where reranking shows its value.
- **Citation validation fails closed.** An uncited answer is replaced by a refusal. End-to-end testing found a splitter bug: initials ("K. Sadlon") were being treated as sentence ends. There is now a regression test for it.
- **Golden vs dev split.** Tune on dev, report and gate on golden. Tuning until the golden numbers go up turns golden into another dev set.
- **The CI gate uses deterministic retrieval metrics first.** Ragas comes second, because LLM-graded metrics are noisy and cost quota.
