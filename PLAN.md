# TurboQuant RAG — Project Plan

## Goal
Build a RAG system from scratch (no LangChain/LlamaIndex) where the vector index
uses a from-scratch implementation of **TurboQuant** (arXiv 2504.19874, Google + NYU,
ICLR 2026) for vector quantization. Deliverable: working system + benchmark report
with real from→to numbers for the resume.

Target resume bullet shape:
> "Implemented an ICLR 2026 online vector-quantization algorithm (TurboQuant) in
> numpy and integrated it as the retrieval index in a from-scratch RAG pipeline;
> achieved 32x index compression with X% recall@10 and no measurable drop in
> answer accuracy on a 50-question eval set."

## Context (who/why)
- Suyash Jain, UC Irvine CS/Business, new-grad SWE job hunt. Already has a basic
  "RAG-Powered Document Assistant" on the resume — this project must go deeper
  and replace it.
- Mentor guidance: the differentiators are (1) implementing a real paper,
  (2) an eval harness with measured numbers, (3) explaining every design choice.
- Learning goal: understand RAG types hands-on (naive → hybrid → reranking →
  agentic) AND learn to read/implement research papers.

## Key papers
- TurboQuant: https://arxiv.org/abs/2504.19874
  - Algorithm: (1) random rotation of vectors → concentrated Beta distribution on
    coordinates, near-independent in high dim; (2) optimal scalar quantizer per
    coordinate; (3) for inner-product search: MSE quantizer + 1-bit QJL transform
    on the residual → unbiased inner-product estimates.
  - Data-oblivious / online: no training pass needed (unlike product quantization).
  - Claims: near-optimal distortion at all bit-widths; beats PQ on recall with much
    faster indexing.
- Fast-TurboQuant (follow-up, multiplier-free): https://arxiv.org/abs/2606.21448
  (optional comparison later)

## Rules
- Build from scratch: Python + numpy + one embedding model + one LLM API.
  No LangChain/LlamaIndex. faiss allowed ONLY as a comparison baseline in Stage 4.
- Don't copy a GitHub TurboQuant implementation; write it yourself, peek only when
  stuck.
- Every improvement must move a measured number. No evals = not done.
- Corpus: pick something real and messy (PDFs with tables), that Suyash cares about.

## Stages

### Stage 1 — Baseline RAG (float32 brute force)
- Ingest corpus → chunk (start fixed-size) → embed (sentence-transformers locally,
  or an API) → store as float32 numpy matrix.
- Retrieval: exact cosine / inner-product top-k. Prompt assembly → LLM answer.
- Working end-to-end CLI. This is ground truth for all later comparisons.

### Stage 2 — Eval harness (BEFORE any optimization)
- 30–50 question/answer pairs over the corpus.
- Retrieval metrics: hit rate, recall@k, MRR (is the gold chunk retrieved?).
- Answer metric: LLM-as-judge correctness.
- One command runs the whole eval and prints a scoreboard. Cache LLM calls.

### Stage 3 — Implement TurboQuant (numpy)
1. MSE quantizer first: random rotation + per-coordinate optimal scalar quantizer.
   Verify distortion on synthetic Gaussian vectors against the paper's reported
   rates before touching real data.
2. Add the inner-product correction (1-bit QJL on the residual).
3. Support 1, 2, 4 bits per coordinate; pack bits for memory measurement.
- Expect the paper to need 2–3 careful reads. That's normal.

### Stage 4 — Swap into retrieval path + benchmark
At 1/2/4 bits, measure vs. Stage 1 baseline:
- recall@k vs exact search
- memory footprint (expect ~32x at 1 bit)
- query latency
- end-to-end RAG answer accuracy (the novel question: how much quantization can
  retrieval tolerate before answers degrade?)
Compare against: float32 brute force, naive binary quantization, faiss product
quantization. Verify or refute the paper's claims on this corpus.

### Stage 5 — Advanced RAG (one technique at a time, each measured)
- Chunking strategies (fixed vs semantic vs structure-aware) — often biggest win
- Hybrid search: BM25 (rank-bm25) + vector, reciprocal rank fusion
- Cross-encoder reranking (retrieve 50 cheap → rerank to 5)
- Query transformation: rewriting, HyDE, multi-query
- Agentic RAG: model decides when/what to retrieve, multi-hop, retry on failure

### Stage 6 — Write-up
- README with architecture, benchmark tables/plots, design decisions, and where
  results agreed/disagreed with the paper.

## Pacing
- Week 1: Stages 1–2.
- Weeks 2–3: Stage 3 (the hard one) + Stage 4.
- Then one Stage-5 technique per sitting.

## First task for the new session
Scaffold the repo (pyproject, src layout, tests), then Stage 1.
