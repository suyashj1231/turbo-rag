# Interview guide: how this RAG system works, and how to talk about it

Read this next to the code. Each section covers what the component is, why
it exists, where it lives, the numbers it produced, and the questions an
interviewer is likely to ask, with answers you can give in about 30 seconds.

---

## 0. The 60-second pitch

> "I built a RAG system from scratch, no LangChain, over about 7,000 book
> descriptions. Retrieval is hybrid: BM25, which I implemented myself, plus
> dense MiniLM embeddings, fused with reciprocal rank fusion, then a
> cross-encoder reranker on the top 50. Generation is citation-enforced: every
> sentence must cite a numbered source, a validator checks that, retries once
> with feedback, and otherwise fails closed to 'I don't know'. It's multi-turn:
> follow-ups get rewritten into standalone queries using session history
> stored server-side. It's served over a FastAPI endpoint, and CI gates every
> PR on retrieval metrics and Ragas faithfulness, relevancy and context
> precision. The headline number: on vague, real-user-style questions,
> reranking took MRR from 0.65 to 0.87."

Practice saying the last sentence. **A number with a cause** is what makes an
interviewer believe you built it.

---

## 1. Ingestion and chunking (`ingest.py`, `chunking.py`)

**What:** CSV row → `Document(doc_id=isbn13, pages=[description], metadata={title, authors, ...})`
→ fixed 800-character chunks with 150 characters of overlap → each chunk is prefixed with
`Title: / Authors: / Categories: / Published:`.

**Why the header:** 980 of the 6,810 descriptions are longer than 800 characters, so they
split. Without the header, chunk `:0001` of a book doesn't contain the book's
name, so neither BM25 nor the embedding can match "books by Robinson" to it.
This is a cheap version of Anthropic's *contextual retrieval* (they use an LLM
to write the context; here the metadata already holds it).

**Why stable ids (`isbn:0003`):** eval gold labels reference chunk ids. A
`uuid4()` would break every label on re-ingest.

**Why keep the books with no description (262 of them):** the header alone
still answers "who wrote X?" and "when was X published?".

**Q: How do you choose chunk size?**
A: It's a trade-off between precision and context. Small chunks embed one idea
precisely but lose surrounding context; large chunks dilute the vector. Start
with a baseline (fixed size plus overlap), then let the eval decide. Here the
median description is about 240 characters, so most books are a single chunk
and chunking matters less than it would for PDFs. For long documents I'd try
structure-aware splitting (headings, paragraphs) and measure recall@k for each
strategy.

**Q: Why overlap?** So a fact that straddles a boundary appears whole in at
least one chunk. The cost is about 20% more vectors.

---

## 2. Dense retrieval (`embed.py`, `index/flat.py`)

**What:** `all-MiniLM-L6-v2` (384 dimensions, local, free). Vectors are
normalized, so cosine similarity equals the inner product, so search is one
matrix-vector product (`vectors @ q`) plus `argpartition` for the top k.

**Numbers:** 8,047 × 384 × 4 bytes ≈ 12 MB, and search takes ~1 ms (the
measured 14 ms is mostly embedding the query).

**Q: Why brute force and not a vector DB / HNSW?**
A: Exact search over 8k vectors takes about a millisecond. An approximate (ANN)
index only pays off around 10⁵–10⁶ vectors, and it costs you recall plus
another moving part. The `VectorIndex` protocol (`index/base.py`) means I can
swap in FAISS, HNSW or a quantized index without touching the pipeline. That's
the next stage of this project (TurboQuant).

**Q: Bi-encoder vs cross-encoder?** See section 4.

**Q: Why did dense do worst on our eval?** MiniLM is small and was trained
mostly on general web text. Book descriptions are full of proper nouns it
doesn't represent well, and our "content" questions reused those nouns, which
favors keyword matching. On paraphrased "vague" questions dense still trailed,
at hit@5 0.67. A stronger embedder (bge, e5, gemini-embedding) is the obvious
next experiment, and the eval would tell you whether it's worth the extra latency.

---

## 3. BM25 from scratch (`retrieval/bm25.py`) and hybrid fusion (`retrieval/fusion.py`)

**BM25 in one breath:** for each query term, IDF (rare terms matter more) ×
saturated term frequency (the 10th "dragon" adds less than the 1st, controlled by k1) ×
length normalization (long documents don't win just by being long, controlled by b).
It's implemented with an **inverted index** (term → postings), so a query only
touches documents that contain its terms.

**Why hybrid:** dense captures *meaning*, BM25 captures *exact tokens* (names,
ISBNs, rare words). Our table shows each one winning a different split.

**RRF:** `score(d) = Σ 1/(60 + rank_r(d))`. It uses ranks, not scores, because BM25
scores (0–30+) and cosine scores (−1 to 1) aren't on comparable scales, and
min-max normalization is fragile to outliers. A document ranked well by **both**
retrievers beats one ranked #1 by just one of them.

**Q: Hybrid didn't beat BM25 on MRR (0.885 vs 0.918). Why ship it?**
Good catch, and say it out loud in the interview. RRF averages the two lists, so
when one retriever is much better on a given question, fusion dilutes it. But
hybrid has the best **recall@5** (0.864) and its job here is to produce a good
*candidate pool* for the reranker, not a final ranking. Hybrid plus rerank is
best on every metric. In a system without a reranker I'd tune weighted RRF or
pick per query type.

**Q: What are k1 and b, and how would you tune them?** k1 ≈ 1.2–2 controls TF
saturation; b ≈ 0.75 controls length normalization. Grid-search them on the eval
set's MRR. With short, uniform documents like ours, b matters little.

**Q: How would this scale?** BM25: Elasticsearch/OpenSearch (or Postgres
full-text search). Dense: an ANN index. Fusion stays the same 10 lines.

---

## 4. Reranking (`retrieval/rerank.py`)

**Bi-encoder** (the embedder): query and document are encoded *separately*, so
document vectors can be precomputed and scale to millions. The model never
sees the two together.
**Cross-encoder** (the reranker): `[query] [SEP] [doc]` goes through one
transformer, with full attention across both. Much more accurate, but it needs a
forward pass per (query, document) pair, so it can only score a short list.

Hence the **funnel**: cheap retrieval of 50 candidates → expensive rerank → top 5.

**Numbers (the headline):** on vague questions, hybrid MRR 0.647 → **0.867**
with reranking, and hit@5 0.80 → 0.93. Cost: p50 latency 9 ms → 145 ms (local
CPU cross-encoder over 50 candidates).

**Two backends behind one interface:** Cohere `rerank-v3.5` (hosted) when
`COHERE_API_KEY` is set, otherwise local `ms-marco-MiniLM-L-6-v2`. Same
pipeline either way, so CI and laptops without a key still run.

**Q: How do you cut the 145 ms?** Rerank fewer candidates (look at recall@20
vs recall@50 on the eval), use a hosted reranker (Cohere is about 50–100 ms over
the network but runs on a GPU), batch requests, distill to a smaller model, or
cache by query.

**Q: When would you NOT rerank?** When latency budgets are tight and the first
stage is already good. Measure it: if rerank only moves MRR by 0.01, it isn't
worth 140 ms.

---

## 5. Citation-enforced generation (`generate.py`)

Three layers:

1. **Prompt contract:** sources are numbered `[1]..[n]`; every factual sentence
   must cite; use only the sources; refuse with an *exact* string otherwise.
2. **Validation:** parse the citations. Every `[n]` must exist (no `[7]` with 5
   sources), and every substantive sentence must have one.
3. **Repair then fail closed:** one retry that quotes the specific violations
   back to the model; if it still fails, return the refusal.

**Why fail closed:** in a citation system, an answer you can't verify is worse
than "I don't know". The refusal is visible; a hallucination is not.

**Story to tell (a real bug found end to end):** a perfectly cited answer was
being refused. The sentence splitter broke on `Zdeněk "Zenny" K. Sadlon [1]`:
it treated `K.` as the end of a sentence, saw a sentence with no citation, and
failed closed. The fix was not splitting after initials or common
abbreviations, plus a regression test. Lesson: **strict validators need their
own tests**, and fail-closed systems hide bugs as "conservative" behavior, so
watch your refusal rate.

**Other plumbing to mention:** a disk cache keyed on `sha256(model + system +
prompt)`, so eval re-runs are free and deterministic; `temperature=0`;
exponential backoff on 429/503, but fail fast on *daily* quota (sleeping
can't fix a daily cap).

**Q: Citation validation only checks that a citation *exists*, not that it's
*correct*. How do you check that?**
Exactly right, and that's what Ragas **faithfulness** measures: it splits
the answer into claims and runs an NLI-style check of each against the
contexts. Online you could do this per answer with a small NLI model, at the
cost of latency. Offline it's measured in the CI gate.

**Q: Why not structured output (JSON with claim → source ids)?** That's a
valid upgrade: it makes parsing trivial and allows per-claim checks. Plain
text with `[n]` keeps the answer readable and streamable, which matters for chat
UX.

---

## 6. Multi-turn (`generate.condense_question`, `sessions.py`, `pipeline.ChatService`)

**Problem:** retrieval only sees the current message. "Which American novel
did it inspire?" contains no book name, so retrieval returns garbage.
**Fix:** *query condensation*: an LLM rewrites the message into a standalone
query using the last N messages. In the real run, after the first turn found
*The Good Soldier Švejk*, the follow-up retrieved it again and answered
"*Catch-22* [1]".

**Persistence:** `SessionStore` (SQLite) with `PRIMARY KEY (session_id, turn)`,
the same shape as a DynamoDB table with partition key `session_id` and sort key
`turn`. Assistant messages store their citations and the standalone query,
which gives you an audit trail.

**Q: Why server-side history instead of the client sending it?** The client
can't tamper with what the assistant "said", history survives across devices,
you get an audit log, and in serverless (Lambda) deployments there's no
process memory between requests, so state *must* live outside the function.

**Q: Why rewrite instead of embedding the whole conversation?** Long history
dilutes the query vector and BM25 tokens. Rewriting keeps the retrieval query
short and focused. It costs one extra LLM call per turn (skipped on turn 1).

**Q: How do you bound history?** Only the last N turns go into condensation
(`history_turns=6`). For long sessions, summarize older turns.

**Q: Concurrency?** SQLite plus a lock is fine for one process. In DynamoDB,
use a conditional write (`attribute_not_exists(turn)`) so two concurrent
requests can't claim the same turn number.

---

## 7. API (`api.py`)

FastAPI: `POST /chat`, `GET /sessions/{id}`, `DELETE /sessions/{id}`, `GET /health`.
Models and indexes load once, in `lifespan` (the equivalent of a Lambda cold
start). Handlers are sync `def`, so FastAPI runs them on a threadpool and the
blocking embed and LLM calls don't stall the event loop.

**Q: How would you deploy this serverlessly?** Wrap `app` with Mangum → Lambda
behind API Gateway; replace `SessionStore` with a DynamoDB implementation (the
interface is 4 methods); put the corpus in S3 and build the index in a separate
ingestion job; ship the model in a container image or on EFS to shrink cold
starts; set provisioned concurrency if p99 matters.

---

## 8. Evaluation (`eval/`): the part that makes this senior-level

**Two kinds of metrics:**

| layer      | metric                 | needs LLM? | catches                         |
|------------|------------------------|------------|---------------------------------|
| retrieval  | hit@k, recall@k, MRR   | no         | wrong chunks retrieved          |
| retrieval  | Ragas context precision| yes        | noisy or badly ordered context  |
| generation | Ragas faithfulness     | yes        | hallucination                   |
| generation | Ragas answer relevancy | yes        | evasive or off-topic answers    |
| generation | LLM-as-judge vs reference | yes     | wrong answers                   |
| safety     | refusal on unanswerables | yes      | inventing answers               |

**Definitions you must know cold:**
- hit@k = 1 if any gold chunk is in the top k. recall@k = fraction of gold chunks in the top k.
  MRR = mean of 1/rank of the first gold chunk.
- Faithfulness = supported claims / total claims (claims extracted by an LLM, checked by NLI).
- Answer relevancy = generate N questions from the answer, take the mean cosine to the real question.
  "I don't know" scores about 0, which is why a system can't game faithfulness by refusing.
- Context precision = average precision over retrieved chunks judged useful,
  so it rewards *ranking* relevant chunks high.

**The eval-design story (tell this one, it's the best one):**
The first synthetic eval set was **saturated**: BM25 got MRR 1.000. The cause was
that the LLM-written questions copied rare names from the descriptions ("Jean
Godin", "Claudia Corvette"), so keyword matching trivially won and the eval
couldn't distinguish retrievers. I added a hand-written **vague** split
(no names, no copied phrases, the way real users half-remember a book). That
split is where dense retrieval fell to 0.67 hit@5 and reranking showed its real
value. Lesson: **a benchmark everything scores 100% on measures nothing.**
Always inspect your eval questions and break results out by question type.

**Golden set vs dev set (`eval/golden.py`, `data/eval/golden.jsonl`):**
- **dev** (55 questions: generated plus quick hand-written ones): iterate and tune on it freely.
- **golden** (35 questions, hand-written and checked against the corpus, frozen):
  report numbers and gate CI on it. *Never tune on it.* If you tweak the
  system until golden goes up, golden has become a dev set and your reported
  numbers are overfit. It's the same reason ML has train/val/test splits.
- **Coverage over size:** 7 categories, each a different failure mode: vague
  (paraphrase), lookup (one fact, pinned by an evidence quote), metadata
  (answered only by the chunk header), author_list (recall), multi_book (two
  books in context), multi_turn (needs history), unanswerable (including
  *near-misses*: real books like *Gone Girl* that aren't in the corpus, where
  the LLM is most tempted to answer from memory).
- **Chunking-independent labels:** gold = book ids (*every edition*) + an
  evidence quote, resolved to chunk ids at eval time. Chunk-id labels silently
  break the day you change `chunk_size`.
- **The validator** (`rag validate-golden`, first step in CI): every doc
  exists, the evidence actually appears in a gold doc, no unlabeled duplicate
  editions (otherwise a correct retrieval is scored as a miss), no overlap
  with dev. It has its own tests proving it catches each kind of bad label.

**What golden showed that dev didn't:**
- RRF fusion alone *hurt* vague questions (MRR 0.56 vs 0.65 dense, 0.67 BM25).
  When both retrievers are mediocre, averaging their ranks doesn't help. The
  reranker fixes it (0.94).
- Multi-book questions ("compare X and Y") have recall 0.53: the top 5 fills
  up with one book's editions. The fix is a feature, not tuning: query
  decomposition (split into one sub-query per book) or diversity in the
  top k (MMR / one chunk per book).
- Query condensation, measured: raw follow-ups hit 2/4, condensed 4/4.

**Q: Isn't 35 questions too small?** It's small, so say so. One question is
about 4% of a metric, so I treat differences under about 0.05 as noise and
break results down by category. The fix is growing it over time, especially
from real user queries that failed (each production bug becomes a golden item).
For a quick sense of variance, bootstrap-resample the questions and report a
confidence interval.

**Q: How do you keep a golden set honest over time?** Version it in git and
review changes to it like code. Add items, don't edit them to make the
system pass. Re-validate whenever the corpus changes. Refresh it with real
traffic.

**CI gate (`.github/workflows/ci.yml`):**
1. Unit tests (44, no keys).
2. **Retrieval gate**: download the dataset, ingest (cached on a hash of the
   ingestion code), `rag validate-golden`, then `rag eval --set golden --no-llm
   --gate` against `data/eval/thresholds.json`. Deterministic and free, so it
   never flakes.
3. **Ragas gate**: only when the `GEMINI_API_KEY` secret exists, on the golden
   set's answerable items, with the LLM cache restored between runs; uploads
   the report as an artifact.

**Q: LLM judges are noisy. How do you keep a CI gate from flaking?**
`temperature=0`; cache judge calls (unchanged answers re-score identically);
set thresholds about 0.05 below the measured score; gate on the mean of N
questions, not on individual ones; deterministic retrieval metrics run first.
Ratchet the thresholds upward as the system improves.

**Q: What does an LLM judge get wrong?** Self-preference (judge with a
different or stronger model than the generator), verbosity bias, and
position bias. Calibrate it: hand-label 30 answers and measure agreement with
the judge.

**Q: Synthetic questions: what's the bias?** They're generated from a single
chunk, so they're answerable by construction and lexically leaky. Mitigate with
style constraints, human review, and adding real user queries over time.

---

## 9. Your resume bullet: make it match what's built

The original bullet names AWS S3, Lambda, API Gateway and DynamoDB. **Don't
claim infrastructure you didn't deploy.** Interviewers drill into exactly
those words ("how did you handle Lambda cold starts with a 90 MB model?"). An
honest bullet with real numbers is stronger:

> Built a from-scratch RAG system (no LangChain) over 7k books with hybrid
> retrieval (self-implemented BM25 + dense embeddings, reciprocal rank fusion)
> and cross-encoder reranking (Cohere Rerank / local), lifting MRR on
> paraphrased queries from 0.65 to 0.87; generated citation-enforced answers
> with validation, repair and fail-closed refusal.
>
> Served multi-turn chat via a FastAPI service with server-side session
> persistence and LLM query condensation; built a CI-gated evaluation
> pipeline (retrieval hit@k/MRR + Ragas faithfulness, answer relevancy,
> context precision) that blocks regressions on every PR.

If you later deploy it to AWS (Mangum + DynamoDB SessionStore, about a day's
work), then add the AWS words back.

---

## 10. Rapid-fire questions to rehearse

- *What's RAG and why not fine-tune?* RAG puts fresh, citable knowledge in the
  prompt at query time; fine-tuning changes behavior and style, not facts, is
  expensive to update, and can't cite sources.
- *Biggest failure mode of RAG?* Retrieval misses. If the right chunk isn't in
  the context, the best LLM can't answer. That's why retrieval metrics come first.
- *Lost in the middle?* LLMs underuse context placed in the middle of long
  prompts. Keep k small (5) and put the best chunk first (the reranker does that).
- *How do you handle questions the corpus can't answer?* The prompt requires an
  exact refusal string, and the golden set's `unanswerable` items (including
  near-misses) measure the refusal rate.
- *Prompt injection via documents?* Retrieved text is untrusted. Keep
  instructions in the system prompt, delimit the sources, and don't give the
  model tools whose actions retrieved text could trigger.
- *How would you make it faster?* Cache query embeddings and answers, shrink
  `candidate_k`, use a hosted reranker, stream tokens, or use an ANN index at scale.
- *What would you do next?* Run a stronger embedder through the eval, try
  HyDE/multi-query for vague questions, and add the quantized index
  (TurboQuant) to cut memory 8–32× while measuring recall loss.
