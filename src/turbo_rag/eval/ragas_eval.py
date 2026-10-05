"""Ragas: LLM-graded RAG quality, gated in CI.

Three metrics, each isolating a different failure:

- faithfulness       (generator)  Split the answer into atomic claims; an LLM
                                  checks each against the retrieved contexts.
                                  score = supported claims / total claims.
                                  Low -> hallucination.
- answer_relevancy   (generator)  Generate N questions FROM the answer, embed
                                  them, mean cosine to the real question.
                                  Low -> off-topic or evasive answers.
                                  ("I don't know" scores ~0 by design, so a
                                  system can't game faithfulness by refusing.)
- context_precision  (retriever)  For each retrieved chunk, is it useful for
                                  the reference answer? Averaged precision@i,
                                  so relevant chunks ranked HIGH score more.
                                  Low -> retriever/reranker noise.

Judge = Gemini through its OpenAI-compatible endpoint (Ragas' native Gemini
adapter has a known instructor safety-settings bug). Embeddings for
answer_relevancy = our local MiniLM — no API quota spent on embeddings.
All judge calls go through Ragas' disk cache, so re-runs cost nothing.

--gate: compare means to data/eval/thresholds.json["ragas"]; exit 1 if any
metric is below its floor. That's the CI contract: a PR that makes answers
less faithful can't merge.
"""

import asyncio
import json
import os
import statistics

from ..config import CONFIG
from ..generate import answer
from ..pipeline import build_retriever
from .golden import load_set
from .harness import search_query

GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _metrics():
    from openai import AsyncOpenAI
    from ragas.cache import DiskCacheBackend
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness

    client = AsyncOpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=GEMINI_OPENAI_BASE, max_retries=8)
    llm = llm_factory(CONFIG.judge_model, client=client,
                      cache=DiskCacheBackend(str(CONFIG.cache_dir / "ragas")), temperature=0.0)
    embeddings = HuggingFaceEmbeddings(model=CONFIG.embedding_model)
    return Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=embeddings), ContextPrecision(llm=llm)


async def _score_all(rows: list[dict], concurrency: int = 4) -> list[dict]:
    faithfulness, relevancy, precision = _metrics()
    sem = asyncio.Semaphore(concurrency)  # stay under the judge's rate limit

    async def score(row: dict) -> dict:
        async with sem:
            out = {"qid": row["qid"], "refused": row["refused"]}
            out["context_precision"] = (await precision.ascore(
                user_input=row["question"], reference=row["reference"], retrieved_contexts=row["contexts"])).value
            out["answer_relevancy"] = (await relevancy.ascore(
                user_input=row["question"], response=row["answer"])).value
            # faithfulness of a refusal is undefined (no claims) -> skip, but
            # refusals still drag answer_relevancy down, so they aren't free
            if not row["refused"]:
                out["faithfulness"] = (await faithfulness.ascore(
                    user_input=row["question"], response=row["answer"], retrieved_contexts=row["contexts"])).value
            return out

    return await asyncio.gather(*(score(r) for r in rows))


def run_ragas(eval_set: str = "golden", limit: int | None = None, gate: bool = False) -> int:
    # unanswerable items have no reference context to score — refusals are
    # measured by `rag eval` instead
    questions = [q for q in load_set(eval_set) if q.get("category") != "unanswerable"][:limit]
    if not questions:
        print(f"no answerable items in data/eval/{eval_set}.jsonl")
        return 1

    retriever = build_retriever(use_reranker=CONFIG.retrieval_mode == "hybrid_rerank")
    rows = []
    for q in questions:  # generation is sync + cached; scoring is async
        query = search_query(q, use_llm=True)  # multi-turn -> condensed standalone query
        hits = retriever.retrieve(query, CONFIG.top_k, CONFIG.retrieval_mode)
        ans = answer(query, [h.chunk for h in hits], CONFIG.llm_model)
        rows.append({"qid": q["qid"], "question": query, "reference": q["reference_answer"],
                     "answer": ans.text, "contexts": ans.contexts, "refused": ans.refused})

    scored = asyncio.run(_score_all(rows))
    means = {}
    for m in ("faithfulness", "answer_relevancy", "context_precision"):
        values = [s[m] for s in scored if s.get(m) is not None]
        means[m] = statistics.mean(values) if values else 0.0
    refusals = sum(s["refused"] for s in scored)

    print(f"ragas  set={eval_set}  n={len(scored)}  mode={CONFIG.retrieval_mode}  judge={CONFIG.judge_model}")
    for m, v in means.items():
        print(f"  {m:<18} {v:.3f}")
    print(f"  refusals           {refusals}/{len(scored)}")
    report = CONFIG.eval_dir / "ragas_report.json"
    report.write_text(json.dumps({"means": means, "per_question": scored}, indent=1))

    if gate:
        thresholds = json.loads((CONFIG.eval_dir / "thresholds.json").read_text())[eval_set]["ragas"]
        failures = [f"{m} {means[m]:.3f} < {t}" for m, t in thresholds.items() if means[m] < t]
        print("GATE " + ("FAILED: " + "; ".join(failures) if failures else "passed"))
        return 1 if failures else 0
    return 0
