"""Stage 2: one command, one scoreboard.

Runs on an eval set (see golden.py): `golden` (frozen, hand-verified — report
and gate on this) or `dev` (generated — iterate on this).

1. Retrieval ablation: every mode (dense / bm25 / hybrid / hybrid_rerank) on
   the same questions -> hit@k, recall@k, MRR, p50 latency, broken down by
   category, because "hybrid wins overall" hides *why*. Multi-turn items are
   retrieved with the condensed query, so they need the LLM; with --no-llm
   they're skipped. Unanswerable items have no gold and aren't scored here.
2. Answer quality for the production mode: LLM-as-judge (stronger model than
   the answerer) grades CORRECT / PARTIAL / WRONG against the reference, plus
   citation stats (repairs needed, failed-closed answers).
3. Refusals: unanswerable items should get "I don't know", not an invented
   book — including near-misses (real books that aren't in the corpus).

--gate compares the production mode's retrieval metrics against
thresholds.json[set]["retrieval"] and returns exit code 1 on regression.
Retrieval metrics are deterministic and LLM-free, so this gate never flakes.
"""

import json
import statistics
import time
from collections import defaultdict

from ..config import CONFIG
from ..generate import REFUSAL, answer, condense_question, llm_call
from ..pipeline import build_retriever
from .golden import CATEGORIES, chunks_by_doc, load_set, resolve_gold
from .metrics import hit_at_k, mrr, recall_at_k

JUDGE = """You are grading a question-answering system about books.

Question: {question}
Reference answer: {reference}
System answer: {answer}

Grade the system answer against the reference:
- CORRECT: contains the key information of the reference (wording may differ; extra correct detail is fine).
- PARTIAL: partly right, or right but hedged/incomplete.
- WRONG: incorrect, unrelated, or says it doesn't know.

Reply with exactly one word: CORRECT, PARTIAL, or WRONG."""

DEV_CATEGORIES = ("content", "vague", "lookup")


def search_query(item: dict, use_llm: bool) -> str | None:
    """The query retrieval actually sees; None if it needs the LLM and we can't."""
    if not item.get("history"):
        return item["question"]
    if not use_llm:
        return None
    return condense_question(item["question"], item["history"], CONFIG.llm_model)


def retrieval_scores(items: list[dict], mode: str, k: int, use_llm: bool) -> dict:
    retriever = build_retriever(use_reranker=mode == "hybrid_rerank")
    by_doc = chunks_by_doc(retriever.chunks)
    buckets = defaultdict(lambda: defaultdict(list))
    latencies, skipped = [], 0
    for item in items:
        if item.get("category") == "unanswerable":
            continue
        query = search_query(item, use_llm)
        if query is None:
            skipped += 1
            continue
        t0 = time.perf_counter()
        hits = retriever.retrieve(query, k, mode)
        latencies.append((time.perf_counter() - t0) * 1000)
        ids = [h.chunk.chunk_id for h in hits]
        gold = resolve_gold(item, by_doc)
        for bucket in ("all", item.get("category", "all")):
            buckets[bucket]["hit"].append(hit_at_k(ids, gold, k))
            buckets[bucket]["recall"].append(recall_at_k(ids, gold, k))
            buckets[bucket]["mrr"].append(mrr(ids, gold))
    out = {b: {m: statistics.mean(v) for m, v in ms.items()} | {"n": len(ms["hit"])} for b, ms in buckets.items()}
    out["all"]["p50_ms"] = statistics.median(latencies)
    out["skipped"] = skipped
    return out


def judge_answers(items: list[dict], mode: str) -> dict:
    retriever = build_retriever(use_reranker=mode == "hybrid_rerank")
    grades = defaultdict(lambda: defaultdict(int))
    repaired = failed_closed = refused_ok = unanswerable = 0
    for item in items:
        query = search_query(item, use_llm=True)
        hits = retriever.retrieve(query, CONFIG.top_k, mode)
        ans = answer(query, [h.chunk for h in hits], CONFIG.llm_model)
        repaired += ans.attempts > 1
        failed_closed += not ans.grounded
        if item.get("category") == "unanswerable":
            unanswerable += 1
            refused_ok += ans.text.strip() == REFUSAL
            continue
        verdict = llm_call(
            JUDGE.format(question=query, reference=item["reference_answer"], answer=ans.text),
            CONFIG.judge_model,
        ).strip().upper()
        grade = next((g for g in ("CORRECT", "PARTIAL", "WRONG") if g in verdict), "WRONG")
        for bucket in ("all", item.get("category", "all")):
            grades[bucket][grade] += 1
    return {"grades": grades, "repaired": repaired, "failed_closed": failed_closed,
            "refused_ok": refused_ok, "unanswerable": unanswerable}


def _fmt_grades(g: dict) -> str:
    return f"{g.get('CORRECT', 0)}C {g.get('PARTIAL', 0)}P {g.get('WRONG', 0)}W"


def run_eval(modes: list[str], eval_set: str = "golden", use_llm: bool = True, gate: bool = False) -> int:
    items = load_set(eval_set)
    if not items:
        print(f"no data/eval/{eval_set}.jsonl")
        return 1
    categories = CATEGORIES if eval_set == "golden" else DEV_CATEGORIES
    k = CONFIG.top_k
    print(f"set={eval_set}  n={len(items)}  k={k}\n")
    print(f"{'mode':<15}{'category':<13}{'n':>4}{'hit@k':>7}{'recall@k':>10}{'MRR':>7}{'p50 ms':>9}")
    results = {}
    for mode in modes:
        results[mode] = scores = retrieval_scores(items, mode, k, use_llm)
        for cat in ("all", *categories):
            if cat not in scores:
                continue
            s = scores[cat]
            latency = f"{s['p50_ms']:9.1f}" if "p50_ms" in s else ""
            print(f"{mode if cat == 'all' else '':<15}{cat:<13}{s['n']:>4}"
                  f"{s['hit']:7.3f}{s['recall']:10.3f}{s['mrr']:7.3f}{latency}")
    skipped = next(iter(results.values()))["skipped"]
    if skipped:
        print(f"\n({skipped} multi-turn items skipped: they need the LLM to condense the query)")

    prod = CONFIG.retrieval_mode
    if use_llm:
        j = judge_answers(items, prod)
        print(f"\nanswers ({prod}, judged by {CONFIG.judge_model}): {_fmt_grades(j['grades']['all'])}")
        for cat in categories:
            if cat in j["grades"]:
                print(f"  {cat:<13}{_fmt_grades(j['grades'][cat])}")
        print(f"citation repairs {j['repaired']}   failed-closed {j['failed_closed']}")
        if j["unanswerable"]:
            print(f"unanswerable correctly refused: {j['refused_ok']}/{j['unanswerable']}")

    if gate:
        thresholds = json.loads((CONFIG.eval_dir / "thresholds.json").read_text())[eval_set]["retrieval"]
        actual = results.get(prod) or retrieval_scores(items, prod, k, use_llm)
        actual = actual["all"]
        failures = [f"{m} {actual[m]:.3f} < {t}" for m, t in thresholds.items() if actual[m] < t]
        print("\nGATE " + ("FAILED: " + "; ".join(failures) if failures else "passed"))
        return 1 if failures else 0
    return 0
