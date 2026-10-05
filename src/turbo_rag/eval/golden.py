"""Eval sets, gold-label resolution, and the golden-set validator.

Two sets, two jobs:

- dev.jsonl     LLM-generated + quick hand-written questions. Iterate and tune
                against these freely. Labels: chunk ids (cheap, but they break
                when chunking changes).
- golden.jsonl  Hand-written, hand-verified, FROZEN. Report numbers and gate CI
                on this set only. Never tune on it — once you tweak the system
                until golden goes up, it's just another dev set and your
                reported numbers are overfit.

Golden labels are chunking-independent: a question names its gold BOOKS
(`gold_doc_ids`, every edition) plus an optional `evidence` quote (or list of
alternative quotes, when editions word it differently). Gold chunks are
resolved at eval time = chunks of those books containing any evidence quote
(or all their chunks when there's no evidence). Change chunk_size tomorrow and
every label still holds.

Provenance: `written_by` / `reviewed_by`. One person writing every question is
a bias of its own (their phrasing, their blind spots) — the validator reports
how many items still lack a second reviewer.

Golden row:
    {"qid": "g009", "category": "lookup", "question": "...",
     "gold_doc_ids": ["9780805076233"], "evidence": "September 1, 1939",
     "reference_answer": "...", "history": [...],  # multi_turn only
     "written_by": "claude", "reviewed_by": null}

`rag validate-golden` checks every label against the corpus before it can be
trusted; CI runs it before the retrieval gate.
"""

import json
import re
from collections import defaultdict

from ..chunking import Chunk
from ..config import CONFIG

CATEGORIES = (
    "vague",         # paraphrased description, no names
    "lookup",        # one fact, pinned by evidence
    "metadata",      # title / author / category — answered by the chunk header
    "stats",         # rating / page count
    "author_list",   # every book by an author — recall
    "multi_book",    # needs two different books in context
    "filter",        # constraint on year / category — retrieval can't filter
    "typo",          # misspelled query
    "multi_turn",    # follow-up that needs history
    "unanswerable",  # must refuse, incl. near-misses
)


def load_set(name: str) -> list[dict]:
    path = CONFIG.eval_dir / f"{name}.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def chunks_by_doc(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    out = defaultdict(list)
    for c in chunks:
        out[c.doc_id].append(c)
    return out


def resolve_gold(item: dict, by_doc: dict[str, list[Chunk]]) -> set[str]:
    """Gold chunk ids for an item (dev rows carry them directly)."""
    if "gold_chunk_ids" in item:
        return set(item["gold_chunk_ids"])
    evidence = item.get("evidence") or []
    quotes = [_norm(e) for e in ([evidence] if isinstance(evidence, str) else evidence)]
    gold = set()
    for doc_id in item["gold_doc_ids"]:
        for c in by_doc.get(doc_id, []):
            if not quotes or any(q in _norm(c.text) for q in quotes):
                gold.add(c.chunk_id)
    return gold


def _edition_key(c: Chunk) -> str:
    title = c.metadata.get("title", "").split(":")[0]
    return re.sub(r"[^a-z0-9]", "", title.lower()) + "|" + c.metadata.get("authors", "").lower()


def validate_golden(golden: list[dict], chunks: list[Chunk], dev: list[dict]) -> tuple[list[str], list[str]]:
    """(errors, warnings). Errors mean a label is wrong; warnings need a human look."""
    errors, warnings = [], []
    by_doc = chunks_by_doc(chunks)
    editions = defaultdict(set)
    for c in chunks:
        editions[_edition_key(c)].add(c.doc_id)
    dev_docs = {cid.split(":")[0] for q in dev for cid in q.get("gold_chunk_ids", [])}

    qids = [q["qid"] for q in golden]
    for dup in {q for q in qids if qids.count(q) > 1}:
        errors.append(f"duplicate qid {dup}")

    for q in golden:
        qid, cat = q["qid"], q.get("category")
        for field in ("question", "reference_answer", "gold_doc_ids", "category"):
            if field not in q:
                errors.append(f"{qid}: missing {field}")
        if cat not in CATEGORIES:
            errors.append(f"{qid}: unknown category {cat!r}")
        if cat == "unanswerable":
            if q.get("gold_doc_ids"):
                errors.append(f"{qid}: unanswerable item must have no gold docs")
            continue
        if cat == "multi_turn" and not q.get("history"):
            errors.append(f"{qid}: multi_turn item needs history")
        if not q.get("gold_doc_ids"):
            errors.append(f"{qid}: answerable item has no gold docs")
        for doc_id in q.get("gold_doc_ids", []):
            if doc_id not in by_doc:
                errors.append(f"{qid}: gold doc {doc_id} not in corpus")
                continue
            # another edition of the same book that isn't labeled -> a correct
            # retrieval of it would be scored as a miss
            missing = editions[_edition_key(by_doc[doc_id][0])] - set(q["gold_doc_ids"])
            if missing:
                warnings.append(f"{qid}: other editions of {doc_id} not labeled: {sorted(missing)}")
        if q.get("evidence") and not resolve_gold(q, by_doc):
            errors.append(f"{qid}: evidence {q['evidence']!r} not found in any gold doc")
        leaked = dev_docs & set(q.get("gold_doc_ids", []))
        if leaked:
            warnings.append(f"{qid}: gold docs also used in dev set: {sorted(leaked)}")
    return errors, warnings


def run_validate() -> int:
    from ..pipeline import load_chunks

    golden = load_set("golden")
    errors, warnings = validate_golden(golden, load_chunks(), load_set("dev"))
    counts = defaultdict(int)
    for q in golden:
        counts[q.get("category")] += 1
    print(f"golden: {len(golden)} items  " + "  ".join(f"{c}={counts[c]}" for c in CATEGORIES))
    unreviewed = [q["qid"] for q in golden if not q.get("reviewed_by")]
    if unreviewed:
        print(f"  INFO  {len(unreviewed)}/{len(golden)} items not yet reviewed by a second person")
    for w in warnings:
        print(f"  WARN  {w}")
    for e in errors:
        print(f"  ERROR {e}")
    print("valid" if not errors else f"{len(errors)} errors")
    return 1 if errors else 0
