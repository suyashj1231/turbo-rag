"""Golden-set label resolution and validation — the validator must catch bad labels."""

import json

from turbo_rag.chunking import Chunk
from turbo_rag.config import CONFIG
from turbo_rag.eval.golden import CATEGORIES, chunks_by_doc, resolve_gold, validate_golden

CHUNKS = [
    Chunk("b1:0000", "b1", 1, 1, "Title: War\nIt began on September 1, 1939.", {"title": "War", "authors": "Gilbert"}),
    Chunk("b1:0001", "b1", 1, 1, "Title: War\nIt ended in 1945.", {"title": "War", "authors": "Gilbert"}),
    Chunk("b2:0000", "b2", 1, 1, "Title: War\nAnother edition.", {"title": "War", "authors": "Gilbert"}),
    Chunk("b3:0000", "b3", 1, 1, "Title: Peace\nCalm.", {"title": "Peace", "authors": "Tolstoy"}),
]


def item(**kw):
    base = {"qid": "g1", "category": "lookup", "question": "q", "reference_answer": "a", "gold_doc_ids": ["b1", "b2"]}
    return base | kw


def test_evidence_narrows_gold_to_matching_chunks():
    gold = resolve_gold(item(evidence="september   1, 1939"), chunks_by_doc(CHUNKS))  # whitespace/case-insensitive
    assert gold == {"b1:0000"}


def test_no_evidence_means_all_chunks_of_gold_docs():
    assert resolve_gold(item(), chunks_by_doc(CHUNKS)) == {"b1:0000", "b1:0001", "b2:0000"}


def test_dev_rows_use_chunk_ids_directly():
    assert resolve_gold({"gold_chunk_ids": ["x:0000"]}, {}) == {"x:0000"}


def test_valid_item_passes():
    errors, warnings = validate_golden([item(evidence="1939")], CHUNKS, dev=[])
    assert errors == [] and warnings == []


def test_validator_catches_bad_labels():
    golden = [
        item(qid="g1", evidence="1066"),                      # evidence not in gold docs
        item(qid="g2", gold_doc_ids=["nope"]),                # doc not in corpus
        item(qid="g3", category="multi_turn"),                # missing history
        item(qid="g4", category="unanswerable"),              # unanswerable with gold
        item(qid="g4"),                                       # duplicate qid
    ]
    errors, _ = validate_golden(golden, CHUNKS, dev=[])
    joined = "\n".join(errors)
    for expected in ("g1: evidence", "g2: gold doc nope", "g3: multi_turn", "g4: unanswerable", "duplicate qid g4"):
        assert expected in joined


def test_validator_warns_on_unlabeled_edition_and_dev_leak():
    dev = [{"gold_chunk_ids": ["b1:0000"]}]
    _, warnings = validate_golden([item(gold_doc_ids=["b1"])], CHUNKS, dev=dev)
    joined = "\n".join(warnings)
    assert "other editions of b1 not labeled: ['b2']" in joined
    assert "also used in dev set" in joined


def test_committed_golden_set_is_well_formed():
    """Schema-level check that runs without the corpus (full check: `rag validate-golden`)."""
    rows = [json.loads(line) for line in open(CONFIG.eval_dir / "golden.jsonl")]
    assert len({r["qid"] for r in rows}) == len(rows)
    assert {r["category"] for r in rows} == set(CATEGORIES)  # every category covered
    for r in rows:
        assert r["question"] and r["reference_answer"]
        assert bool(r["gold_doc_ids"]) == (r["category"] != "unanswerable")
        assert bool(r.get("history")) == (r["category"] == "multi_turn")
