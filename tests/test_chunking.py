"""Executable spec for chunking.py — make these pass, then add your own."""

import pytest

from turbo_rag.chunking import chunk_document
from turbo_rag.ingest import Document


def make_doc(text: str) -> Document:
    return Document(doc_id="doc", source_path="doc.pdf", pages=[text])


def test_short_text_is_one_chunk():
    chunks = chunk_document(make_doc("hello world"), chunk_size=100, overlap=20)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"


def test_chunks_cover_all_text():
    text = "abcdefghij" * 50  # 500 chars
    chunks = chunk_document(make_doc(text), chunk_size=120, overlap=30)
    assert all(len(c.text) <= 120 for c in chunks)
    # every character position is inside some chunk (overlap means no gaps)
    reconstructed = "".join(c.text[: 120 - 30] for c in chunks[:-1]) + chunks[-1].text
    assert text in reconstructed or reconstructed.startswith(text[:400])


def test_ids_are_stable_across_runs():
    doc = make_doc("abcdefghij" * 50)
    a = chunk_document(doc, chunk_size=120, overlap=30)
    b = chunk_document(doc, chunk_size=120, overlap=30)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)  # unique


def test_bad_overlap_raises():
    with pytest.raises(ValueError):
        chunk_document(make_doc("x" * 500), chunk_size=100, overlap=100)


def test_no_empty_chunks():
    chunks = chunk_document(make_doc("word " * 200), chunk_size=100, overlap=20)
    assert all(c.text.strip() for c in chunks)
