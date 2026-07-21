"""Stage 1, step 2 — YOUR CODE: split documents into chunks.

Why chunk at all? Two reasons:
1. Embedding models have a token limit and, more importantly, one vector can
   only faithfully represent one *idea*. Embed a whole 30-page doc into one
   vector and it means nothing.
2. The LLM context you assemble at answer time should be small and dense with
   relevant text.

Stage 1 rule: FIXED-SIZE chunks with overlap. It's the dumb baseline, and you
need a dumb baseline so Stage 5's clever chunking has a number to beat.

Design decisions you must make (write your reasoning in the README later):
- chunk_size / overlap: start with config values, don't agonize.
- Split on characters or tokens? Characters are fine for the baseline.
- What metadata does a chunk carry? At minimum: which doc, which page(s),
  and a stable chunk_id — the eval harness will reference chunk_ids as
  "gold" labels, so ids must not change between runs (no uuid4!).

Hint: a stable id scheme like f"{doc_id}:{chunk_index:04d}" works.
"""

from dataclasses import dataclass

from .ingest import Document


@dataclass
class Chunk:
    chunk_id: str    # stable across runs — eval gold labels depend on it
    doc_id: str
    page_start: int  # 1-based, inclusive
    page_end: int
    text: str


def chunk_document(doc: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    """Fixed-size character chunks with overlap.

    Edge cases your test suite should cover:
    - text shorter than chunk_size -> exactly one chunk
    - overlap >= chunk_size -> raise ValueError (would loop forever)
    - no empty/whitespace-only chunks
    """
    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk size")
    stride = chunk_size - overlap
    text = "\n".join(doc.pages)
    start = 0
    index = 0
    chunks = []
    while start < len(text):
        window = text[start: start+chunk_size]
        if not window.strip():
            start += stride
            continue

        curr = Chunk(
            chunk_id=f"{doc.doc_id}:{index:04d}",
            doc_id=doc.doc_id,
            page_start=1,
            page_end=1,
            text=window,
        )
        chunks.append(curr)
        start += stride
        index += 1
    return chunks


def chunk_corpus(docs: list[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk every document in the corpus into one flat list of chunks.

    Applies `chunk_document` to each `Document` with the same chunk_size and
    overlap, then flattens the per-document chunk lists into a single list —
    the shape the embedding and index stages expect downstream.

    chunk_ids stay unique across the corpus because each is prefixed with its
    own `doc_id` (e.g. "report:0000" vs "memo:0000"), so no collisions even
    though every doc restarts its index at 0000.
    """
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size, overlap))
    return all_chunks
    
