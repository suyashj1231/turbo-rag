"""Stage 1, step 2: split documents into chunks.

Why chunk at all? Two reasons:
1. Embedding models have a token limit and, more importantly, one vector can
   only faithfully represent one *idea*. Embed a whole 30-page doc into one
   vector and it means nothing.
2. The LLM context you assemble at answer time should be small and dense with
   relevant text.

Baseline: FIXED-SIZE character chunks with overlap, with stable ids
(f"{doc_id}:{chunk_index:04d}") because eval gold labels reference them.

Contextual header: when a Document carries metadata (the books corpus), every
chunk is prefixed with "Title / Authors / Categories / Year". A chunk from the
middle of a long description otherwise has no idea which book it belongs to,
so neither BM25 ("books by Tolkien") nor the embedding can match it. This is
the cheap version of Anthropic's "contextual retrieval".
"""

from dataclasses import dataclass, field

from .ingest import Document


@dataclass
class Chunk:
    chunk_id: str    # stable across runs — eval gold labels depend on it
    doc_id: str
    page_start: int  # 1-based, inclusive
    page_end: int
    text: str
    metadata: dict = field(default_factory=dict)


def header_for(metadata: dict) -> str:
    """'Title: X\\nAuthors: Y\\n...' — the context stamped onto every chunk."""
    rating = metadata.get("average_rating")
    if rating:
        rating = f"{rating} ({metadata.get('ratings_count')} ratings)"
    fields = [
        ("Title", metadata.get("title")),
        ("Authors", metadata.get("authors")),
        ("Categories", metadata.get("categories")),
        ("Published", metadata.get("published_year")),
        # added after the golden set's "stats" questions showed the corpus
        # couldn't answer "how many pages / how well rated is X?"
        ("Rating", rating),
        ("Pages", metadata.get("num_pages")),
    ]
    return "".join(f"{name}: {value}\n" for name, value in fields if value)


def _page_of(offset: int, page_starts: list[int]) -> int:
    """1-based page containing character `offset` of the joined text."""
    page = 1
    for i, start in enumerate(page_starts):
        if offset >= start:
            page = i + 1
    return page


def chunk_document(doc: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    """Fixed-size character chunks with overlap.

    - text shorter than chunk_size -> exactly one chunk
    - overlap >= chunk_size -> ValueError (would loop forever)
    - no empty/whitespace-only chunks
    """
    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk size")
    stride = chunk_size - overlap
    header = header_for(doc.metadata)

    page_starts = []
    offset = 0
    for page in doc.pages:
        page_starts.append(offset)
        offset += len(page) + 1  # +1 for the joining "\n"
    text = "\n".join(doc.pages)

    # A book with no description still gets one chunk: its header alone
    # answers "who wrote X?" / "when was X published?".
    if not text.strip() and header:
        return [Chunk(f"{doc.doc_id}:0000", doc.doc_id, 1, 1, header.strip(), doc.metadata)]

    chunks = []
    start = 0
    index = 0
    while start < len(text):
        window = text[start: start + chunk_size]
        if window.strip():
            end = start + len(window) - 1
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}:{index:04d}",
                doc_id=doc.doc_id,
                page_start=_page_of(start, page_starts),
                page_end=_page_of(end, page_starts),
                text=header + window,
                metadata=doc.metadata,
            ))
            index += 1
        if start + chunk_size >= len(text):
            break  # this window reached the end; another would be pure overlap
        start += stride
    return chunks


def chunk_corpus(docs: list[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk every document in the corpus into one flat list of chunks.

    chunk_ids stay unique across the corpus because each is prefixed with its
    own `doc_id` (e.g. "report:0000" vs "memo:0000").
    """
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size, overlap))
    return all_chunks
