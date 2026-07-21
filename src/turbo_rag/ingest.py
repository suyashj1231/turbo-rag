"""Stage 1, step 1 — YOUR CODE: turn raw PDFs into clean text.

Goal: for every PDF in `data/raw/`, produce a Document with its full text.

Why this matters: garbage in, garbage out. Most real-world RAG failures are
ingestion failures (mangled tables, headers/footers glued into sentences,
hyphen- ated words). You chose a messy corpus on purpose — look at the raw
extracted text of at least one full document before moving on. Seriously. Open
it and read it.

Hints:
- `from pypdf import PdfReader; reader.pages[i].extract_text()`
- Keep page numbers! Store text per-page or record page boundaries — you'll
  want "cite the page" later, and it makes debugging retrieval much easier.
- Don't over-clean at first. Get end-to-end working, then come back when the
  eval tells you ingestion is the bottleneck (it will).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from pypdf import PdfReader


@dataclass
class Document:
    doc_id: str          # e.g. the filename stem
    source_path: str
    pages: list[str]     # text per page, index = page number - 1


def load_pdf(path: Path) -> Document:
    """Extract text from one PDF, page by page."""
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        pages.append(text)
    
    doc_id = path.stem
    return Document(doc_id, str(path), pages)


def load_corpus(raw_dir: Path) -> list[Document]:
    """Load every PDF in raw_dir. Skip non-PDFs. Print what you loaded."""
    docs = []
    for path in raw_dir.iterdir():
        if path.suffix == ".pdf":
            print(f"Loading {path}")
            docs.append(load_pdf(path))
            print(f"Loaded {path} : {len(docs[-1].pages)} pages")
    
    return docs
