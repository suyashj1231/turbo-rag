"""Stage 1, step 1: turn raw sources into Documents.

Two loaders, one output type:
- PDFs (the original corpus): one Document per file, text kept per page so we
  can cite pages.
- The Kaggle "7k Books with metadata" CSV: one Document per book. The
  description is the body; title/authors/categories/year live in `metadata`
  so chunking can stamp them onto every chunk and citations can name the book.

Garbage in, garbage out: most real-world RAG failures are ingestion failures.
For the books CSV that means: ~260 rows have no description (we still index
title + metadata so "who wrote X?" works), authors are ';'-separated, and
ratings with ratings_count == 0 are dropped (a 0.0 "rating" is missing data).
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Document:
    doc_id: str          # e.g. the filename stem, or isbn13 for books
    source_path: str
    pages: list[str]     # text per page, index = page number - 1
    metadata: dict = field(default_factory=dict)


def load_pdf(path: Path) -> Document:
    """Extract text from one PDF, page by page."""
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return Document(path.stem, str(path), pages)


def load_books_csv(path: Path) -> list[Document]:
    """One Document per book row. Dedupes on isbn13 so chunk_ids stay unique."""
    docs = []
    seen = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            isbn = row["isbn13"].strip()
            if not isbn or isbn in seen:
                continue
            seen.add(isbn)
            title = row["title"].strip()
            if row["subtitle"].strip():
                title = f"{title}: {row['subtitle'].strip()}"
            year = row["published_year"].strip()
            ratings_count = int(float(row["ratings_count"] or 0))
            metadata = {
                "title": title,
                "authors": ", ".join(a.strip() for a in row["authors"].split(";") if a.strip()),
                "categories": row["categories"].strip(),
                "published_year": int(float(year)) if year else None,
                # rating is meaningless with no ratings behind it
                "average_rating": row["average_rating"].strip() if ratings_count else "",
                "ratings_count": ratings_count,
                "num_pages": row["num_pages"].strip() if row["num_pages"].strip() not in ("", "0") else "",
            }
            docs.append(Document(isbn, str(path), [row["description"].strip()], metadata))
    return docs


def load_corpus(raw_dir: Path) -> list[Document]:
    """Load every PDF and CSV in raw_dir. Print what you loaded."""
    docs = []
    for path in sorted(raw_dir.iterdir()):
        if path.suffix == ".pdf":
            docs.append(load_pdf(path))
            print(f"Loaded {path.name}: {len(docs[-1].pages)} pages")
        elif path.suffix == ".csv":
            books = load_books_csv(path)
            docs.extend(books)
            print(f"Loaded {path.name}: {len(books)} books")
    return docs
