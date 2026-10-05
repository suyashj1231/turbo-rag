"""Synthetic DEV set: sample books, have an LLM write questions about them.

This is the set you iterate on. Reported numbers come from the hand-verified
golden set (golden.py), never from this one.

Hand-writing 50 questions is the gold standard; generating them is how teams
bootstrap (Ragas' TestsetGenerator does the same thing). The trick that makes
synthetic questions useful for RETRIEVAL eval: we know which book each
question was generated from, so the gold label comes for free.

Three question styles, because they stress different retrievers:
- "content": describes the book WITHOUT naming title or author
  ("Which novel follows a dying preacher in 1950s Iowa writing to his son?").
  Paraphrase-heavy -> dense retrieval should win.
- "lookup": names the title or author and asks a detail
  ("In Gilead, what town is the story set in?"). Exact-token heavy -> BM25 wins.
- "vague": how real users actually ask — half-remembered, no names, no
  distinctive phrases copied from the description ("that sci-fi book where a
  kid wins a space suit and ends up on the moon"). This style exists because
  the first eval run was SATURATED: LLM-written "content" questions copied
  rare words (character names) from the text, so BM25 scored MRR 1.000 and
  the eval could no longer tell retrievers apart.
Hybrid should beat both on the mix. That's the hypothesis the eval tests.

Gold = every chunk of every edition of the book (the dataset has duplicate
titles under different ISBNs; retrieving any edition is a correct hit).

Known bias to mention in interviews: questions written by an LLM from a single
chunk are easier and more "lexically leaky" than real user questions. Mitigate
by spot-checking, banning title words in content questions, and adding
hand-written questions over time.
"""

import json
import random
import re
from collections import defaultdict

from ..config import CONFIG
from ..generate import llm_call
from ..pipeline import load_chunks

PROMPT = """You write evaluation questions for a book search assistant.
For each book below, write ONE question a reader might ask, plus a short reference answer, using ONLY the book's description.

Style for each book is given as "style":
- "content": describe what the book is about so the book can be identified, but do NOT mention the title, any word of the title, or the author. The reference answer is the book's title and author.
- "vague": write like a reader who half-remembers the book, in casual words. Do NOT use the title, the author, ANY proper noun (no character, place, or organization names), or any distinctive phrase from the description — paraphrase everything with common words. The reference answer is the book's title and author.
- "lookup": mention the exact title, and ask about a specific detail found in the description (a character, place, time period, or event). The reference answer is that detail, in one sentence.

Return a JSON array, one object per book, in the same order: [{{"id": "...", "question": "...", "reference_answer": "..."}}]. Output only JSON.

Books:
{books}"""


def _parse_json(text: str) -> list[dict]:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


def make_questions(n: int = 45, seed: int = 7, batch: int = 10) -> None:
    chunks = load_chunks()
    by_book = defaultdict(list)  # (title, authors) -> chunk_ids across editions
    for c in chunks:
        by_book[(c.metadata.get("title"), c.metadata.get("authors"))].append(c.chunk_id)

    # first chunk of books with a meaty description, deterministic sample
    candidates = [c for c in chunks if c.chunk_id.endswith(":0000") and len(c.text) > 400]
    rng = random.Random(seed)
    sample = rng.sample(candidates, n)

    out_path = CONFIG.eval_dir / "dev.jsonl"
    rows = []
    for start in range(0, n, batch):
        group = sample[start: start + batch]
        books = []
        for i, c in enumerate(group):
            style = ("content", "vague", "lookup")[(start + i) % 3]
            books.append({"id": c.chunk_id, "style": style, "text": c.text})
        items = _parse_json(llm_call(PROMPT.format(books=json.dumps(books, indent=1)), CONFIG.judge_model))
        for book, item in zip(books, items):
            chunk = next(c for c in group if c.chunk_id == book["id"])
            key = (chunk.metadata.get("title"), chunk.metadata.get("authors"))
            rows.append({
                "qid": f"q{len(rows) + 1:03d}",
                "category": book["style"],
                "question": item["question"],
                "gold_chunk_ids": sorted(by_book[key]),
                "reference_answer": item["reference_answer"],
            })
        print(f"generated {len(rows)}/{n}")

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} questions -> {out_path}. Read them! Delete bad ones.")
