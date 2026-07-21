"""Stage 2 — YOUR CODE: one command, one scoreboard.

Reads data/eval/questions.jsonl, where each line is:

    {"qid": "q001",
     "question": "What discount rate did the 2023 report assume?",
     "gold_chunk_ids": ["report2023:0042"],
     "reference_answer": "5.5%"}

For each question:
  1. embed -> search -> record retrieved chunk_ids  (retrieval metrics)
  2. generate an answer                             (cached LLM call)
  3. LLM-as-judge: show a strong model the question, reference_answer, and
     the generated answer; ask for CORRECT / PARTIAL / WRONG. Judge with a
     better model than the one answering, and cache judge calls too.

Print a scoreboard like:

    n=50   hit@5 0.84   recall@5 0.79   MRR 0.71   answers: 38C 7P 5W

Writing the 30-50 questions is real work — budget an afternoon. Mix difficulty:
some answerable from one chunk, some needing a table, a few multi-chunk. Label
gold_chunk_ids by grepping chunks.jsonl for the answer text.

Design note: pass the index in as a parameter (default FlatIndex) — in Stage 4
this same harness runs unchanged over TurboQuant at every bit-width.
"""


def run_eval() -> None:
    raise NotImplementedError("TODO(suyash)")
