"""Stage 1, step 5 — YOUR CODE: retrieved chunks + question -> answer.

This is the "G" in RAG and deliberately the least interesting part of this
project. Keep it simple:

1. Build a prompt: the retrieved chunks (with their doc/page metadata so the
   model can cite sources) + the user question + an instruction to answer ONLY
   from the provided context and say "I don't know" otherwise. That last
   instruction is what makes hallucinations visible instead of hidden.
2. One call to the Gemini API:
       from google import genai
       client = genai.Client()  # reads GEMINI_API_KEY from env
       resp = client.models.generate_content(model=model, contents=prompt)
       resp.text
   Docs: https://ai.google.dev/gemini-api/docs
3. Return the text.

Hints:
- Free tier has per-minute rate limits — on a 429, sleep and retry. The disk
  cache (below) is also your rate-limit shield: cached questions cost nothing.
- Cache responses to disk keyed on hash(model + prompt) — the eval harness
  (Stage 2) will re-ask the same 50 questions dozens of times as you iterate,
  and cached runs are free and fast. Build the cache NOW, thank yourself later.
"""

from .chunking import Chunk


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    raise NotImplementedError("TODO(suyash)")


def answer(question: str, chunks: list[Chunk], model: str) -> str:
    """Call the LLM (through the disk cache) and return the answer text."""
    raise NotImplementedError("TODO(suyash)")
