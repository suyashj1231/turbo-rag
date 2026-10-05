"""Stage 1, step 5: retrieved chunks + question -> CITED answer.

Citation enforcement is three layers, each catching what the previous misses:

1. Prompt contract: sources are numbered [1]..[n]; the model must cite a
   source after every factual sentence, use ONLY the sources, and reply with
   the exact refusal string when they don't contain the answer.
2. Validation: parse the answer. Every [n] must point at a real source, and
   every substantive sentence must carry at least one citation.
3. Repair, then fail closed: one retry that tells the model exactly what was
   wrong; if it still fails, return the refusal instead of an uncited answer.
   An unverifiable answer is worse than "I don't know" for a citation system.

Plumbing:
- Disk cache keyed on sha256(model + prompt). The eval re-asks the same
  questions dozens of times while you iterate; cached runs are free and fast.
- 429/503 -> exponential backoff (Gemini free tier has per-minute limits).
- temperature=0 so cached and uncached answers agree, and evals are stable.
"""

import hashlib
import json
import re
import time
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from .chunking import Chunk
from .config import CONFIG

REFUSAL = "I don't know based on the provided sources."
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# split after . ! ? — but not after initials ("K. Sadlon") or common abbreviations
_SENTENCE = re.compile(
    r"(?<=[.!?])(?<!\b[A-Z]\.)(?<!Mr\.)(?<!Ms\.)(?<!Dr\.)(?<!St\.)(?<!Jr\.)(?<!Sr\.)(?<!vs\.)"
    r"\s+(?=[A-Z\"'(*-])|\n+"
)
# "...Robinson. [1]" -> "...Robinson [1]." so the citation stays with its sentence
_TRAILING_CITATION = re.compile(r"([.!?])\s*((?:\[\d+(?:\s*,\s*\d+)*\]\s*)+)")

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def llm_call(prompt: str, model: str, system: str | None = None) -> str:
    """One cached, retried Gemini call. Returns the response text."""
    key = hashlib.sha256(f"{model}\x00{system}\x00{prompt}".encode()).hexdigest()
    path = CONFIG.cache_dir / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["text"]

    cfg = types.GenerateContentConfig(temperature=0.0, system_instruction=system)
    for attempt in range(6):
        try:
            resp = _get_client().models.generate_content(model=model, contents=prompt, config=cfg)
            text = (resp.text or "").strip()
            break
        except Exception as e:
            transient = any(code in str(e) for code in ("429", "500", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
            daily_quota = "PerDay" in str(e)  # waiting seconds won't fix a daily cap
            if not transient or daily_quota or attempt == 5:
                raise
            time.sleep(min(60, 2 ** attempt * 4))

    CONFIG.cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model, "prompt": prompt, "text": text}))
    return text


# ---------------------------------------------------------------- prompting

SYSTEM = f"""You are a careful assistant answering questions about books using ONLY the numbered sources provided.

Rules:
- Every sentence that states a fact MUST end with one or more citations like [1] or [2, 3], referring to the source numbers.
- Use only information in the sources. Do not use outside knowledge, even if you know the answer.
- If the sources do not contain the answer, reply with exactly: {REFUSAL}
- Be concise: at most 5 sentences unless the user asks for a list."""


def format_sources(chunks: list[Chunk]) -> str:
    blocks = []
    for n, c in enumerate(chunks, start=1):
        blocks.append(f"[{n}] (id: {c.chunk_id})\n{c.text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    return f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}\n\nAnswer (cite sources):"


# --------------------------------------------------------------- validation

@dataclass
class Citation:
    n: int            # 1-based source number as written in the answer
    chunk_id: str
    title: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = True      # passed validation (or is an honest refusal)
    refused: bool = False
    attempts: int = 1
    contexts: list[str] = field(default_factory=list)  # texts shown to the LLM (Ragas needs these)


def cited_numbers(text: str) -> list[int]:
    nums = []
    for group in _CITATION.findall(text):
        nums.extend(int(x) for x in group.split(","))
    return nums


def validate(text: str, n_sources: int) -> list[str]:
    """Problems with an answer's citations; empty list == valid."""
    if text.strip() == REFUSAL:
        return []
    problems = []
    nums = cited_numbers(text)
    if not nums:
        problems.append("The answer contains no citations.")
    bad = sorted({n for n in nums if not 1 <= n <= n_sources})
    if bad:
        problems.append(f"Citations {bad} do not exist; valid sources are 1..{n_sources}.")
    normalized = _TRAILING_CITATION.sub(lambda m: f" {m.group(2).strip()}{m.group(1)} ", text.strip())
    for sentence in _SENTENCE.split(normalized):
        words = re.sub(r"[^A-Za-z ]", " ", sentence).split()
        is_lead_in = sentence.rstrip().endswith(":")  # "Here are three options:"
        if len(words) >= 4 and not is_lead_in and not _CITATION.search(sentence):
            problems.append(f"This sentence has no citation: {sentence.strip()!r}")
    return problems


def answer(question: str, chunks: list[Chunk], model: str) -> Answer:
    """Generate, validate citations, retry once with feedback, fail closed."""
    contexts = [c.text for c in chunks]
    if not chunks:
        return Answer(REFUSAL, refused=True, contexts=contexts)

    prompt = build_prompt(question, chunks)
    text = llm_call(prompt, model, SYSTEM)
    problems = validate(text, len(chunks))
    attempts = 1
    if problems:
        attempts = 2
        repair = (
            f"{prompt}\n\nYour previous answer was:\n{text}\n\n"
            "It violated the citation rules:\n- " + "\n- ".join(problems)
            + "\n\nRewrite it so every factual sentence cites a valid source, "
            f"or reply exactly '{REFUSAL}'."
        )
        text = llm_call(repair, model, SYSTEM)
        problems = validate(text, len(chunks))

    if problems:
        return Answer(REFUSAL, grounded=False, refused=True, attempts=attempts, contexts=contexts)

    refused = text.strip() == REFUSAL
    citations = []
    for n in dict.fromkeys(cited_numbers(text)):  # unique, in order of first use
        c = chunks[n - 1]
        citations.append(Citation(n, c.chunk_id, c.metadata.get("title", c.doc_id)))
    return Answer(text, citations, grounded=True, refused=refused, attempts=attempts, contexts=contexts)


# --------------------------------------------------------------- multi-turn

CONDENSE = """Rewrite the user's latest message as a standalone search query that can be understood without the conversation. Resolve pronouns and references ("it", "that author", "the second one") using the history. Keep names and titles exact. Output only the rewritten query.

Conversation:
{history}

Latest message: {question}

Standalone query:"""


def condense_question(question: str, history: list[dict], model: str) -> str:
    """'Who wrote it?' + history -> 'Who wrote Gilead?'. No-op on turn one.

    Retrieval only sees the current query, so without this step follow-ups
    retrieve garbage. This is the core trick of multi-turn RAG.
    """
    if not history:
        return question
    lines = [f"{m['role']}: {m['content']}" for m in history]
    return llm_call(CONDENSE.format(history="\n".join(lines), question=question), model).strip() or question
