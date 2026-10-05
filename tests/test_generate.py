"""Citation enforcement: validation rules + retry/fail-closed behavior (LLM mocked)."""

from turbo_rag import generate
from turbo_rag.chunking import Chunk
from turbo_rag.generate import REFUSAL, answer, cited_numbers, validate

CHUNKS = [
    Chunk("b1:0000", "b1", 1, 1, "Title: Gilead\nA preacher in Iowa.", {"title": "Gilead"}),
    Chunk("b2:0000", "b2", 1, 1, "Title: Home\nA prodigal son returns.", {"title": "Home"}),
]


def test_cited_numbers_parses_groups():
    assert cited_numbers("A [1]. B [2, 3]. C [1,2].") == [1, 2, 3, 1, 2]


def test_valid_answer_passes():
    assert validate("Gilead is about a preacher in Iowa [1].", 2) == []


def test_citation_after_period_is_accepted():
    assert validate("Gilead is about a preacher in Iowa. [1]", 2) == []


def test_initials_do_not_split_sentences():
    # regression: "K. Sadlon" was split into two sentences, failing a valid answer
    text = 'The authors are Jaroslav Hasek and Zdenek "Zenny" K. Sadlon [1]. Dr. Smith liked it [2].'
    assert validate(text, 2) == []


def test_refusal_is_valid():
    assert validate(REFUSAL, 2) == []


def test_missing_citations_fail():
    problems = validate("Gilead is about a preacher in Iowa.", 2)
    assert any("no citations" in p for p in problems)


def test_out_of_range_citation_fails():
    problems = validate("Gilead is about a preacher in Iowa [3].", 2)
    assert any("[3]" in p for p in problems)


def test_uncited_sentence_fails():
    problems = validate("Gilead is about a preacher [1]. It was a huge bestseller everywhere.", 2)
    assert any("no citation" in p for p in problems)


def test_list_lead_in_does_not_need_citation():
    text = "Here are two matching books:\n- Gilead, about a preacher [1]\n- Home, about a son [2]"
    assert validate(text, 2) == []


def test_answer_repairs_on_retry(monkeypatch):
    replies = iter(["Gilead is about a preacher in Iowa.", "Gilead is about a preacher in Iowa [1]."])
    monkeypatch.setattr(generate, "llm_call", lambda prompt, model, system=None: next(replies))
    ans = answer("what is gilead about", CHUNKS, "m")
    assert ans.attempts == 2 and ans.grounded and not ans.refused
    assert [(c.n, c.chunk_id, c.title) for c in ans.citations] == [(1, "b1:0000", "Gilead")]


def test_answer_fails_closed(monkeypatch):
    monkeypatch.setattr(generate, "llm_call", lambda prompt, model, system=None: "Uncited claim about books here.")
    ans = answer("q", CHUNKS, "m")
    assert ans.text == REFUSAL and ans.refused and not ans.grounded


def test_condense_is_noop_without_history():
    assert generate.condense_question("who wrote it?", [], "m") == "who wrote it?"
