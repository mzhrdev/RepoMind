"""Tests for deterministic relevance ranking."""

from repomind.retrieval import (
    evidence_is_weak,
    question_terms,
    rank_chunks,
    tokenize,
)
from repomind.scanner import Chunk


def test_tokenize_splits_camel_and_snake():
    tokens = set(tokenize("applyOperation calculate_total"))
    assert "apply" in tokens
    assert "operation" in tokens
    assert "calculate" in tokens
    assert "total" in tokens
    # Whole identifiers retained too.
    assert "applyoperation" in tokens


def test_question_terms_drop_stopwords():
    terms = question_terms("How does the divide function handle zero")
    assert "divide" in terms
    assert "zero" in terms
    assert "the" not in terms
    assert "how" not in terms


def _chunk(path, text, start=1, end=10):
    return Chunk(path=path, start_line=start, end_line=end, text=text)


def test_ranking_prefers_relevant_chunk():
    chunks = [
        _chunk("calculator.py", "def divide(a, b):\n    if b == 0:\n        raise CalculatorError"),
        _chunk("unrelated.py", "def greet(name):\n    return 'hi'"),
    ]
    ranked = rank_chunks(chunks, "how does divide handle zero", top_k=5)
    assert ranked
    assert ranked[0].chunk.path == "calculator.py"
    assert ranked[0].score > 0


def test_path_match_boosts_score():
    body = "def helper():\n    return 1"
    chunks = [
        _chunk("inventory.py", body),
        _chunk("other.py", body),
    ]
    ranked = rank_chunks(chunks, "inventory", top_k=5)
    assert ranked
    assert ranked[0].chunk.path == "inventory.py"


def test_ranking_is_deterministic():
    chunks = [
        _chunk("b.py", "divide zero error"),
        _chunk("a.py", "divide zero error"),
    ]
    r1 = rank_chunks(chunks, "divide zero error", top_k=5)
    r2 = rank_chunks(chunks, "divide zero error", top_k=5)
    assert [c.chunk.path for c in r1] == [c.chunk.path for c in r2]
    # Tie broken by path: a.py before b.py.
    assert r1[0].chunk.path == "a.py"


def test_weak_evidence_detection():
    assert evidence_is_weak([]) is True
    chunks = [_chunk("x.py", "something entirely different content here")]
    ranked = rank_chunks(chunks, "nonexistent terminology xyzzy", top_k=5)
    assert evidence_is_weak(ranked) is True
