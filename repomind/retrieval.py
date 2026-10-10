"""Deterministic, transparent relevance ranking of code chunks.

The retrieval function is intentionally simple and explainable (no vector DB, no
external API). It scores each chunk against the question with a weighted
combination of:

- token overlap: how many distinct question tokens appear in the chunk body,
- identifier matches: question tokens that appear as whole identifiers in code,
- path matches: question tokens that appear in the chunk's file path.

Determinism matters for a hackathon demo and for trust: the same question and
repo always produce the same ranking, and the score components are inspectable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .scanner import Chunk

# Weights for the scoring components. Tunable but fixed for determinism.
W_TOKEN_OVERLAP = 1.0
W_IDENTIFIER = 2.0
W_PATH = 3.0

# A threshold below which we consider the evidence weak and say so in the UI.
WEAK_EVIDENCE_THRESHOLD = 2.0

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Very common words that add noise to overlap scoring.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "to", "in", "on", "and", "or", "for",
    "how", "what", "where", "does", "do", "it", "this", "that", "with", "from",
    "which", "when", "why", "be", "as", "at", "by", "we", "i", "you", "code",
    "function", "method", "class", "file", "repo", "repository",
}


def tokenize(text: str) -> List[str]:
    """Lowercase identifier-style tokens, splitting camelCase and snake_case."""
    raw = _TOKEN_RE.findall(text)
    tokens: List[str] = []
    for word in raw:
        # Split snake_case
        parts = word.split("_")
        for part in parts:
            if not part:
                continue
            # Split camelCase / PascalCase into sub-words
            sub = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", part)
            if sub:
                tokens.extend(s.lower() for s in sub)
            else:
                tokens.append(part.lower())
        # Also keep the whole original token (preserves full identifier matches)
        tokens.append(word.lower())
    return tokens


def question_terms(question: str) -> set[str]:
    """Extract meaningful lowercase search terms from a question."""
    return {t for t in tokenize(question) if t not in _STOPWORDS and len(t) > 1}


@dataclass
class ScoredChunk:
    """A chunk with its relevance score and a breakdown of components."""

    chunk: Chunk
    score: float
    token_overlap: int
    identifier_matches: int
    path_matches: int

    @property
    def is_weak(self) -> bool:
        return self.score < WEAK_EVIDENCE_THRESHOLD


def score_chunk(chunk: Chunk, terms: set[str]) -> ScoredChunk:
    """Score a single chunk against the question terms deterministically."""
    if not terms:
        return ScoredChunk(chunk, 0.0, 0, 0, 0)

    body_tokens = set(tokenize(chunk.text))
    path_tokens = set(tokenize(chunk.path))

    # Whole-identifier matches: terms that appear exactly in the chunk's tokens.
    identifier_hits = terms & body_tokens
    # Token overlap counts the same set here, but we keep it distinct so the
    # weighting can emphasize identifier matches over mere presence.
    overlap = len(identifier_hits)
    path_hits = terms & path_tokens

    token_overlap = overlap
    identifier_matches = len(identifier_hits)
    path_matches = len(path_hits)

    score = (
        W_TOKEN_OVERLAP * token_overlap
        + W_IDENTIFIER * identifier_matches
        + W_PATH * path_matches
    )
    return ScoredChunk(
        chunk=chunk,
        score=float(score),
        token_overlap=token_overlap,
        identifier_matches=identifier_matches,
        path_matches=path_matches,
    )


def rank_chunks(
    chunks: List[Chunk], question: str, top_k: int = 6
) -> List[ScoredChunk]:
    """Rank chunks by relevance to the question and return the top_k scored.

    Chunks with a zero score are dropped. Ties are broken deterministically by
    file path then start line so results are stable across runs.
    """
    terms = question_terms(question)
    scored = [score_chunk(c, terms) for c in chunks]
    scored = [s for s in scored if s.score > 0]
    scored.sort(
        key=lambda s: (-s.score, s.chunk.path, s.chunk.start_line)
    )
    return scored[:top_k]


def evidence_is_weak(ranked: List[ScoredChunk]) -> bool:
    """Report whether the retrieved evidence is weak or empty."""
    if not ranked:
        return True
    return ranked[0].score < WEAK_EVIDENCE_THRESHOLD
