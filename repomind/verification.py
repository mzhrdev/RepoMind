"""Programmatic citation validation.

The model is instructed to cite evidence as [path/to/file.py:L10-L25]. This
module parses those citations out of the model's free text and checks two things
for each one:

1. Was the cited file actually part of the retrieved evidence? (A citation to a
   file the model never saw is unsupported.)
2. Does the cited line range exist within the real source file on disk?

Important honesty note: passing validation means a citation *points at real
lines that were shown to the model*. It does NOT prove the model's claim about
those lines is correct. The UI must present this as a provenance check, not a
truth guarantee.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Matches [path:L10-L25] or [path:L10] (single line).
_CITATION_RE = re.compile(
    r"\[([^\[\]:]+?):L(\d+)(?:-L?(\d+))?\]"
)


@dataclass(frozen=True)
class Citation:
    """A parsed citation reference from the model output."""

    raw: str
    path: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class ValidatedCitation:
    """A citation annotated with validation outcome and reason."""

    citation: Citation
    is_valid: bool
    reason: str


def parse_citations(text: str) -> List[Citation]:
    """Extract all [path:Lx-Ly] citations from model text, in order."""
    citations: List[Citation] = []
    for match in _CITATION_RE.finditer(text):
        raw = match.group(0)
        path = match.group(1).strip()
        start = int(match.group(2))
        end = int(match.group(3)) if match.group(3) else start
        if end < start:
            start, end = end, start
        citations.append(Citation(raw=raw, path=path, start_line=start, end_line=end))
    return citations


def _file_line_count(repo_root: str, rel_path: str) -> int | None:
    """Return the number of lines in a repo-relative file, or None if unreadable.

    Resolves the path and confirms it stays within repo_root (no traversal).
    """
    root = Path(repo_root).resolve()
    target = (root / rel_path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    if not target.is_file():
        return None
    try:
        text = target.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        try:
            text = target.read_text(encoding="latin-1")
        except (UnicodeDecodeError, OSError):
            return None
    return len(text.splitlines())


def validate_citations(
    text: str,
    repo_root: str,
    evidence_paths: set[str],
) -> List[ValidatedCitation]:
    """Validate every citation in text against evidence and real source files.

    evidence_paths is the set of repo-relative paths that were actually sent to
    the model. A citation is valid only if its file was in the evidence AND the
    line range exists in the on-disk file.
    """
    results: List[ValidatedCitation] = []
    for citation in parse_citations(text):
        if citation.path not in evidence_paths:
            results.append(
                ValidatedCitation(
                    citation=citation,
                    is_valid=False,
                    reason="Cited file was not part of the retrieved evidence.",
                )
            )
            continue

        line_count = _file_line_count(repo_root, citation.path)
        if line_count is None:
            results.append(
                ValidatedCitation(
                    citation=citation,
                    is_valid=False,
                    reason="Cited file could not be read from the repository.",
                )
            )
            continue

        if citation.start_line < 1 or citation.end_line > line_count:
            results.append(
                ValidatedCitation(
                    citation=citation,
                    is_valid=False,
                    reason=(
                        f"Line range L{citation.start_line}-L{citation.end_line} "
                        f"is outside the file (1-{line_count})."
                    ),
                )
            )
            continue

        results.append(
            ValidatedCitation(
                citation=citation,
                is_valid=True,
                reason=(
                    f"File was in evidence and lines "
                    f"L{citation.start_line}-L{citation.end_line} exist."
                ),
            )
        )
    return results


def split_valid_invalid(
    validated: List[ValidatedCitation],
) -> tuple[List[ValidatedCitation], List[ValidatedCitation]]:
    """Partition validated citations into (valid, invalid) lists."""
    valid = [v for v in validated if v.is_valid]
    invalid = [v for v in validated if not v.is_valid]
    return valid, invalid
