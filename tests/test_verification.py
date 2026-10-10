"""Tests for citation parsing and validation."""

from pathlib import Path

from repomind.verification import (
    parse_citations,
    split_valid_invalid,
    validate_citations,
)


def test_parse_citations_variants():
    text = (
        "The function divides here [calculator.py:L10-L25] and the error type "
        "is defined at [calculator.py:L1] and a range [a/b.py:L5-L5]."
    )
    cites = parse_citations(text)
    assert len(cites) == 3
    assert cites[0].path == "calculator.py"
    assert cites[0].start_line == 10
    assert cites[0].end_line == 25
    # Single line expands end to start.
    assert cites[1].start_line == 1
    assert cites[1].end_line == 1


def test_validate_valid_and_invalid(tmp_path: Path):
    src = tmp_path / "calculator.py"
    src.write_text("\n".join(f"line{i}" for i in range(1, 21)))  # 20 lines

    evidence_paths = {"calculator.py"}
    text = (
        "Valid cite [calculator.py:L1-L10]. "
        "Out of range [calculator.py:L1-L99]. "
        "Not in evidence [secret.py:L1-L2]."
    )
    validated = validate_citations(text, str(tmp_path), evidence_paths)
    valid, invalid = split_valid_invalid(validated)

    assert len(valid) == 1
    assert valid[0].citation.path == "calculator.py"
    assert valid[0].citation.start_line == 1
    assert valid[0].citation.end_line == 10

    assert len(invalid) == 2
    reasons = " ".join(v.reason for v in invalid)
    assert "outside the file" in reasons
    assert "not part of the retrieved evidence" in reasons


def test_citation_to_missing_file_is_invalid(tmp_path: Path):
    evidence_paths = {"ghost.py"}
    text = "See [ghost.py:L1-L3]."
    validated = validate_citations(text, str(tmp_path), evidence_paths)
    valid, invalid = split_valid_invalid(validated)
    assert not valid
    assert len(invalid) == 1
    assert "could not be read" in invalid[0].reason


def test_no_citations_returns_empty(tmp_path: Path):
    validated = validate_citations("No citations here.", str(tmp_path), set())
    assert validated == []
