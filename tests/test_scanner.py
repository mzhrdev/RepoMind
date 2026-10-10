"""Tests for safe file filtering and chunk line-range correctness."""

from pathlib import Path

from repomind import scanner
from repomind.scanner import (
    chunk_text,
    classify_file,
    is_source_file,
    scan_repository,
)


def test_excludes_lockfiles_secrets_and_binaries(tmp_path: Path):
    (tmp_path / "main.py").write_text("print('hi')\n")
    (tmp_path / "package-lock.json").write_text("{}")
    (tmp_path / ".env").write_text("SECRET=abc")
    (tmp_path / "server.key").write_text("-----BEGIN KEY-----")
    (tmp_path / "my_secret.py").write_text("token = 1")
    (tmp_path / "image.png").write_text("not really png")

    assert is_source_file(tmp_path / "main.py") is True
    assert is_source_file(tmp_path / "package-lock.json") is False
    assert is_source_file(tmp_path / ".env") is False
    assert is_source_file(tmp_path / "server.key") is False
    assert is_source_file(tmp_path / "my_secret.py") is False
    assert is_source_file(tmp_path / "image.png") is False


def test_excludes_dependency_directories(tmp_path: Path):
    (tmp_path / "app.py").write_text("x = 1\n")
    nm = tmp_path / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("module.exports = {}\n")
    git = tmp_path / ".git"
    git.mkdir()
    (git / "config").write_text("[core]\n")

    result = scan_repository(tmp_path)
    indexed_paths = {c.path for c in result.chunks}
    assert "app.py" in indexed_paths
    assert not any(p.startswith("node_modules") for p in indexed_paths)
    assert not any(p.startswith(".git") for p in indexed_paths)


def test_file_size_limit(tmp_path: Path):
    big = tmp_path / "big.py"
    big.write_text("x = 1\n" * 1000)
    assert is_source_file(big, max_bytes=10) is False
    assert is_source_file(big, max_bytes=1_000_000) is True


def test_chunk_line_ranges_small_file():
    text = "line1\nline2\nline3"
    chunks = chunk_text("a.py", text, chunk_lines=60, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 3
    assert chunks[0].text == text


def test_chunk_line_ranges_overlap():
    # 25 lines, chunk size 10, overlap 2 -> step 8
    lines = [f"l{i}" for i in range(1, 26)]
    text = "\n".join(lines)
    chunks = chunk_text("a.py", text, chunk_lines=10, overlap=2)

    # Expected starts (0-based): 0, 8, 16, 24 -> 1-based: 1, 9, 17, 25
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 10
    assert chunks[1].start_line == 9
    assert chunks[1].end_line == 18
    assert chunks[2].start_line == 17
    assert chunks[2].end_line == 25

    # Every chunk's text must match the source lines exactly.
    for ch in chunks:
        expected = "\n".join(lines[ch.start_line - 1 : ch.end_line])
        assert ch.text == expected

    # Last chunk must end exactly on the final line.
    assert chunks[-1].end_line == 25


def test_classify_file_reasons(tmp_path: Path):
    (tmp_path / "ok.py").write_text("x = 1\n")
    (tmp_path / "data.bin").write_text("x")
    (tmp_path / "package-lock.json").write_text("{}")
    (tmp_path / "server.key").write_text("k")

    assert classify_file(tmp_path / "ok.py") is None
    assert "non-source extension" in classify_file(tmp_path / "data.bin")
    assert "excluded filename" in classify_file(tmp_path / "package-lock.json")
    assert "secret or key" in classify_file(tmp_path / "server.key")


def test_scan_records_skipped_files_with_reasons(tmp_path: Path):
    (tmp_path / "app.py").write_text("x = 1\n")
    (tmp_path / "notes.bin").write_text("binary-ish")
    (tmp_path / "yarn.lock").write_text("locked")

    result = scan_repository(tmp_path)

    assert result.file_count == 1
    # Two skipped files, each with a path and a reason.
    skipped_paths = {s.path: s.reason for s in result.skipped}
    assert "notes.bin" in skipped_paths
    assert "yarn.lock" in skipped_paths
    assert "non-source extension" in skipped_paths["notes.bin"]
    assert "excluded filename" in skipped_paths["yarn.lock"]
    # skipped_count stays consistent with the list length.
    assert result.skipped_count == len(result.skipped) == 2


def test_scan_invalid_paths(tmp_path: Path):
    missing = tmp_path / "nope"
    try:
        scan_repository(missing)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass

    f = tmp_path / "file.py"
    f.write_text("x = 1\n")
    try:
        scan_repository(f)
        assert False, "expected NotADirectoryError"
    except NotADirectoryError:
        pass
