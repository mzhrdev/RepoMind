"""Safe repository scanning and indexing.

This module walks a selected repository, filters out files that are unsafe or
irrelevant to index (binaries, dependency directories, lockfiles, likely
secrets), reads text source files, and splits them into overlapping chunks that
retain exact 1-based start and end line numbers.

Design notes / original engineering:
- Filtering is deterministic and transparent (plain allow/deny rules).
- Chunking preserves exact line provenance so later citation validation can
  check a model's [path:Lx-Ly] references against real source lines.
- We never execute repository code and never follow symlinks, so a malicious or
  broken repo cannot escape the selected root or run arbitrary code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

# Directories we never descend into.
EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "dist",
    "build",
    "out",
    "target",
    "bin",
    "obj",
    ".next",
    ".nuxt",
    "coverage",
    ".tox",
    "site-packages",
}

# File extensions we treat as indexable text source.
SOURCE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cc",
    ".cs",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".sh",
    ".bash",
    ".sql",
    ".html",
    ".css",
    ".scss",
    ".vue",
    ".md",
    ".rst",
    ".txt",
    ".toml",
    ".cfg",
    ".ini",
    ".yaml",
    ".yml",
    ".json",
}

# Exact filenames we skip: lockfiles, generated files, and likely secrets.
EXCLUDED_FILENAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "pipfile.lock",
    "cargo.lock",
    "composer.lock",
    "gemfile.lock",
    "go.sum",
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    "credentials",
    "credentials.json",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}

# Filename substrings / suffixes that strongly suggest secrets or keys.
SECRET_SUFFIXES = (".pem", ".key", ".pfx", ".p12", ".keystore", ".crt")
SECRET_SUBSTRINGS = ("secret", "credential", "private_key", "privatekey")

# Default maximum file size to index (bytes).
DEFAULT_MAX_FILE_BYTES = 500_000

# Chunking defaults (in lines).
DEFAULT_CHUNK_LINES = 60
DEFAULT_CHUNK_OVERLAP = 15


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of a source file with exact line provenance."""

    path: str          # repo-relative POSIX path
    start_line: int    # 1-based, inclusive
    end_line: int      # 1-based, inclusive
    text: str


@dataclass(frozen=True)
class SkippedFile:
    """A file that was not indexed, with a human-readable reason."""

    path: str      # repo-relative POSIX path when possible, else the raw path
    reason: str


@dataclass
class ScanResult:
    """Outcome of scanning a repository."""

    root: str
    chunks: List[Chunk] = field(default_factory=list)
    file_count: int = 0
    skipped: List[SkippedFile] = field(default_factory=list)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)


def is_excluded_dir(name: str) -> bool:
    """Return True if a directory name should never be descended into."""
    return name in EXCLUDED_DIRS


def classify_file(
    path: Path, max_bytes: int = DEFAULT_MAX_FILE_BYTES
) -> str | None:
    """Return None if the file is safe to index, else a reason it was skipped.

    Checks are ordered from cheapest/most specific to broadest so the reason is
    the most meaningful one (e.g. "likely secret" before "not a source file").
    """
    name = path.name.lower()

    # Never follow symlinks (prevents escaping the repo root).
    if path.is_symlink():
        return "symlink (not followed)"
    if not path.is_file():
        return "not a regular file"

    if name in EXCLUDED_FILENAMES:
        return "excluded filename (lockfile / secret / generated)"
    if any(name.endswith(sfx) for sfx in SECRET_SUFFIXES):
        return "likely secret or key file"
    if any(sub in name for sub in SECRET_SUBSTRINGS):
        return "filename suggests credentials/secret"

    if path.suffix.lower() not in SOURCE_EXTENSIONS:
        suffix = path.suffix.lower() or "(none)"
        return f"non-source extension {suffix}"

    try:
        size = path.stat().st_size
    except OSError:
        return "could not stat file"
    if size > max_bytes:
        return f"too large ({size} bytes > {max_bytes} limit)"

    return None


def is_source_file(path: Path, max_bytes: int = DEFAULT_MAX_FILE_BYTES) -> bool:
    """True if a path is a safe, relevant text source file to index.

    Thin wrapper over classify_file for callers that only need a boolean.
    """
    return classify_file(path, max_bytes=max_bytes) is None


def _rel_path(root: Path, candidate: Path) -> str:
    """Best-effort repo-relative POSIX path, falling back to the file name."""
    try:
        return candidate.resolve().relative_to(root).as_posix()
    except (ValueError, OSError):
        return candidate.name


def iter_source_files(
    root: Path, max_bytes: int = DEFAULT_MAX_FILE_BYTES
) -> Iterable[Path]:
    """Yield safe source files under root, skipping excluded dirs and symlinks."""
    for candidate, reason in walk_files(root, max_bytes=max_bytes):
        if reason is None:
            yield candidate


def walk_files(
    root: Path, max_bytes: int = DEFAULT_MAX_FILE_BYTES
) -> Iterable[tuple[Path, str | None]]:
    """Walk the repo yielding (path, reason) for every file encountered.

    reason is None for files that pass filtering (should be indexed) and a
    human-readable string for files that are skipped. Files under excluded
    directories are pruned entirely and are not reported individually.
    """
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Prune excluded directories in place so os.walk does not descend.
        dirnames[:] = [d for d in dirnames if not is_excluded_dir(d)]
        for filename in filenames:
            candidate = Path(dirpath) / filename
            # Ensure the resolved path stays inside root (defense in depth).
            try:
                resolved = candidate.resolve()
                resolved.relative_to(root)
            except (ValueError, OSError):
                yield candidate, "resolves outside the repository root"
                continue
            yield candidate, classify_file(candidate, max_bytes=max_bytes)


def read_text(path: Path) -> str | None:
    """Read a text file defensively; return None if it cannot be decoded."""
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        try:
            return path.read_text(encoding="latin-1")
        except (UnicodeDecodeError, OSError):
            return None


def chunk_text(
    rel_path: str,
    text: str,
    chunk_lines: int = DEFAULT_CHUNK_LINES,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Chunk]:
    """Split text into overlapping chunks with exact 1-based line ranges.

    A file shorter than chunk_lines becomes a single chunk spanning the whole
    file. Overlap keeps context continuous across chunk boundaries.
    """
    if chunk_lines <= 0:
        raise ValueError("chunk_lines must be positive")
    if overlap < 0 or overlap >= chunk_lines:
        raise ValueError("overlap must be >= 0 and < chunk_lines")

    lines = text.splitlines()
    if not lines:
        return []

    step = chunk_lines - overlap
    chunks: List[Chunk] = []
    start = 0  # 0-based index into lines
    total = len(lines)

    while start < total:
        end = min(start + chunk_lines, total)
        chunk_body = "\n".join(lines[start:end])
        chunks.append(
            Chunk(
                path=rel_path,
                start_line=start + 1,   # convert to 1-based inclusive
                end_line=end,           # end is already the inclusive last line
                text=chunk_body,
            )
        )
        if end >= total:
            break
        start += step

    return chunks


def scan_repository(
    root: str | os.PathLike[str],
    max_bytes: int = DEFAULT_MAX_FILE_BYTES,
    chunk_lines: int = DEFAULT_CHUNK_LINES,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> ScanResult:
    """Scan a repository into a ScanResult of line-anchored chunks.

    Raises FileNotFoundError if root does not exist and NotADirectoryError if it
    is not a directory, so callers can surface clear UI errors.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    root_path = root_path.resolve()
    result = ScanResult(root=str(root_path))

    for file_path, reason in walk_files(root_path, max_bytes=max_bytes):
        rel = _rel_path(root_path, file_path)

        if reason is not None:
            result.skipped.append(SkippedFile(path=rel, reason=reason))
            continue

        text = read_text(file_path)
        if text is None:
            result.skipped.append(
                SkippedFile(path=rel, reason="unreadable / not valid text")
            )
            continue

        file_chunks = chunk_text(
            rel, text, chunk_lines=chunk_lines, overlap=overlap
        )
        if file_chunks:
            result.file_count += 1
            result.chunks.extend(file_chunks)
        else:
            result.skipped.append(
                SkippedFile(path=rel, reason="empty file (no lines)")
            )

    return result
