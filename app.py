"""RepoMind Streamlit interface.

An evidence-grounded AI codebase investigator: index a local repo, retrieve
relevant code deterministically, ask an open-weight model about it, and validate
the model's citations against the real source.
"""

from __future__ import annotations

import streamlit as st

from repomind.investigator import (
    InvestigatorConfig,
    InvestigatorError,
    check_model_available,
    investigate,
)
from repomind.retrieval import evidence_is_weak, rank_chunks
from repomind.scanner import scan_repository
from repomind.verification import split_valid_invalid, validate_citations

st.set_page_config(
    page_title="RepoMind",
    page_icon="🧭",
    layout="wide",
)

# --- Minimal dark professional styling -------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    .rm-badge {
        display:inline-block; padding:2px 10px; border-radius:12px;
        font-size:0.8rem; font-weight:600; margin-right:6px;
    }
    .rm-ok { background:#10391f; color:#4ade80; }
    .rm-bad { background:#3b1414; color:#f87171; }
    .rm-warn { background:#3a2f12; color:#fbbf24; }
    .rm-score { color:#60a5fa; font-family:monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLE_QUESTIONS = [
    "How does the calculator handle division by zero?",
    "What happens when I remove more stock than exists in the inventory?",
    "How does apply_operation dispatch operations?",
]


def _init_state():
    st.session_state.setdefault("scan", None)
    st.session_state.setdefault("repo_root", "examples/demo_repo")
    st.session_state.setdefault("question", "")


def main():
    _init_state()
    config = InvestigatorConfig.from_env()

    st.title("🧭 RepoMind")
    st.caption(
        "Evidence-grounded AI codebase investigator. The open-weight model "
        "reasons only over retrieved code; every citation is validated against "
        "the real source."
    )

    with st.sidebar:
        st.header("Model status")
        st.write(f"**Host:** `{config.host}`")
        st.write(f"**Model:** `{config.model}`")
        if st.button("Check model", use_container_width=True):
            try:
                status = check_model_available(config)
                if status.state == "local":
                    st.success(status.message)
                elif status.state == "cloud":
                    st.info(status.message)
                else:
                    st.warning(status.message)
            except InvestigatorError as exc:
                st.error(str(exc))

        st.divider()
        st.header("Index settings")
        max_kb = st.slider("Max file size (KB)", 50, 1000, 500, step=50)
        chunk_lines = st.slider("Chunk size (lines)", 20, 120, 60, step=10)
        overlap = st.slider("Chunk overlap (lines)", 0, 40, 15, step=5)
        top_k = st.slider("Evidence chunks to retrieve", 2, 12, 6)

    # --- Repository selection + indexing ---
    st.subheader("1. Select & index a repository")
    col1, col2 = st.columns([4, 1])
    with col1:
        repo_root = st.text_input(
            "Local repository path",
            value=st.session_state["repo_root"],
            help="Absolute or relative path to a local source repository.",
        )
    with col2:
        st.write("")
        st.write("")
        scan_clicked = st.button("Scan & Index", use_container_width=True)

    if scan_clicked:
        st.session_state["repo_root"] = repo_root
        try:
            with st.spinner("Scanning repository..."):
                result = scan_repository(
                    repo_root,
                    max_bytes=max_kb * 1000,
                    chunk_lines=chunk_lines,
                    overlap=overlap,
                )
            if result.file_count == 0:
                st.warning(
                    "No indexable source files were found. The repository may be "
                    "empty or contain only excluded/binary files."
                )
                st.session_state["scan"] = None
            else:
                st.session_state["scan"] = result
                st.success(
                    f"Indexed {result.file_count} files into "
                    f"{result.chunk_count} chunks "
                    f"({result.skipped_count} skipped)."
                )
        except FileNotFoundError:
            st.error(f"Path does not exist: `{repo_root}`")
            st.session_state["scan"] = None
        except NotADirectoryError:
            st.error(f"Path is not a directory: `{repo_root}`")
            st.session_state["scan"] = None

    scan = st.session_state.get("scan")
    if scan:
        c1, c2, c3 = st.columns(3)
        c1.metric("Files indexed", scan.file_count)
        c2.metric("Chunks", scan.chunk_count)
        c3.metric("Skipped", scan.skipped_count)
        st.caption(f"Repository root: `{scan.root}`")
        _render_index_detail(scan)

    # --- Question + investigation ---
    st.subheader("2. Ask about the implementation")
    st.write("Example questions:")
    ex_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for i, example in enumerate(EXAMPLE_QUESTIONS):
        if ex_cols[i].button(example, key=f"ex_{i}", use_container_width=True):
            st.session_state["question"] = example

    question = st.text_area(
        "Your question",
        value=st.session_state.get("question", ""),
        height=90,
    )
    ask = st.button("Investigate", type="primary")

    if ask:
        if not scan:
            st.error("Index a repository first.")
            return
        if not question.strip():
            st.error("Enter a question.")
            return

        ranked = rank_chunks(scan.chunks, question, top_k=top_k)
        if not ranked:
            st.error(
                "No relevant code was found for this question. Try different "
                "keywords or re-index with a larger chunk size."
            )
            return

        if evidence_is_weak(ranked):
            st.warning(
                "⚠️ Evidence is weak: the best match scored low. The answer may "
                "be unreliable. Consider rephrasing with identifiers used in the "
                "code."
            )

        _render_evidence(ranked)

        with st.spinner(f"Asking {config.model}..."):
            try:
                result = investigate(question, ranked, config)
            except InvestigatorError as exc:
                st.error(str(exc))
                return

        st.subheader("3. Answer")
        st.markdown(result.answer)

        _render_citations(result.answer, scan.root, ranked)


def _render_index_detail(scan):
    """Show a real per-file breakdown of what was indexed.

    All values come from the ScanResult chunks already in memory, so this
    reflects exactly what RepoMind indexed and will search over.
    """
    # Aggregate chunks per file from the actual scan result.
    per_file = {}
    for ch in scan.chunks:
        info = per_file.setdefault(
            ch.path, {"chunks": 0, "min": ch.start_line, "max": ch.end_line}
        )
        info["chunks"] += 1
        info["min"] = min(info["min"], ch.start_line)
        info["max"] = max(info["max"], ch.end_line)

    rows = [
        {
            "File": path,
            "Chunks": d["chunks"],
            "Lines": d["max"],  # last line covered == file length indexed
            "Line span": f"L{d['min']}-L{d['max']}",
        }
        for path, d in sorted(per_file.items())
    ]

    with st.expander(f"Indexed files ({len(rows)})", expanded=False):
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(
            "Each file is split into overlapping chunks that keep exact line "
            "numbers; retrieval and citation validation use these ranges."
        )

    if scan.skipped:
        with st.expander(f"Skipped files ({len(scan.skipped)})", expanded=False):
            skip_rows = [
                {"File": s.path, "Reason": s.reason}
                for s in sorted(scan.skipped, key=lambda s: s.path)
            ]
            st.dataframe(skip_rows, use_container_width=True, hide_index=True)
            st.caption(
                "Files are skipped for safety or relevance: binaries and "
                "non-source extensions, oversized or unreadable files, likely "
                "secrets/keys, and lockfiles. Entire excluded directories (.git, "
                "node_modules, virtualenvs, build output) are pruned and not "
                "listed individually."
            )


def _render_evidence(ranked):
    st.subheader("Retrieved evidence")
    for i, sc in enumerate(ranked, start=1):
        c = sc.chunk
        label = (
            f"Evidence {i} — {c.path}:L{c.start_line}-L{c.end_line} "
            f"(score {sc.score:.1f})"
        )
        with st.expander(label, expanded=(i == 1)):
            st.markdown(
                f"<span class='rm-score'>token_overlap={sc.token_overlap} "
                f"identifiers={sc.identifier_matches} "
                f"path={sc.path_matches}</span>",
                unsafe_allow_html=True,
            )
            st.code(c.text)


def _render_citations(answer, repo_root, ranked):
    st.subheader("Citation validation")
    evidence_paths = {sc.chunk.path for sc in ranked}
    validated = validate_citations(answer, repo_root, evidence_paths)
    valid, invalid = split_valid_invalid(validated)

    if not validated:
        st.info("The model produced no parseable [path:Lx-Ly] citations.")
    else:
        col_v, col_i = st.columns(2)
        with col_v:
            st.markdown(
                f"<span class='rm-badge rm-ok'>Valid: {len(valid)}</span>",
                unsafe_allow_html=True,
            )
            for v in valid:
                st.write(f"✅ `{v.citation.raw}` — {v.reason}")
        with col_i:
            st.markdown(
                f"<span class='rm-badge rm-bad'>Invalid: {len(invalid)}</span>",
                unsafe_allow_html=True,
            )
            for v in invalid:
                st.write(f"❌ `{v.citation.raw}` — {v.reason}")

    st.caption(
        "Note: validation confirms a citation points at real lines that were "
        "shown to the model. It does NOT prove the model's claim about those "
        "lines is correct."
    )


if __name__ == "__main__":
    main()
