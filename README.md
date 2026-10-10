# 🧭 RepoMind

**An evidence-grounded AI codebase investigator.** Point RepoMind at a local
source repository, ask a question about its implementation, and get an answer
from a local open-weight model that is required to cite its evidence — with
every citation validated against the real source.

Built for **Hacktoberfest Islamabad 2026**.

---

## Problem

Developers constantly need to understand unfamiliar code: onboarding to a new
repo, reviewing a dependency, or auditing a legacy service. General chatbots
answer confidently but hallucinate file names, invent line numbers, and give you
no way to check their claims against the actual code.

## Solution

RepoMind grounds every answer in retrieved evidence from *your* repository and
then **programmatically validates** the model's citations:

1. **Index** the repo into line-anchored chunks.
2. **Retrieve** the most relevant chunks for your question using a transparent,
   deterministic ranking function.
3. **Investigate** by sending the question and the selected evidence to a local
   open-weight model, instructing it to cite with `[path:Lstart-Lend]`.
4. **Validate** each citation: was that file in the evidence, and does the cited
   line range actually exist in the source file?

Valid and invalid citations are shown separately so you always know which parts
of the answer are traceable to real code.

## AI component

- **Core reasoning is done by an open-weight model** served through the
  **Ollama** HTTP API. No proprietary AI API is used for the core explanation.
- **Default model:** `qwen2.5-coder:3b` — a ~3B-parameter (3.09B) open-weight
  code model from Alibaba's Qwen2.5-Coder family, run locally through Ollama.
- The model only ever sees the retrieved evidence, wrapped in delimiters and
  labeled as **untrusted data**. The system prompt tells it to ignore any
  instructions embedded in repository content (prompt-injection defense).
- There is **no fake AI fallback.** If the model is unavailable, RepoMind tells
  you exactly how to configure it instead of fabricating an answer.
- RepoMind is model-agnostic: any Ollama-served open-weight model works via the
  `REPOMIND_MODEL` environment variable (e.g. a cloud model like `gemma4:cloud`
  after `ollama signin`).

### Model provenance and license

- **Model:** `qwen2.5-coder:3b` (Qwen2.5-Coder, 3B size), distributed through
  the [Ollama library](https://ollama.com/library/qwen2.5-coder) and
  [Hugging Face](https://huggingface.co/Qwen).
- **Parameter count:** ~3.09B.
- **License:** the Qwen2.5-Coder 3B weights are released under the **Qwen
  Research License** (not Apache-2.0). Note: most other sizes in the Qwen2.5
  family *are* Apache-2.0, but the 3B and 72B variants use the Qwen Research
  License — review its terms before commercial use. Verified against the
  [official Qwen2.5 release notes](https://qwenlm.github.io/blog/qwen2.5/).
- The model weights are **not** bundled with this repository; you download them
  yourself with `ollama pull`. RepoMind's own code is MIT licensed (see below).

## Architecture

```
app.py                      Streamlit UI (dark, developer-focused)
repomind/
  scanner.py                Safe scanning + line-anchored overlapping chunking
  retrieval.py              Deterministic relevance ranking (token/identifier/path)
  investigator.py           Ollama client, prompt construction, typed errors
  verification.py           Citation parsing + validation against source
tests/                      pytest suite for filtering, chunking, ranking, citations
examples/demo_repo/         Tiny synthetic repo to demo without external code
```

Pipeline: `scan_repository` → `rank_chunks` → `investigate` → `validate_citations`.

### Original engineering contributions

This is **not** a thin wrapper around a model. The meaningful original work is:

- **Safe scanner:** deterministic allow/deny filtering that excludes dependency
  dirs, lockfiles, binaries, and likely secrets; enforces a size limit; never
  follows symlinks or executes repo code; and keeps resolved paths inside the
  selected root.
- **Line-anchored chunking:** overlapping chunks that retain exact 1-based start
  and end line numbers so citations can be checked against real lines.
- **Transparent retrieval:** a deterministic relevance function combining token
  overlap, whole-identifier matches, and path matches, with camelCase/snake_case
  tokenization and an explicit weak-evidence signal.
- **Structured evidence contract:** labeled, delimited evidence with a strict
  citation format and prompt-injection-resistant framing.
- **Programmatic citation validation:** parsing `[path:Lx-Ly]` references and
  checking both evidence membership and real line-range existence, presented
  honestly as a provenance check — not a proof of correctness.

## Setup

**Prerequisites:** Python **3.11 or newer** (verified on 3.13), and
[Ollama](https://ollama.com/download) for model inference (see below).

```powershell
# 1. create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. copy the example environment file and adjust if needed
copy .env.example .env
```

On macOS/Linux use `source .venv/bin/activate` and `cp .env.example .env`.

## Model installation (Ollama)

1. Install Ollama from https://ollama.com/download.
   - On **Windows** and **macOS** the background server starts automatically
     after install. On **Linux**, start it with `ollama serve`.
   - Verify: `ollama --version`

2. **Download the default model** (~1.9 GB):

   ```bash
   ollama pull qwen2.5-coder:3b
   ```

3. Confirm the server is running and the model is present:

   ```bash
   ollama list
   # or: curl http://localhost:11434/api/tags
   ```

   Do not start a second `ollama serve` if one is already running.

Runs entirely on your machine. Without a supported GPU, responses are slower
(CPU inference) but still work.

### Alternative: cloud model (optional)

To offload compute from weak hardware, sign in and use a cloud model instead:

```bash
ollama signin                 # you do this; the app never authenticates for you
```

Then set `REPOMIND_MODEL=gemma4:cloud` in `.env`. See `.env.example` for the
direct-cloud profile (`https://ollama.com` + `OLLAMA_API_KEY`).

### Configuration

Configure via environment variables (see `.env.example`):

- `OLLAMA_HOST` (default `http://localhost:11434`)
- `REPOMIND_MODEL` (default `qwen2.5-coder:3b`)
- `REPOMIND_TIMEOUT` (seconds, default `120`)
- `OLLAMA_API_KEY` (only for calling `https://ollama.com` directly)

## Run

```bash
streamlit run app.py
```

Then in the UI: enter a repository path (the bundled `examples/demo_repo` is
pre-filled), click **Scan & Index**, pick or type a question, and click
**Investigate**. Use **Check model** in the sidebar to confirm Ollama and the
model are ready.

## Example questions (against `examples/demo_repo`)

- How does the calculator handle division by zero?
- What happens when I remove more stock than exists in the inventory?
- How does `apply_operation` dispatch operations?

## Tests

```bash
pytest
```

Covers safe file filtering, chunk line-range correctness, deterministic ranking,
and citation validation.

## Limitations

- **Keyword retrieval, not semantic search.** Ranking is lexical (token /
  identifier / path overlap). Questions that use words absent from the code may
  retrieve weak evidence; RepoMind flags this but cannot invent relevant code.
- **Citation validation checks provenance, not truth.** A valid citation means
  the model pointed at real lines that were shown to it. It does **not** prove
  the model's interpretation of those lines is correct.
- **Answer quality depends on the model.** A small 3B model is fast but less
  capable than larger models; results vary with the chosen model.
- **Context is bounded by retrieval.** Only the top-k chunks are sent to the
  model, so answers requiring repo-wide synthesis may be incomplete.
- **Text source files only.** Binaries, generated files, and excluded paths are
  not indexed by design.
- **Local only.** No authentication, no database, no code execution, no
  autonomous editing of your source.

## Security considerations

- **Repository content is treated as untrusted data.** Retrieved code is wrapped
  in delimiters and the system prompt instructs the model to ignore any
  instructions embedded in it (prompt-injection resistance).
- **No code execution.** RepoMind only reads and indexes text; it never runs,
  imports, or evaluates repository code.
- **No symlink escape.** Scanning never follows symlinks and verifies every
  resolved path stays inside the selected repository root.
- **Secret-aware scanning.** `.env` files, key/credential files, and lockfiles
  are skipped during indexing, so secrets are not fed to the model.
- **No secrets in the repo.** Configuration lives in a local `.env` (gitignored);
  only `.env.example` with placeholders is published. Set `OLLAMA_API_KEY` via
  your environment, never in committed files.
- **Citation validation is provenance, not proof.** It confirms cited lines
  exist and were shown to the model; it does not verify the claim is correct.

## Team

- GitHub: [@mzhrdev](https://github.com/mzhrdev)
- _Additional team members: to be added by the team._

## Acknowledgements

- The **Qwen2.5-Coder** model authors (Alibaba Qwen team) for the default
  open-weight model.
- [Ollama](https://ollama.com/) for local open-weight model serving.
- [Streamlit](https://streamlit.io/) for the UI framework.
- Hacktoberfest Islamabad 2026 organizers.

## Third-party dependencies

| Dependency | Purpose | License |
|------------|---------|---------|
| streamlit  | Web UI  | Apache-2.0 |
| requests   | HTTP client for the Ollama API | Apache-2.0 |
| pytest     | Test runner | MIT |

The AI model (`qwen2.5-coder:3b` by default) is served via Ollama and is **not**
bundled with this repository; it is downloaded separately with `ollama pull` and
carries its own license (Qwen Research License — see *Model provenance* above).

## License

Released under the [MIT License](LICENSE).
