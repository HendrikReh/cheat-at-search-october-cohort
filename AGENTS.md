# Repository Guidelines

## Project Structure & Module Organization
- `notebooks/analyze_bm25.py` is the primary Marimo notebook; each `@app.cell` should expose only the variables it deliberately returns to avoid namespace collisions.
- Dependency manifests live in `pyproject.toml` and `uv.lock`; keep them in sync and under version control so contributors can reproduce the environment with `uv`.
- Generated artifacts such as `__marimo__/` and `__pycache__/` are workspace-local; avoid checking them in unless explicitly needed for reproducibility.

## Build, Test, and Development Commands
- `uv sync` – install or update the virtual environment defined by `pyproject.toml` and `uv.lock`.
- `uv run notebooks/analyze_bm25.py` – execute the Marimo app headlessly to catch syntax/import errors.
- `uv run marimo run notebooks/analyze_bm25.py` – launch the interactive notebook UI for exploratory work.
- `uv run python -m compileall notebooks/analyze_bm25.py` – quick syntax check before sharing changes.

## Coding Style & Naming Conventions
- Target Python 3.12, four-space indentation, and PEP 8 spacing for readability.
- Prefer explicit tuple returns from Marimo cells (for example, `return (strategy,)`) and prefix throwaway imports or helpers with `_` to prevent multiple-definition errors.
- Keep logging and diagnostics lightweight; replace `print` calls with structured reporting before merging.

## Testing Guidelines
- No automated suite exists yet; at minimum, validate notebooks with `uv run analyze_bm25.py` and a manual pass through the Marimo UI.
- When adding automated coverage, scaffold tests with `pytest` and run them via `uv run pytest`.
- Treat reproducible notebooks or scripts as regression checks—save representative queries such as `"desk for kids"` to demonstrate expected BM25 rankings.

## Commit & Pull Request Guidelines
- Write imperative, concise commit subjects (e.g., `Refine BM25 tokenizer flow`) and include focused bodies when explaining rationale or trade-offs.
- Reference relevant issues or discussion threads, and attach screenshots or query outputs when changes affect ranking behavior.
- Describe verification steps (commands run, datasets used) in the pull request to help reviewers reproduce results quickly.

## Agent-Specific Notes
- Favor additive edits with `apply_patch`; never discard user-provided changes or rewrite `uv.lock` without confirmation.
- Validate changes in a clean shell session using the commands above, and call out any assumptions about external datasets or credentials.
