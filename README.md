# cheat-at-search-october-cohort

Companion repo for my experiments following [Doug Turnbull](https://www.linkedin.com/in/softwaredoug/)'s
[*Cheat at Search with LLMs*](https://maven.com/softwaredoug/cheat-at-search) (Maven) on building LLM‑based
query and content understanding.

## Why I Am Using [marimo](https://marimo.io) Instead Of Jupyter

Marimo better matches this project’s focus on experimentation:

- **Fast iteration without hidden state.** Marimo’s reactive execution model updates cells when dependencies change, reducing out‑of‑order surprises and keeping runs deterministic.
- **Tweak‑and‑see workflows.** It’s straightforward to add inputs/controls so you can vary parameters, prompts, and settings while observing effects inline.
- **Readable diffs for experiment review.** Notebooks are stored as plain text/code (not large `.ipynb` JSON), so changes to experiments are easy to review in git.
- **Scriptable and CI‑friendly.** The same notebooks run headless like scripts, making it simple to repeat experiments in CI to catch regressions early.

## Marimo Quickstart & Jupyter Differences

### Edit in the browser (auto‑saves to file)

  ```bash
  uv run marimo edit notebooks/analyze_bm25.py
```

### View‑only run

  ```bash
  uv run marimo run notebooks/analyze_bm25.py
  ```

### Headless execution

  ```bash
  uv run notebooks/analyze_bm25.py
  ```
## MoLab Hosted Experience

* **What it is:** MoLab is marimo’s managed workspace for running notebooks in the browser.
* **Getting started:** Visit [molab.marimo.io/notebooks](https://molab.marimo.io/notebooks) and open this project by
  pointing MoLab at your Git repository or uploading the notebook from `notebooks/`.
* **Why it’s convenient:** Sessions start with the marimo runtime preinstalled so you can edit, execute, and share
  notebooks without local Python setup. MoLab persists the notebook file and deterministically replays cells for
  reproducible runs.
* **More background:** From the [launch announcement](https://marimo.io/blog/announcing-molab): MoLab emphasizes
  reproducibility (single‑definition rule, no hidden state), integrates with git‑based workflows, and supports sharing
  read‑only views or copyable notebooks via secure links.

## Environment Variables & API Keys

* **Keep secrets in `.env`.** Place `OPENAI_API_KEY` (and other keys) in the `.env` beside `pyproject.toml`; marimo loads
  this file automatically, thanks to the runtime configuration:

  ```toml
  # pyproject.toml
  [tool.marimo.runtime]
  dotenv = [".env", ".env.testing"]
  ```
* **Example entry:**

  ```bash
  OPENAI_API_KEY=sk-your-api-key
  ```

* **Running notebooks:** Commands such as `uv run marimo run notebooks/synonyms_from_llms.py` pick up the key with no
  extra exports, so the enrichers can authenticate.
* **Multiple environments:** If you need `.env.testing` or similar, list them under `[tool.marimo.runtime] dotenv` in
  `pyproject.toml` to control which files load per configuration.
