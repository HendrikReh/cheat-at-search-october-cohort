# cheat-at-search-october-cohort

A companion repo for my experiments following [Doug Turnbull's](https://www.linkedin.com/in/softwaredoug/) [Cheat at Search with LLMs](https://maven.com/softwaredoug/cheat-at-search) (Maven) on building query and content understanding with LLMs.

## Marimo Quickstart & Jupyter Differences
- Edit notebooks with `uv run marimo edit notebooks/analyze_bm25.py`; the browser UI auto-saves changes back to the file.
- Launch in view mode via `uv run marimo run notebooks/analyze_bm25.py` or execute headlessly with `uv run notebooks/analyze_bm25.py`.
- Marimo enforces single definitions per name across cells: refining a value means assigning once (e.g., `QUERY = "desk"`), then deriving new variables (`refined_query = QUERY + " chair"`). Unlike Jupyter, reusing the same name in multiple cells fails fast to guarantee reproducible state.
