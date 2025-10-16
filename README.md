# cheat-at-search-october-cohort

Companion repo for my experiments following [Doug Turnbull](https://www.linkedin.com/in/softwaredoug/)'s
[*Cheat at Search with LLMs*](https://maven.com/softwaredoug/cheat-at-search) (Maven) course on building LLM‑based
query and content understanding.

## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Notebook Guide & Recommended Path](#notebook-guide--recommended-path)
- [Dataset](#dataset)
- [Why Marimo Instead Of Jupyter](#why-marimo-instead-of-jupyter)
- [Marimo Quickstart](#marimo-quickstart--jupyter-differences)
- [MoLab Hosted Experience](#molab-hosted-experience)
- [Environment Variables & API Keys](#environment-variables--api-keys)
- [Development](#development)
- [Resources](#resources)
- [License](#license)
- [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

- **Python 3.12+** – Required for this project
- **uv** – Fast Python package installer and resolver ([install guide](https://github.com/astral-sh/uv))
- **OpenAI API Key** – Required for LLM-based notebooks (sign up at [platform.openai.com](https://platform.openai.com/))

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/cheat-at-search-october-cohort.git
   cd cheat-at-search-october-cohort
   ```

2. **Install dependencies:**

   ```bash
   uv sync
   ```

3. **Set up environment variables:**

   Create a `.env` file in the project root:

   ```bash
   # .env
   OPENAI_API_KEY=sk-your-api-key
   ```

4. **Verify installation:**

   ```bash
   uv run marimo --version
   ```

You're now ready to explore the notebooks!

## Project Structure

```text
cheat-at-search-october-cohort/
├── notebooks/              # Marimo notebooks for experiments
│   ├── 0_AI_*.py          # Lexical search fundamentals (BM25)
│   ├── 0_Cheat_*.py       # BM25 baseline evaluation
│   ├── 2b_Cheat_*.py      # LLM query categorization experiments
│   └── ...
├── .env                    # API keys (not in git)
├── pyproject.toml         # Dependencies and project config
├── uv.lock                # Locked dependency versions
├── AGENTS.md              # Development guidelines for contributors
├── LICENSE                # MIT License
└── README.md              # This file
```

## Notebook Guide & Recommended Path

The notebooks in `notebooks/` build on one another to explore how lexical search works and how LLM-driven enrichment can improve it. If you are new to the project, work through them in the order below:

### Part 1: Lexical Search Fundamentals (Phase 1)

1. **`0_AI_Introduction_to_Lexical_and_BM25_tokenization.py` – Tokenization fundamentals**

   Introduces lexical search concepts, shows how naive tokenization fails on case/punctuation, and demonstrates how normalization fixes matches.

2. **`1_AI_Introduction_to_Lexical_and_BM25_query_tokenization.py` – Query-time control**

   Applies the same tokenizer to incoming queries, illustrates OR vs AND semantics, and connects those choices to Elasticsearch/Vespa-style query DSLs.

3. **`2_AI_Introduction_to_Lexical_and_BM25_TFIDF_scoring.py` – TF\*IDF and BM25 intuition**

   Builds custom similarity functions to illustrate term frequency, document frequency, TF\*IDF weighting, and how BM25 extends those ideas.

4. **`3_AI_Introduction_to_Lexical_and_BM25_Searching_multiple_fields.py` – Multi-field scoring**

   Compares field-centric (sum) versus term-centric (dismax) strategies, explores stemming to align fields, and shows how to blend contributions responsibly.

5. **`4_AI_Introduction_to_Lexical_and_BM25_Is_there_a_better_TFIDF_.py` – Toward BM25**

   Highlights the shortcomings of naive TF\*IDF (saturation, length bias) and evolves the scoring formula toward BM25's parameterization.

6. **`5_AI_Introduction_to_Lexical_and_BM25_BM25F_step_by_step.py` – BM25F blending**

   Demonstrates how to share document frequency across fields, apply per-field normalization, and replicate BM25F logic.

**Run Phase 1:**
```bash
cd notebooks
uv run marimo edit 0_AI_Introduction_to_Lexical_and_BM25_tokenization.py
# ... continue through notebooks 1-5 in sequence
```

### Part 2: LLM-Enhanced Search

#### Phase 2: Establish Baselines

7. **`0_Cheat_at_Search_with_LLMs_Analyze_BM25.py` – BM25 baseline evaluation**

   Runs a pure BM25 strategy on the WANDS dataset, surfaces low-NDCG queries, and establishes metrics you'll reuse when comparing experiments. This notebook provides the performance baseline against which all LLM enhancements will be measured.

8. **`2b_Cheat_at_Search_with_LLMs_Perfect_Categorization.py` – Perfect categorization (oracle)**

   Explores how perfect query categorization could improve search relevance as an upper bound for LLM-driven improvements. Uses ground-truth product categories to establish the theoretical maximum gain from categorization-based boosting.

**Run Phase 2:**
```bash
uv run marimo edit 0_Cheat_at_Search_with_LLMs_Analyze_BM25.py
uv run marimo edit 2b_Cheat_at_Search_with_LLMs_Perfect_Categorization.py
```

#### Phase 3: Query Categorization Experiments

This phase explores different approaches to LLM-based query categorization. Work through these sequentially to understand the progression from basic to advanced techniques:

9. **`2_Cheat_at_Search_with_LLMs_Query_Categories.py` – Foundation categorization**

   Introduces LLM-based query→category classification using Pydantic `Literal` types to constrain model outputs. Establishes the core category/subcategory boosting strategy and demonstrates structured LLM outputs for search enhancement.

10. **`2a_Cheat_at_Search_with_LLMs_Query_Categories_No_Category_Found.py` – Defensive classification**

    Adds "No Category Fits" and "No SubCategory Fits" options to the categorization system. Explores the precision vs recall tradeoff—avoiding harmful misclassifications is sometimes better than forcing a match. Essential for production systems where false positives degrade user experience.

11. **`2c_Cheat_at_Search_with_LLMs_Query_Categories_Fully_Qualified.py` – Fully qualified categories**

    Implements LLM-based query categorization with fully qualified category paths (e.g., "Furniture > Outdoor > Bistro Sets") for more granular understanding. Uses Pydantic's constrained types to enforce valid category hierarchies and prevent hallucination.

12. **`2d_cheat_at_search_with_llms_query_categories_list_of_categories.py` – Multi-category classification**

    Experiments with returning multiple categories per query to capture query ambiguity and multi-intent searches. Introduces multi-label evaluation using Jaccard similarity to measure overlap between predicted and ground-truth category sets.

13. **`2e_Cheat_at_Search_with_LLMs_Query_Categories_Examples.py` – Few-shot learning**

    Provides domain-specific hints and examples in prompts to improve categorization accuracy. Demonstrates few-shot learning techniques (e.g., "bistro tables are for outdoors") to guide the LLM toward domain-appropriate classifications.

14. **`2f_Cheat_at_Search_with_LLMs_Query_Categories_Fully_Qualified_Hallucinated.py` – Embedding-based resolution**

    Alternative approach: LLM generates plausible (hallucinated) categories, then uses SentenceTransformer embeddings to resolve them to real classifications. More cost-effective (gpt-4o-mini + embedding lookup) than constrained generation for large label sets. Trades constraint enforcement for semantic similarity matching.

**Run Phase 3:**
```bash
# Progressive refinements: 2 → 2a → 2c → 2d
uv run marimo edit 2_Cheat_at_Search_with_LLMs_Query_Categories.py
uv run marimo edit 2a_Cheat_at_Search_with_LLMs_Query_Categories_No_Category_Found.py
uv run marimo edit 2c_Cheat_at_Search_with_LLMs_Query_Categories_Fully_Qualified.py
uv run marimo edit 2d_cheat_at_search_with_llms_query_categories_list_of_categories.py

# Enhanced techniques: examples and embeddings
uv run marimo edit 2e_Cheat_at_Search_with_LLMs_Query_Categories_Examples.py
uv run marimo edit 2f_Cheat_at_Search_with_LLMs_Query_Categories_Fully_Qualified_Hallucinated.py
```

#### Phase 4: Query Understanding & Caching

15. **`2g_Query_to_Query_Similarity.py` – Semantic query similarity**

    Explores using embeddings (`intfloat/e5-small-v2`) for query-to-query similarity to enable semantic caching. Tests threshold-based matching to identify equivalent queries (e.g., "red tennis shoes" ≈ "tennis shoes in red") while avoiding false matches ("red shoes" ≠ "brown shoes"). Demonstrates handling of numerical attributes (sizes) in embedding space.

**Run Phase 4:**
```bash
uv run marimo edit 2g_Query_to_Query_Similarity.py
```

### Key Learning Progressions

- **2 → 2a → 2c → 2d**: Progressive refinements to query categorization—from basic classification to defensive strategies, hierarchical categories, and multi-label outputs
- **2c vs 2f**: Compare constrained generation (enforced valid categories) vs hallucinated+embedding (semantic resolution)—different tradeoffs in cost, accuracy, and scalability
- **Baseline → Oracle → Real**: Understand the gap between BM25 baseline (notebook 7), perfect categorization oracle (notebook 8), and achievable LLM improvements (notebooks 9-14)

Running them sequentially gives you the prerequisite context for each successive experiment and mirrors the enablement arc from core lexical control to LLM-assisted ranking tweaks.

## Dataset

This project uses the **WANDS (Wayfair ANnotation Dataset)**, a large-scale e-commerce search relevance dataset:

- **Size:** ~42k queries with relevance judgments on product catalog
- **Domain:** E-commerce (home goods)
- **Task:** Query-product relevance evaluation
- **Source:** Provided via the [`cheat-at-search`](https://github.com/softwaredoug/cheat-at-search) library

The dataset is automatically downloaded when you run the notebooks that require it.

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

- **What it is:** MoLab is marimo's managed workspace for running notebooks in the browser.
- **Getting started:** Visit [molab.marimo.io/notebooks](https://molab.marimo.io/notebooks) and open this project by
  pointing MoLab at your Git repository or uploading the notebook from `notebooks/`.
- **Why it's convenient:** Sessions start with the marimo runtime preinstalled so you can edit, execute, and share
  notebooks without local Python setup. MoLab persists the notebook file and deterministically replays cells for
  reproducible runs.
- **More background:** From the [launch announcement](https://marimo.io/blog/announcing-molab): MoLab emphasizes
  reproducibility (single‑definition rule, no hidden state), integrates with git‑based workflows, and supports sharing
  read‑only views or copyable notebooks via secure links.

## Environment Variables & API Keys

- **Keep secrets in `.env`.** Place `OPENAI_API_KEY` (and other keys) in the `.env` beside `pyproject.toml`; marimo loads
  this file automatically, thanks to the runtime configuration:

  ```toml
  # pyproject.toml
  [tool.marimo.runtime]
  dotenv = [".env"]
  ```

- **Example entry:**

  ```bash
  # .env
  OPENAI_API_KEY=sk-your-api-key
  ```

- **Running notebooks:** Commands such as `uv run marimo run notebooks/0_Cheat_at_Search_with_LLMs_Analyze_BM25.py` pick up the key with no extra exports, so the enrichers can authenticate.

- **Multiple environments:** If you need `.env.testing` or similar, list them under `[tool.marimo.runtime] dotenv` in `pyproject.toml` to control which files load per configuration:

  ```toml
  # pyproject.toml
  [tool.marimo.runtime]
  dotenv = [".env", ".env.testing"]
  ```

## Development

### Running Tests & Validation

```bash
# Validate notebook syntax
uv run python -m compileall notebooks/

# Execute a notebook headlessly
uv run notebooks/0_Cheat_at_Search_with_LLMs_Analyze_BM25.py

# Launch interactive mode for development
uv run marimo edit notebooks/0_Cheat_at_Search_with_LLMs_Analyze_BM25.py
```

### Contributing

This is a personal learning repository, but contributions and suggestions are welcome! Please see [AGENTS.md](AGENTS.md) for:

- Code style guidelines (PEP 8, Python 3.12+)
- Commit message conventions
- Testing and validation procedures
- Marimo cell best practices

Key points:

- Use `uv sync` to ensure dependencies are up to date
- Test notebooks with `uv run <notebook>.py` before committing
- Keep the focus on reproducibility and clear documentation

## Resources

### Course & Related Materials

- [Cheat at Search with LLMs](https://maven.com/softwaredoug/cheat-at-search) – Doug Turnbull's Maven course
- [cheat-at-search library](https://github.com/softwaredoug/cheat-at-search) – Supporting Python library
- [Doug Turnbull's LinkedIn](https://www.linkedin.com/in/softwaredoug/) – Course instructor

### Tools & Libraries

- [Marimo](https://marimo.io) – Reactive Python notebooks
- [MoLab](https://molab.marimo.io) – Marimo's hosted workspace
- [uv](https://github.com/astral-sh/uv) – Fast Python package manager
- [searcharray](https://github.com/softwaredoug/searcharray) – Array-based search library
- [PyStemmer](https://github.com/snowballstem/pystemmer) – Snowball stemming library

### Search & Information Retrieval

- [BM25 Explained](https://en.wikipedia.org/wiki/Okapi_BM25) – Wikipedia overview
- [WANDS Dataset Paper](https://github.com/wayfair/WANDS) – Original Wayfair dataset publication
- [Relevant Search](https://www.manning.com/books/relevant-search) – Book by Doug Turnbull & John Berryman

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Hendrik Reh

## Troubleshooting

### Common Issues

#### Problem: `uv sync` fails with dependency resolution errors

```bash
# Try clearing the cache
uv cache clean

# Re-run sync
uv sync
```

#### Problem: Notebooks can't find OPENAI_API_KEY

- Verify `.env` file exists in the project root (not in `notebooks/`)
- Check the key format: `OPENAI_API_KEY=sk-...`
- Confirm the key is valid at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

#### Problem: Marimo shows "module not found" errors

```bash
# Ensure you're using uv to run marimo
uv run marimo edit notebooks/your_notebook.py

# Not: marimo edit notebooks/your_notebook.py
```

#### Problem: Dataset download fails

- Check your internet connection
- The WANDS dataset is downloaded automatically on first run
- If it fails, try running the notebook again - it will resume the download

#### Problem: Jupyter-specific issues

Remember: This project uses **Marimo**, not Jupyter. If you're having issues:

- Don't use `jupyter notebook` or `jupyter lab`
- Use `uv run marimo edit` or `uv run marimo run` instead
- Marimo notebooks are Python files (`.py`), not JSON (`.ipynb`)

### Getting Help

If you encounter issues not covered here:

1. Check [Marimo documentation](https://docs.marimo.io)
2. Review [AGENTS.md](AGENTS.md) for development guidelines
3. Open an issue in this repository

---

**Happy searching!** 🔍
