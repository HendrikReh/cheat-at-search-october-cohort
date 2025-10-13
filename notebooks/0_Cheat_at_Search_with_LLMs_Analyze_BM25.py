import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Compute BM25 baseline and view NDCG

    This notebook establishes the BM25 control experiment for the rest of the series. We run a straightforward lexical strategy on the WANDS (Wayfair Annotated Dataset for Search) judgments, inspect the ranked results, and capture metrics we can compare future experiments against.

    **You are:** a search practitioner or ML engineer who wants reproducible numbers before shipping retrieval changes.

    **You will:** load the dataset, execute the baseline strategy, explore low-performing queries, and record key metrics (NDCG, delta vs ideal) that feed later notebooks.

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Boilerplate

    This project ships utilities for mounting the WANDS dataset and reading configuration from Drive. Run the next cell once to ensure your environment is ready (data directory available, API keys loaded). If you are not using Google Drive you can swap in a local path when calling `mount`.
    """
    )
    return


@app.cell
def _():
    # (use marimo's built-in package management features instead) !pip install git+https://github.com/softwaredoug/cheat-at-search.git
    from cheat_at_search.data_dir import mount
    mount(use_gdrive=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Introducing our baseline search strategy

    `BM25Search` mirrors a minimal production setup:

    - **Indexing:** we create Snowball-tokenized indexes for product name and description.
    - **Scoring:** for each query token we add the BM25 score from both fields, weighting name higher than description.
    - **Output:** we return the top-k product ids and their BM25 scores.

    Keeping the strategy intentionally simple gives us a stable baseline before layering on LLM-assisted features.
    """
    )
    return


@app.cell
def _():
    from cheat_at_search.tokenizers import snowball_tokenizer

    snowball_tokenizer("red couches for two forts")
    return (snowball_tokenizer,)


@app.cell
def _(snowball_tokenizer):
    from searcharray import SearchArray
    import numpy as np
    from cheat_at_search.strategy.strategy import SearchStrategy

    class BM25Search(SearchStrategy):

        def __init__(self, products, name_boost=9.3, description_boost=4.1):
            super().__init__(products)
            self.index = products
            self.name_boost = name_boost
            self.description_boost = description_boost
            self.index['product_name_snowball'] = SearchArray.index(products['product_name'], snowball_tokenizer)
            self.index['product_description_snowball'] = SearchArray.index(products['product_description'], snowball_tokenizer)  # ***
      # Remember the boosts for later
        def search(self, query, k=10):
            """Dumb baseline lexical search"""
            tokenized = snowball_tokenizer(query)
            print(tokenized)  # ***
            bm25_scores = np.zeros(len(self.index))  # Index product name and product description (via snowball)
            for token in tokenized:
                print(token)
                bm25_scores = bm25_scores + self.index['product_name_snowball'].array.score(token) * self.name_boost
                bm25_scores = bm25_scores + self.index['product_description_snowball'].array.score(token) * self.description_boost
            top_k = np.argsort(-bm25_scores)[:k]
            scores = bm25_scores[top_k]
            return (top_k, scores)  # ***  # Tokenize the query  # ***  # Initialize scores to 0 (an array with a float for eac doc)  # ***  # Search one token at a time, in each field, and add each docs BM25  # score to the bm25_scores  # ***  # Return top k row indices
    return (BM25Search,)


@app.cell
def _(BM25Search, products):
    strategy = BM25Search(products)
    return (strategy,)


@app.cell
def _(strategy):
    top_k, scores = strategy.search("desk for kids")
    top_k
    return scores, top_k


@app.cell
def _(products, scores, top_k):
    top_k_results = products.iloc[top_k]
    top_k_results['score'] = scores
    top_k_results[['product_name', 'product_description']]
    return


@app.cell
def _(strategy):
    from cheat_at_search.search import run_strategy
    graded_bm25 = run_strategy(strategy)
    return (graded_bm25,)


@app.cell
def _(graded_bm25):
    graded_bm25.groupby('query')['ndcg'].mean().sort_values(ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Metric helpers

    The helper functions below keep the evaluation workflow ergonomic:

    - `ndcgs(results)` returns a per-query NDCG series.
    - `ndcg_delta(experiment, control)` highlights wins and losses relative to baseline.
    - `vs_ideal(results)` aligns any run with the assessor-graded ideal ordering for spot checks.

    Caching these utilities saves us from rerunning the full pipeline every time we tweak a query.
    """
    )
    return


@app.cell
def _():
    from cheat_at_search.search import ndcgs, ndcg_delta, vs_ideal
    return ndcgs, vs_ideal


@app.cell
def _(graded_bm25):
    graded_bm25[['query', 'rank', 'product_name', 'grade']]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Inspect low-NDCG queries

    Sorting by NDCG surfaces the queries where vanilla BM25 underperforms. Use this view to spot vocabulary mismatches, missing attributes, or taxonomy issues before experimenting with improvements.
    """
    )
    return


@app.cell
def _(graded_bm25, ndcgs):
    ndcgs(graded_bm25).sort_values().head(20)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Compare BM25 vs the ideal ordering

    `vs_ideal` overlays each query's BM25 ranking with the assessor-approved ideal list. Investigate large gaps here to build intuition about what the baseline is missing (e.g., wrong category, missing synonyms).
    """
    )
    return


@app.cell
def _(graded_bm25, vs_ideal):
    # podium with locking cabinet	-> item type

    QUERY = "star wars rug"
    bm25_vs_ideal = vs_ideal(graded_bm25)
    bm25_vs_ideal[bm25_vs_ideal['query'] == QUERY]
    return QUERY, bm25_vs_ideal


@app.cell
def _(QUERY):
    from cheat_at_search.wands_data import labeled_query_products

    RELEVANT = 2
    MEH = 1
    labeled_query_products[(labeled_query_products['query'] == QUERY) &
                           (labeled_query_products['grade'] == MEH)][['query', 'product_id', 'category hierarchy']]
    return


@app.cell
def _():
    from cheat_at_search.wands_data import products

    products[products['product_id'] == 6243]
    return (products,)


@app.cell
def _(products):
    products['product_class'].value_counts().head(50)
    return


@app.cell
def _(bm25_vs_ideal):
    QUERY_1 = 'white abstract'
    bm25_vs_ideal[bm25_vs_ideal['query'] == QUERY_1]
    return


@app.cell
def _(bm25_vs_ideal):
    QUERY_2 = 'star wars rug'
    bm25_vs_ideal[bm25_vs_ideal['query'] == QUERY_2]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---

    ### Key takeaways

    - BM25 provides a transparent lexical baseline; always record its metrics before layering in new techniques.
    - Per-query NDCG views reveal which intents suffer from pure lexical matching.
    - Comparing against the assessor ideal exposes systematic gaps (category mismatches, missing synonyms) to target next.

    ### Next steps

    - Head to `1_AI_Introduction_to_Lexical_and_BM25_query_tokenization.py` or `2_AI_Introduction_to_Lexical_and_BM25_TFIDF_scoring.py` to deepen your lexical intuition.
    - When you prototype enhancements (synonyms, category boosts), reuse the `ndcg_delta` and `vs_ideal` helpers from this notebook to measure impact.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
