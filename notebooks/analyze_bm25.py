import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Compute BM25 baseline and view NDCG

    This notebook just does a basic analysis of poor performing queries with a BM25 baseline against the WANDS (Wayfair Annotated Dataset) search query dataset
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Introducing our baseline search strategy

    Here is a very basic search strategy that executes a naive BM25 search. Notice

    * In the constructor we initialize some weights (name and body weights)
    * In `search` we search each field, summing scores as we go
    """
    )
    return


@app.cell
def _():
    from cheat_at_search.tokenizers import snowball_tokenizer as _snowball_tokenizer

    _snowball_tokenizer("red couches for two forts")
    return


@app.cell
def _():
    from cheat_at_search.wands_data import products
    return (products,)


@app.cell
def _():
    from searcharray import SearchArray
    from cheat_at_search.tokenizers import snowball_tokenizer
    import numpy as np
    from cheat_at_search.strategy.strategy import SearchStrategy


    class BM25Search(SearchStrategy):
        def __init__(self, products,
                    name_boost=9.3,
                    description_boost=4.1):
            super().__init__(products)
            self.index = products
            # ***
            # Remember the boosts for later
            self.name_boost = name_boost
            self.description_boost = description_boost

            # ***
            # Index product name and product description (via snowball)
            self.index['product_name_snowball'] = SearchArray.index(
                products['product_name'], snowball_tokenizer)
            self.index['product_description_snowball'] = SearchArray.index(
                products['product_description'], snowball_tokenizer)

        def search(self, query, k=10):
            """Dumb baseline lexical search"""
            # ***
            # Tokenize the query
            tokenized = snowball_tokenizer(query)
            print(tokenized)

            # ***
            # Initialize scores to 0 (an array with a float for eac doc)
            bm25_scores = np.zeros(len(self.index))

            # ***
            # Search one token at a time, in each field, and add each docs BM25
            # score to the bm25_scores
            for token in tokenized:
                print(token)
                bm25_scores += self.index['product_name_snowball'].array.score(token) * self.name_boost
                bm25_scores += self.index['product_description_snowball'].array.score(
                    token) * self.description_boost

            # ***
            # Return top k row indices
            top_k = np.argsort(-bm25_scores)[:k]
            scores = bm25_scores[top_k]
            return top_k, scores
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
    top_k_results = products.iloc[top_k].copy()
    top_k_results['score'] = scores
    top_k_results[['product_name', 'product_description', 'score']]
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
    ## Shortcuts (saved baseline, etc)

    Helpers to compute ndcg of each query and other comparison functions
    """
    )
    return


@app.cell
def _(graded_bm25):
    graded_bm25[['query', 'rank', 'product_name', 'grade']]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### View random Queries in WANDS dataset with BM25""")
    return


@app.cell
def _():
    from cheat_at_search.search import ndcgs, ndcg_delta, vs_ideal
    return ndcgs, vs_ideal


@app.cell
def _(graded_bm25, ndcgs):
    ndcgs(graded_bm25).sort_values().head(20)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### View random Queries in WANDS dataset with BM25""")
    return


@app.cell
def _(graded_bm25, ndcgs):
    ndcgs(graded_bm25).sort_values().head(20)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Compare BM25 vs the 'ideal' search results""")
    return


@app.cell
def _(graded_bm25, vs_ideal):
    # podium with locking cabinet	-> item type

    QUERY1 = "star wars rug"
    bm25_vs_ideal = vs_ideal(graded_bm25)
    bm25_vs_ideal[bm25_vs_ideal['query'] == QUERY1]
    return (bm25_vs_ideal,)


@app.cell
def _(QUERY):
    from cheat_at_search.wands_data import labeled_query_products

    RELEVANT = 2
    MEH = 1
    labeled_query_products[(labeled_query_products['query'] == QUERY) &
                           (labeled_query_products['grade'] == MEH)][['query', 'product_id', 'category hierarchy']]
    return


@app.cell
def _(products):
    products[products['product_id'] == 6243]
    return


@app.cell
def _(products):
    products['product_class'].value_counts().head(50)
    return


@app.cell
def _(bm25_vs_ideal):
    QUERY2 = "white abstract"
    bm25_vs_ideal[bm25_vs_ideal['query'] == QUERY2]
    return


@app.cell
def _(bm25_vs_ideal):
    QUERY3 = "star wars rug"
    bm25_vs_ideal[bm25_vs_ideal['query'] == QUERY3]
    return


if __name__ == "__main__":
    app.run()
