import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Homework -- classify queries to their item type

    We have pre-enriched products assigned with an item type. Below is a skeleton query classifier for the item type. What NDCG can you achieve my changing the query side?

    Your task is to:

    1. Use the provided query ground truth for item types
    2. Change your classifier to preduct that ground truth
    3. Use the search strategy to improve NDCG

    Before you begin, just run through the notebook once. It should run. THEN

    Change the prompt and prompting strategy to try to improve precision against the ground truth.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Boilerplate

    Install deps, mount GDrive, prompt for your OpenAI Key (placed in your GDrive)
    """
    )
    return


@app.cell
def _():
    # (use marimo's built-in package management features instead) !pip install git+https://github.com/softwaredoug/cheat-at-search.git
    from cheat_at_search.data_dir import mount
    mount(use_gdrive=True)
    from cheat_at_search.search import run_strategy, graded_bm25, ndcgs, ndcg_delta, vs_ideal
    from cheat_at_search.wands_data import products, enriched_products, enriched_queries
    from cheat_at_search.cache import StoredLruCache
    from sentence_transformers import SentenceTransformer, util
    _model = SentenceTransformer('all-MiniLM-L6-v2')
    return (
        SentenceTransformer,
        StoredLruCache,
        enriched_products,
        enriched_queries,
        graded_bm25,
        ndcg_delta,
        ndcgs,
        products,
        run_strategy,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Review item types classified

    In a [previous example notebook](https://colab.research.google.com/drive/1S6GdDMN-I4wFY4obWLs0STNlT_7mKrXM) we classified items to item types. Below you can see they've been precomputed for you to use as an attribute on every product.
    """
    )
    return


@app.cell
def _(enriched_products):
    enriched_products
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Classify queries to item types

    Below is a scaffold query classifier for you to do your work.
    """
    )
    return


@app.cell
def _(SentenceTransformer, StoredLruCache):
    from pydantic import BaseModel, Field
    from typing import List, Literal
    import numpy as np
    ItemType = Literal['area rug', 'accent pillow', 'bed', 'cocktail table', 'floor & wall tile', 'entertainment center', 'kitchen mat', 'sectional', 'sofa', 'patio sofa', 'doormat', 'furniture cushion', 'wall clock', 'garden statue', 'kitchen island', 'garment rack', 'mattress pad', 'loveseat', 'armchair', 'recliner', 'coffee table', 'end table', 'tv stand', 'media console', 'bookshelf', 'bed frame', 'mattress', 'nightstand', 'dresser', 'wardrobe', 'chest of drawers', 'dining table', 'dining chair', 'bar stool', 'sideboard', 'buffet', 'bench', 'office chair', 'desk', 'filing cabinet', 'bookcase', 'patio chair', 'patio table', 'outdoor sofa', 'umbrella', 'grill', 'toolbox', 'door knob', 'door lock', 'deadbolt', 'light switch', 'outlet', 'extension cord', 'smart bulb', 'ceiling fan', 'floor lamp', 'table lamp', 'chandelier', 'rug', 'curtains', 'blinds', 'shower curtain', 'mirror', 'wall art', 'picture frame', 'clock', 'candle holder', 'vase', 'planter', 'kitchen faucet', 'sink', 'toilet', 'bathroom faucet', 'chaise lounge', 'shower head', 'plunger', 'broom', 'dustpan', 'mop', 'bucket', 'vacuum', 'trash can', 'recycling bin', 'laundry basket', 'ironing board', 'drying rack', 'cutlery', 'slow cooker', 'frying pan', 'saucepan', 'mixing bowl', 'cutting board', 'storage bin', 'shelving unit', 'no item type matches', 'unknown', 'ottoman', 'comforter', 'chair cushion', 'refrigerator', 'greenhouse', 'crown molding', 'vanity', 'flag', 'potted plant', 'basket', 'podium', 'blanket', 'anti-fatigue mat', 'serving tray']
    _model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    @StoredLruCache(maxsize=50000)
    def encode(text):
        return _model.encode(text)

    class Query(BaseModel):
        """
        Base model for search queries, containing common query attributes.
        """
        keywords: str = Field(..., description='The original search query keywords sent in as input')

    class ItemTypeQuery(Query):
        """
        Represents the item type of relevant products in a search index for this query (ie dining table, bed, etc)
        """

        @property
        def similarity(self):
            """Compare item_type to item_type_unconstrained"""
            return np.dot(encode(self.item_type), encode(self.item_type_unconstrained))
        item_type: ItemType = Field(..., description="The type of item of relevant products for the query from the provided list. Use 'no item type matches' if no item type matches the item")
    # Import MiniLM to encode via sentence transformer
        item_type_unconstrained: str = Field(..., description='The type of item of relevant products for the query, ie dining table, bed, etc')
    return ItemTypeQuery, Query, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Query prompt + enricher

    We ask for item_type and item_type_unconstrained back from the prompt
    """
    )
    return


@app.cell
def _(ItemTypeQuery, Query):
    from cheat_at_search.enrich import AutoEnricher, ProductEnricher

    item_type_enricher = AutoEnricher(
        model="openai/gpt-4.1-mini",
        system_prompt="You are a helpful furniture, hardware, and home-goods ecommerce shopping assistant that understands furniture search queries",
        response_model=ItemTypeQuery
    )

    def get_item_type_prompt(query) -> str:
        prompt = f"""
    Given a search query, what item type descripbes the relevant product?

    For 'item_type' Use 'no item type matches' if no listed item type matches the item.
    For 'item_type_unconstrained' just extract any item type

    Here's the search query to classify:

    {query}
            """
        return prompt

    # Don't provide an item type unless compelling evidence is in the search query that it matches to one of the listed item types.

    def query_enricher(query: str) -> ItemTypeQuery:
        prompt = get_item_type_prompt(Query(keywords=query))
        return item_type_enricher.enrich(prompt)


    query_enricher('sheffield pillow')
    return (query_enricher,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Measure the quality against ground truth""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Examine ground truth

    The ground truth for the queries is the pre-enriched queries, based on aggregating known query-document relationships.
    """
    )
    return


@app.cell
def _(enriched_queries):
    ground_truth = enriched_queries[['item_type_same', 'item_type_unconstrained', 'query']]
    ground_truth.loc[ground_truth['item_type_unconstrained'] == 'No Category Fits', 'item_type_unconstrained'] = 'unknown'
    ground_truth
    return (ground_truth,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Measure ground truth of classifier

    Classify all queries than compare to ground truth
    """
    )
    return


@app.cell
def _(ground_truth, query_enricher):
    import pandas as pd
    from tqdm import tqdm
    enriched_item_types = []
    for query in tqdm(ground_truth['query']):
        item_type_predicted = query_enricher(query)
        item_type_as_dict = item_type_predicted.model_dump()
        similarity = item_type_predicted.similarity
        item_type_as_dict['item_type_sim'] = similarity
        item_type_as_dict['query'] = query
        enriched_item_types.append(item_type_as_dict)

    enriched_item_types = pd.DataFrame(enriched_item_types)
    return (enriched_item_types,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Measure against ground truth

    Below we merge our predictions with the ground truth to build a datafrome holding both. We compute precision and coverage.
    """
    )
    return


@app.cell
def _(enriched_item_types, ground_truth):
    q_enriched_item_types = enriched_item_types.add_suffix('_predicted').merge(ground_truth.add_suffix('_ground_truth'),
                                                                               left_on='keywords_predicted', right_on='query_ground_truth')
    q_enriched_item_types = q_enriched_item_types.drop(columns=['query_ground_truth']).rename(columns={'keywords_predicted': 'query'})
    q_enriched_item_types
    return (q_enriched_item_types,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Measure precision + coverage

    Remember that precision is the percentage of products with predictions that are correct.

    Precision is pretty poor. What can be done to improve it?

    1. Only allow predictions where item_type_sim > a threshold, otherwise label as 'no item type matches'?
    2. Change the above prediction to allow a list?
    """
    )
    return


@app.cell
def _(q_enriched_item_types):
    THRESHOLD = -1.0
    # THRESHOLD = 0.5 <--what happens if you change this to 0.5 or some other value?
    THRESHOLD = 0.75
    q_enriched_item_types.loc[q_enriched_item_types['item_type_sim_predicted'] < THRESHOLD, 'item_type_predicted'] = 'no item type matches'
    q_enriched_item_types
    return


@app.cell
def _(enriched_item_types, q_enriched_item_types):
    q_enriched_with_prediction = q_enriched_item_types[q_enriched_item_types['item_type_predicted'] != 'no item type matches']
    precision = (q_enriched_with_prediction['item_type_predicted'] ==
                 q_enriched_with_prediction['item_type_same_ground_truth']).mean()
    coverage = len(q_enriched_with_prediction) / len(enriched_item_types)
    print(f"Precision: {precision}")
    print(f"Coverage: {coverage}")
    return (q_enriched_with_prediction,)


@app.cell
def _(q_enriched_with_prediction):
    misclassifications = q_enriched_with_prediction[q_enriched_with_prediction['item_type_predicted']
                                                    != q_enriched_with_prediction['item_type_same_ground_truth']]
    misclassifications[['query', 'item_type_predicted', 'item_type_same_ground_truth']]
    return


@app.cell
def _(q_enriched_with_prediction):
    correct_classifications = q_enriched_with_prediction[q_enriched_with_prediction['item_type_predicted']
                                                         == q_enriched_with_prediction['item_type_same_ground_truth']]
    correct_classifications[['query', 'item_type_predicted', 'item_type_same_ground_truth']]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Item Type search strategy

    Just as previously, we add a constant boost to the underlying BM25 scores when the query's item type matches the product's
    """
    )
    return


@app.cell
def _(enriched_products, np, query_enricher, run_strategy):
    from searcharray import SearchArray
    from cheat_at_search.tokenizers import snowball_tokenizer
    from cheat_at_search.strategy.strategy import SearchStrategy

    class ItemTypeSearch(SearchStrategy):

        def __init__(self, products, query_to_item_type, name_boost=9.3, description_boost=4.1, item_type_boost=100):
            super().__init__(products)
            self.index = products
            self.index['product_name_snowball'] = SearchArray.index(products['product_name'], snowball_tokenizer)
            self.index['product_description_snowball'] = SearchArray.index(products['product_description'], snowball_tokenizer)
            cat_split = products['category hierarchy'].fillna('').str.split('/')
            self.index['item_type_snowball'] = SearchArray.index(products['item_type_same'], snowball_tokenizer)
            self.query_to_item_type = query_to_item_type
            self.name_boost = name_boost
            self.description_boost = description_boost
            self.item_type_boost = item_type_boost

        def search(self, query, k=10):
            """Dumb baseline lexical search, but add a constant boost when
               the desired category or subcategory"""
            bm25_scores = np.zeros(len(self.index))
            item_type = self.query_to_item_type(query).item_type
            tokenized = snowball_tokenizer(query)
            for token in tokenized:
                bm25_scores += self.index['product_name_snowball'].array.score(token) * self.name_boost
                bm25_scores += self.index['product_description_snowball'].array.score(token) * self.description_boost
            if item_type and item_type != 'no item type matches':
                tokenized_item_type = snowball_tokenizer(item_type)
                item_type_match = np.ones(len(self.index))
                if tokenized_item_type:
                    item_type_match = self.index['item_type_snowball'].array.score(tokenized_item_type) > 0
                bm25_scores[item_type_match] += self.item_type_boost
            top_k = np.argsort(-bm25_scores)[:k]
            scores = bm25_scores[top_k]
            return (top_k, scores)
    item_type_search = ItemTypeSearch(enriched_products, query_to_item_type=query_enricher)
    graded_item_type = run_strategy(item_type_search)  # ****  # Baseline BM25 search from before  # ****  # If there's a item_type mentioned, boost that by a constant amount
    return (graded_item_type,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Analyze the results

    We observe a reasonable improvement by adding item type. If we can predict item type reasonably well on queries, we have a chance to meaningfully improve search.

    We further observe that MANY queries were improved, so the impact is broad.
    """
    )
    return


@app.cell
def _(graded_bm25, graded_item_type, ndcgs):
    ndcgs(graded_item_type).mean(), ndcgs(graded_bm25).mean()
    return


@app.cell
def _(graded_bm25, graded_item_type, ndcg_delta):
    deltas = ndcg_delta(graded_item_type, graded_bm25)
    print(len(deltas))
    deltas[deltas != 0]
    return (deltas,)


@app.cell
def _(deltas):
    deltas[deltas < -0.1]
    return


@app.cell
def _(graded_item_type):
    QUERY = 'sheffield home bath set'
    graded_item_type[graded_item_type['query'] == QUERY][['product_name', 'product_description', 'grade', 'item_type', 'score']]
    return (QUERY,)


@app.cell
def _(QUERY, graded_bm25):
    graded_bm25[graded_bm25['query'] == QUERY][['product_id', 'product_name', 'product_description', 'grade', 'score']]
    return


@app.cell
def _(products):
    products[products['product_id'] == 39056].iloc[0]
    return


@app.cell
def _(enriched_queries):
    enriched_queries
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

