import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Generate query synonyms with an LLM

    <small>
    (from <a href="http://maven.com/softwaredoug/cheat-at-search">Cheat at Search with LLMs</a> training course by Doug Turnbull.)
    </small>

    Many search teams experimented with large language models by asking a simple question: can a model invent synonyms that make lexical search better? This notebook recreates that experiment end to end so we can measure the trade offs ourselves.

    The flow we follow:

    - load the baseline BM25 results for the WANDS dataset,
    - call an LLM to generate synonym phrases for each shopper query,
    - inject those phrases into a lightweight lexical strategy, and
    - compare ranking quality (NDCG) against the original BM25 run.

    Treat this as a tour through the infrastructure we will reuse in later lessons: structured LLM calls, SearchArray indexing, and evaluation helpers.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Import helpers

    We lean on utility functions from `cheat_at_search.search` throughout the notebook:

    - `run_strategy` executes a search strategy against every WANDS query and returns graded results.
    - `graded_bm25` is the cached baseline run that we will compare against.
    - `ndcgs` computes per-query NDCG so we can summarize performance.
    - `ndcg_delta` highlights wins and losses between two result sets.
    - `vs_ideal` aligns any result set against the human graded ideal ordering.

    Keeping these helpers in one place keeps the rest of the notebook focused on how synonyms alter ranking, not on boilerplate evaluation code.
    """
    )
    return


@app.cell
def _():
    from cheat_at_search.search import run_strategy, graded_bm25, ndcgs, ndcg_delta, vs_ideal
    return graded_bm25, ndcg_delta, ndcgs, run_strategy, vs_ideal


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Import WANDS data

    The [Wayfair Annotated Dataset](https://github.com/wayfair/WANDS) (WANDS) pairs 480 real furniture queries with roughly 45k products and human relevance grades. Scores range from 0 (not relevant) to 2 (exact fit). We treat this dataset as our regression test bench: every strategy we build runs against the same queries so we can compare apples to apples.

    The next cell simply loads the product catalog into a dataframe so you can inspect the fields available for indexing.
    """
    )
    return


@app.cell
def _():
    from cheat_at_search.wands_data import products

    products
    return (products,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Synonym generation

    Our goal is to map each shopper query to a list of synonymous phrases that might appear in product data. The sections below define the structured response we expect from the LLM and the helper that will call it.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Pydantic models for structured output

    We rely on [Pydantic](https://docs.pydantic.dev/latest/) to describe the payload we want back from the LLM. By handing the model a schema we:

    - document the fields we need (`keywords`, `phrase`, `synonyms`),
    - give the LLM descriptions it can use to stay on task, and
    - let the client library validate every response before we touch it.

    OpenAI, Ollama, Gemini, and others all expose structured output modes. The mechanics differ slightly, but the pattern is the same: define a Pydantic model, pass it with the request, and get a validated object back instead of fragile JSON.
    """
    )
    return


@app.cell
def _():
    from pydantic import BaseModel, Field
    from typing import List
    from cheat_at_search.enrich import AutoEnricher


    class Query(BaseModel):
        """
        Base model for search queries, containing common query attributes.
        """
        keywords: str = Field(
            ...,
            description="The original search query keywords sent in as input"
        )


    class SynonymMapping(BaseModel):
        """
        Model for mapping phrases in the query to equivalent phrases or synonyms.
        """
        phrase: str = Field(
            ...,
            description="The original phrase from the query"
        )
        synonyms: List[str] = Field(
            ...,
            description="List of synonyms or equivalent phrases for the original phrase"
        )


    class QueryWithSynonyms(Query):
        """
        Extended model for search queries that includes synonyms for keywords.
        Inherits from the base Query model.
        """
        synonyms: List[SynonymMapping] = Field(
            ...,
            description="Mapping of phrases in the query to equivalent phrases or synonyms"
        )
    return AutoEnricher, QueryWithSynonyms


@app.cell
def _(QueryWithSynonyms):
    QueryWithSynonyms.model_json_schema()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Synonym generation code

    The `AutoEnricher` helper centralizes LLM calls for the course. When we create an instance we specify:

    - `model`: which OpenAI model tier to call (`gpt-4.1-nano` here to keep costs down),
    - `system_prompt`: a short instruction block that frames the task, and
    - `response_model`: the Pydantic schema we just defined.

    The `get_prompt` helper formats each shopper query into a short instruction string. Calling `syn_enricher.enrich(...)` returns a `QueryWithSynonyms` object that already passed validation, so downstream code can assume every field exists.
    """
    )
    return


@app.cell
def _(AutoEnricher, QueryWithSynonyms):
    syn_enricher = AutoEnricher(model="openai/gpt-4.1-nano",
                                system_prompt="You are a helpful AI assistant extracting synonyms from queries.",
                                response_model=QueryWithSynonyms)

    def get_prompt(query: str):
        prompt = f"""
            Extract synonyms from the following query that will help us find relevant products for the query.

            {query}
        """

        return prompt

    print(get_prompt("rack glass"))
    return get_prompt, syn_enricher


@app.cell
def _(get_prompt, syn_enricher):
    def query_to_syn(query: str):
        return syn_enricher.enrich(get_prompt(query))

    query_to_syn("foldout blue ugly love seat")
    return (query_to_syn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Snowball tokenizer

    A consistent tokenizer is critical when combining query text and synonym phrases. We use the shared `snowball_tokenizer` helper so both the original query and any generated synonym phrases collapse to the same stems before scoring.
    """
    )
    return


@app.cell
def _():
    from cheat_at_search.tokenizers import snowball_tokenizer
    snowball_tokenizer("fancy furniture")
    return (snowball_tokenizer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Build a SearchStrategy -- enrich, index, search

    `SearchStrategy` is the lightweight harness we use to mimic a production retrieval stack. The `SynonymSearch` subclass does three things:

    1. **Indexing**: during `__init__` we create Snowball token indexes for product name and description using [SearchArray](http://github.com/softwaredoug/search-array). These act like in-memory BM25 postings lists.
    2. **Baseline scoring**: in `search` we tokenize the shopper query, score every token against both fields, and add the weighted BM25 contributions to a single score vector.
    3. **Synonym boosts**: after the baseline pass we call the LLM for synonyms, tokenize each suggested phrase, and add their BM25 scores to the same vector. Multi-word phrases naturally fan out into individual tokens, giving us a cheap approximation of query expansion.

    This mirrors what many teams tried in production systems such as Elasticsearch: bolt synonym generation on top of an existing lexical ranker before investing in embedding based retrieval.
    """
    )
    return


@app.cell
def _(products, query_to_syn, snowball_tokenizer):
    from searcharray import SearchArray
    from cheat_at_search.strategy.strategy import SearchStrategy
    import numpy as np


    class SynonymSearch(SearchStrategy):
        def __init__(self, products, synonym_generator,
                     name_boost=9.3,
                     description_boost=4.1):
            """ Build an index."""
            super().__init__(products)
            self.index = products
            self.name_boost = name_boost
            self.description_boost = description_boost

            #*****
            # Take an array of text (here `products['product_name']`)
            # Tokenize it with snowball (the passed function)
            # Produce a searchable index on "product_name_snowball"
            self.index['product_name_snowball'] = SearchArray.index(
                products['product_name'],
                snowball_tokenizer
            )
            self.index['product_description_snowball'] = SearchArray.index(
                products['product_description'], snowball_tokenizer)
            self.query_to_syn = synonym_generator

        def search(self, query, k=10):
            """Dumb baseline lexical search with LLM generated synonyms"""
            # ***
            # Tokenize the query with snowball
            tokenized = snowball_tokenizer(query)
            bm25_scores = np.zeros(len(self.index))

            # ***
            # For each token, get the BM25 score of that token in product name and
            # product description. Sum them
            for token in tokenized:
                bm25_scores += self.name_boost * self.index['product_name_snowball'].array.score(token)
                bm25_scores += self.description_boost * self.index['product_description_snowball'].array.score(
                    token)

            # ***
            # Generate synonyms
            synonyms = self.query_to_syn(query)

            # ***
            # Boost by each synonym phrase
            # (repeat the same above, except we add the BM25 scores of the generated synonyms)
            all_single_tokens = set()
            for mapping in synonyms.synonyms:
                for phrase in mapping.synonyms:
                    tokenized = snowball_tokenizer(phrase)
                    bm25_scores += self.index['product_name_snowball'].array.score(tokenized)
                    bm25_scores += self.index['product_description_snowball'].array.score(tokenized)
                    for token in tokenized:
                        all_single_tokens.add(token)

            # ***
            # Boost by each single token
            # for token in all_single_tokens:
            #     bm25_scores += self.index['product_name_snowball'].array.score(token)
            #     bm25_scores += self.index['product_description_snowball'].array.score(token)

            # ***
            # Sort by BM25 scores
            top_k = np.argsort(-bm25_scores)[:k]
            scores = bm25_scores[top_k]

            return top_k, scores


    syns = SynonymSearch(products, query_to_syn)
    return (syns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Run the strategy and capture results

    `run_strategy` iterates over all 480 WANDS queries, calls our `SynonymSearch` instance, joins in relevance grades, and returns a single dataframe. Expect 4800 rows (480 queries * 10 ranked products) that we can slice and aggregate just like the BM25 baseline.
    """
    )
    return


@app.cell
def _(run_strategy, syns):
    # for each query
    #   results = syns.search(query)
    #   -- Give each result a 'grade'
    #   --- Compute DCG
    graded_syns = run_strategy(syns)
    graded_syns
    return (graded_syns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Inspect a single query

    It helps to sanity-check at least one query before diving into aggregates. The next cells show the ranked products for `wood bar stools` along with the synonyms the model produced.
    """
    )
    return


@app.cell
def _(graded_syns):
    graded_syns[graded_syns['query'] == "wood bar stools"]
    return


@app.cell
def _(query_to_syn):
    query_to_syn("wood bar stools")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Analyze the results

    With both result sets in hand we can compute NDCG per query. The first comparison takes the global mean so we get a sense of the overall lift (or drop) when we add synonym expansion.
    """
    )
    return


@app.cell
def _(graded_bm25, graded_syns, ndcgs):
    ndcgs(graded_bm25).mean(), ndcgs(graded_syns).mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Wins and losses versus BM25

    `ndcg_delta` surfaces the per-query difference between the synonym strategy and the baseline. Large positive deltas point to queries where synonyms surfaced products BM25 could not reach; large negatives highlight cases where noisy expansions drowned out the original intent. Watch the variance here to judge whether the experiment is safe to ship.
    """
    )
    return


@app.cell
def _(graded_bm25, graded_syns, ndcg_delta):
    ndcg_delta(graded_syns, graded_bm25)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Examine a single query (what went right or wrong?)

    Aggregates can hide pathologies. Here we drill into `seat cushions desk`, check what the baseline returned, what the synonym strategy produced, how each compares to the human ideal, and which synonym phrases the LLM added. Toggle the `QUERY` constant to inspect other edge cases.
    """
    )
    return


@app.cell
def _():
    QUERY = "seat cushions desk"
    return (QUERY,)


@app.cell
def _(QUERY, graded_bm25):
    graded_bm25[graded_bm25['query'] == QUERY][['rank', 'product_name', 'product_description', 'grade']]
    return


@app.cell
def _(QUERY, graded_syns):
    graded_syns[graded_syns['query'] ==  QUERY][['rank', 'product_name', 'product_description', 'grade']]
    return


@app.cell
def _(QUERY, graded_syns, vs_ideal):
    against_ideal = vs_ideal(graded_syns)
    against_ideal[against_ideal['query'] == QUERY]
    return


@app.cell
def _(QUERY, query_to_syn):
    query_to_syn(QUERY)
    return


if __name__ == "__main__":
    app.run()
