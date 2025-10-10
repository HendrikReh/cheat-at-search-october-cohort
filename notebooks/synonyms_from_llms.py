import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Synonyms generation with an LLM

    <small>
    (from <a href="http://maven.com/softwaredoug/cheat-at-search">Cheat at Search with LLMs</a> training course by Doug Turnbull.)
    </small>

    Let's get familiar with the code we'll use for this class by doing what a lot of search teams did when they heard about LLMs

    * Can I generate synonyms using LLMs?

    We'll try to expand queries -> their synonyms and see if it helps NDCG
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

    Import the following helpers:

    * `run_strategy` -- this runs a "strategy" and gives us the search results for each query back (more on this in a second)
    * `graded_bm25` -- a BM25 search baseline. A dump of the search results of every test query in the Wayfair dataset run using a BM25 baseline. Useful to compare our attempts against.
    * `ndcgs` -- Take one of the sets of search results (ie `graded_bm25`) and get the NDCG of each query
    * `ndcg_delta` -- Compare two sets of search results (ie `graded_bm25` vs `graded_my_cool_experiment`) and see which queries do better / worse
    * `vs_ideal` -- Take a set of search results (ie `graded_bm25`) and compare against the ideal according to the ground truth data.
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

    Import [Wayfair Annotated Dataset](https://github.com/wayfair/WANDS) a labeled furniture e-commerce dataset. This is a helpful dataset that has 480 e-commerce queries, along with ~45K furniture / home goods products, and relevance labels for each. In WANDS relevance labels range from 0 (not at all relevant) to 2 (relevant)

    Below you see a sample of the corpus as a pandas dataframe.
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

    We'll first setup the scaffolding of setting up query -> synonym mapping. Expecting a list back of phrases -> their synonyms.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Pydantic Models for Structured Output

    ["Pydantic"](https://docs.pydantic.dev/latest/) is a Python way of having a struct or simple data class. It can be a useful way to serialize data to/from underlying data formats (ie JSON, protobuf). And we'll largely work at this level of abstraction.

    We're using [OpenAI's structured output](https://platform.openai.com/docs/guides/structured-outputs). Which means:

    * Using pydantic to define the expected output (with a description that the model can use)
    * Creating a 'struct like' view of the data we want OpenAI to produce.
    * Forcing OpenAI to return a specific format, and not begging it to return parsable JSON

    This pattern of using structured outputs is common across other vendors such al Ollama, Gemini, etc. Though there may be mild differences in how the pydantic types are interpreted.
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

    We use `AutoEnricher` in this class. This is something that wraps the calls to OpenAI in the `cheat_at_search` package.

    Notice when constructing it, we provide three values:

    * `model` -- the underlying LLM to use. If you load ChatGPT, you would notice the dropdown of models you can select. They each have pros/cons with cost and quality.
    * `system_prompt` -- the general behavior of the agent, priming it for the task its about to perform
    * `response_model` -- the Pydantic class to use to generate structured outputs

    We can then call `enricher.enrich(prompt)` and get back an instance of `QueryWithSynonyms`

    Notice too `get_prompt` generates a prompt given a search query.
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

    We'll use a [snowball stemmer](https://www.nltk.org/api/nltk.stem.SnowballStemmer.html) when we index the data. This is just a function that takes a string and returns a list of tokens, each snowball stemmed.
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
    ### Build a SearchStrategy -- Enrich, index, search

    A SearchStrategy emulates a typical search system, but in a mini form suitable for dorking around in this notebook.

    Notice in `__init__`, indexing:

    ```
        self.index['product_name_snowball'] = SearchArray.index(
                products['product_name'],
                snowball_tokenizer
            )
    ```

    Then later we `search`, summing up BM25 scores across different fields:

    ```
            # ***
            # For each token, get the BM25 score of that token in product name and
            # product description. Sum them
            for token in tokenized:
                bm25_scores += self.name_boost * self.index['product_name_snowball'].array.score(token)
                bm25_scores += self.description_boost * self.index['product_description_snowball'].array.score(
                    token)
    ```

    Farther down, you see we boost also when we match a synonym phrase.

    #### SearchArray

    We use a lexical search library [SearchArray](http://github.com/softwaredoug/search-array) for simple lexical searches. (See the notebooks and information in the prework for the class)

    In the case of synonyms, a lot of teams trying this have a mature lexical search system like Elasticsearch. Instead of adding embedding retrieval to the search, they try this hack to see if they can cheat at search.
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
    ### Run strategy, get results back

    We call `run_strategy` which behind the scene passes every WANDS query to the `syns` strategy to get search results. Then appends them all to `graded_syns` which has 480 queries times 10 results per query (4800 rows)
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
    mo.md(r"""### Look at one search result...""")
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

    Let's look at the results to see how we did against a BM25 baseline

    Here we get ndcg of each query with `ndcgs`, then compute the mean for all queries. We do this comparing BM25 vs our synonym variant
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
    ### Win / loss against BM25 baseline

    `ndcg_delta` shows us the per-query NDCG difference

    * We note some massive wins
    * We unfortunately also note massive variance in outcomes (meaning a risky change)
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
    ### Examine a single query (what went right/wrong?)

    First we see what BM25 produced...
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
