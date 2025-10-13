import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Lexical tokenization – query-time control

    We move from index-time tokenization to query-time parsing: what happens when a user types a multi-term query, and how do we keep the query analysis in sync with the index?

    ### You

    You are a data/search engineer who already understands how documents were tokenized and now needs to reason about how user queries are interpreted.

    ### Goal

    See why lexical engines don’t tokenize queries for you, explore OR vs AND semantics, and connect those concepts to production query DSLs.

    ## This notebook: query-time tokenization

    We [previously discussed index-time tokenization](https://colab.research.google.com/drive/1Mz2H05900XlNdnV_IXveDukYeEV3HABi); now we reuse the exact same tokenizer on the query side. Along the way you will implement OR and AND combinations manually to feel the control that SearchArray (and engines like Elasticsearch) hand to you.
    """
    )
    return


@app.cell
def _():
    # (use marimo's built-in package management features instead) !pip install searcharray

    from searcharray import SearchArray
    import pandas as pd
    import numpy as np
    return SearchArray, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Tokenize and index

    We reuse the normalized tokenizer from the previous notebook so document tokens behave predictably. Keeping the corpus tiny makes it easy to track which messages should match once we tokenize the query the same way.
    """
    )
    return


@app.cell
def _(SearchArray, pd):
    from string import punctuation


    def better_tokenize(text):
        lowercased = text.lower()
        without_punctuation = lowercased.translate(str.maketrans('', '', punctuation))
        split = without_punctuation.split()
        return split


    chat_transcript = [
      "Hi this is Doug, I have a complaint about the weather",
      "Doug, this is Tom, support for Earth's Climate, how can we help?",
      "Tom, can I speak to your manager?",
      "Hi, this is Sue, Tom's boss. What can I do for you?",
      "I'd like to complain about the ski conditions in West Virginia",
      "Oh doug thats terrible, lets see what we can do."
    ]

    msgs = pd.DataFrame({"name": ["Doug", "Doug", "Tom", "Sue", "Doug", "Sue"],
                         "msg": chat_transcript})
    msgs['msg_tokenized'] = SearchArray.index(msgs['msg'],
                                              tokenizer=better_tokenize)
    msgs
    return better_tokenize, msgs


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Search with two terms

    Let’s search for `"doug complaint"`. Intuitively we want documents discussing Doug’s complaint, but we have to decide how the query string should be tokenized before it hits the index.
    """
    )
    return


@app.cell
def _(msgs):
    QUERY = 'doug complaint'
    _matches = msgs['msg_tokenized'].array.score(QUERY) > 0
    msgs[_matches]
    return (QUERY,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Again nothing matched!?

    Nothing returned because `SearchArray.array.score` does **not** tokenize the query for you:

    1. A single string is treated as **one token**.
    2. A list of strings is treated as a **phrase** (all tokens must appear).

    Passing the raw string `"doug complaint"` asked the index for a single token spelled exactly like that—effectively a bigram. Because we never stored that token, the result set is empty.

    The fix is to tokenize the query using the same function we used at index time.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Fix by tokenizing the query?

    Step one: run the shopper query through `better_tokenize` so the tokens are shaped exactly like the ones stored in the index.
    """
    )
    return


@app.cell
def _(QUERY, better_tokenize):
    query_tokenized = better_tokenize(QUERY)
    query_tokenized
    return (query_tokenized,)


@app.cell
def _(msgs, np, query_tokenized):
    _matches = np.zeros(len(msgs), dtype=bool)
    for _query_token in query_tokenized:
        _matches |= msgs['msg_tokenized'].array.score(_query_token) > 0
    msgs[_matches]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Fixed! (kinda)

    1. Tokenize the query: `['doug', 'complaint']`.
    2. Score each token separately and OR the results together.
    3. Any document containing either token is returned.

    This recovers relevant hits, but we also accept messages that mention only Doug or only the complaint. Sometimes that is fine; other times we need stricter matching.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## OR vs AND control

    The OR loop above used `|=` to accumulate matches. Here we flip the logic to an AND query by initializing to all `True` and intersecting (`&=`) with each token’s matches. Only documents containing *every* token survive.

    Having direct boolean control is why SearchArray leaves query tokenization (and combination) in your hands.
    """
    )
    return


@app.cell
def _(msgs, np, query_tokenized):
    _matches = np.ones(len(msgs), dtype=bool)
    for _query_token in query_tokenized:
        _matches &= msgs['msg_tokenized'].array.score(_query_token) > 0
    msgs[_matches]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Breadcrumbs for Elasticsearch, Vespa, etc

    Production engines expose comparable controls:

    - Elasticsearch/OpenSearch offer the [bool query](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-bool-query.html) and [`match`/`multi_match`](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-match-query.html) operators where you pick `operator: "or"`, `operator: "and"`, or `minimum_should_match` thresholds.
    - Vespa’s [YQL `contains`](https://docs.vespa.ai/en/reference/query-language-reference.html) clause lets you express similar logical requirements.

    By tokenizing queries yourself you can translate notebook experiments directly into those DSL settings.

    ---

    ### Key takeaways

    - Query tokenization is not automatic; engines expect you to send either a single token or a phrase list.
    - OR and AND semantics boil down to simple boolean operations—prototype them before choosing DSL parameters.
    - Matching the query tokenizer to the index tokenizer prevents head-scratching mismatches in production.

    ### Next steps

    - Proceed to `2_AI_Introduction_to_Lexical_and_BM25_TFIDF_scoring.py` to explore how scores are computed once tokens match.
    - Experiment with hybrid logic, such as requiring one “must have” token and allowing optional others via `minimum_should_match` in your engine of choice.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
