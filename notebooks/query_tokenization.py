import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Lexical tokenization – query-time tokenization

    This notebook builds on the index-time tokenization concepts. Instead of modifying how we store documents, we focus on how queries are parsed before they hit the index.

    ### You

    You are an ML engineer comfortable with the Python data stack who wants to understand how a lexical engine interprets multi-term queries.

    ### Goal

    Learn why query-time tokenization is **not** automatic, how different combinations (OR vs AND) change matches, and how these knobs map to production query DSLs.

    ## Recap & scope

    We [previously discussed index-time tokenization](https://colab.research.google.com/drive/1Mz2H05900XlNdnV_IXveDukYeEV3HABi). Now we mirror the same ideas at query time: what happens when a shopper types `"doug complaint"` and how do we control whether the engine searches for the whole phrase, either term, or both terms together?
    """
    )
    return


@app.cell
def _():
    from searcharray import SearchArray
    import pandas as pd
    import numpy as np
    return SearchArray, np, pd


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Tokenize and index

    We reuse the "better" tokenizer from the index-time notebook—lowercasing, stripping punctuation, and splitting on whitespace—so the index behaves the way we expect. The focus here is on how we prepare the incoming query to match that index.
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


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Search with two terms

    Imagine a user typing `"doug complaint"`. Intuitively we want to find conversations where Doug filed a complaint. Let's see what happens if we pass that raw string to the index.
    """
    )
    return


@app.cell
def _(msgs):
    QUERY = "doug complaint"
    matches1 = msgs['msg_tokenized'].array.score(QUERY) > 0
    msgs[matches1]
    return (QUERY,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Again nothing matched!?

    Zero matches. Why? Because `SearchArray.array.score(...)` does not tokenize the string for us:

    1. A single string argument is treated as **one token**.
    2. A list of strings is treated as a **phrase**.

    When we passed `"doug complaint"` we effectively asked for the token `"doug complaint"`—as though the index stored bigrams. SearchArray deliberately avoids tokenizing queries automatically so you get to choose the behavior.

    Fortunately, this version of the problem is easy to fix by applying the same tokenizer to the query.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Fix by tokenizing the query?

    Step one is to run the query text through the same `better_tokenize` function we used for indexing. That ensures we compare apples to apples.
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
    matches2 = np.zeros(len(msgs), dtype=bool)
    for query_token2 in query_tokenized:
        matches2 |= (msgs['msg_tokenized'].array.score(query_token2) > 0)

    msgs[matches2]
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Fixed! (kinda)

    1. We tried searching for `doug complaint`
    2. We changed to tokenizing `doug complaint` - > `['doug', 'complaint']`
    3. Any match for either term we accept as a match
    4. We end up with 4 matches (despite only one document mentioning both `doug` AND `complaint`)
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## OK Actually fix - require all terms to match

    The previous loop used `|=` to accumulate matches: if any token appeared we kept the row. That is an **OR query**. Now switch to an **AND query** by starting with all `True` values and intersecting (`&=`) each token's matches. Only rows that contain **every** token survive.

    One reason to give YOU control and not SearchArray, is so you can make decisions like this!
    """
    )
    return


@app.cell
def _(msgs, np, query_tokenized):
    matches3 = np.ones(len(msgs), dtype=bool)
    for query_token3 in query_tokenized:
        matches3 &= (msgs['msg_tokenized'].array.score(query_token3) > 0)

    msgs[matches3]
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Breadcrumbs for Elasticsearch, Vespa, etc

    Production engines expose these choices through their query DSLs:

    - Elasticsearch/OpenSearch offer the [bool query](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-bool-query.html) along with [match](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-match-query.html) and [multi_match](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-multi-match-query.html) operators that let you pick `operator: "or"` vs `operator: "and"`, minimum-should-match thresholds, and per-field boosts.
    - Vespa uses the [YQL `contains`](https://docs.vespa.ai/en/reference/query-language-reference.html) clause and filters to express similar semantics.

    By handling query tokenization yourself—even in a notebook—you get a feel for the levers those DSLs expose and why engines avoid guessing your intent automatically.
    """
    )
    return


if __name__ == "__main__":
    app.run()
