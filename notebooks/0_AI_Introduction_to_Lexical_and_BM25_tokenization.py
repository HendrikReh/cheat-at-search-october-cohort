import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Basics of lexical search

    Welcome to the first stop on the lexical search learning path.

    ### You

    You are an ML or data engineer who lives in pandas/NumPy notebooks and wants to understand how classical search engines (Lucene, Elasticsearch, Vespa) decide when two strings match.

    ### What this is

    A guided walkthrough of tokenization, the bedrock of lexical search. Every later lesson in this series builds on the control you gain here.

    ## This notebook: tokenization

    We explore why tokenization is so influential, using [word-based tokenization](https://towardsdatascience.com/word-subword-and-character-based-tokenization-know-the-difference-ea0976b64e17/) as a starting point and layering in normalization steps. By the end you will see that lexical engines give you a **scalpel**: precise string control compared with the semantic sledgehammer of embeddings.
    """
    )
    return


@app.cell
def _():
    # (use marimo's built-in package management features instead) !pip install searcharray

    from searcharray import SearchArray
    import pandas as pd
    import numpy as np
    return SearchArray, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Tokenization and indexing

    We start with a tiny chat transcript so it is easy to trace tokens back to their source sentences. Think of each row as a document and the `msg` column as the field we will tokenize.

    **Try it:** add a new sentence (e.g., `"CAPS LOCK DOUG!!!"`) and rerun this notebook to see how each tokenizer treats it.
    """
    )
    return


@app.cell
def _(pd):
    chat_transcript = [
      "Hi this is Doug, I'd like to complain about the weather",
      "Doug, this is Tom, support for Earth's Climate, how can we help?",
      "Tom, can I speak to your manager?",
      "Hi, this is Sue, Tom's boss. What can I do for you?",
      "I'd like to complain about the ski conditions in West Virginia",
      "Oh doug thats terrible, lets see what we can do."
    ]

    msgs = pd.DataFrame({"name": ["Doug", "Doug", "Tom", "Sue", "Doug", "Sue"],
                         "msg": chat_transcript})
    msgs
    return (msgs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Word based tokenization

    Out of the box, most lexical systems split on whitespace and basic punctuation. We mimic that with `whitespace_tokenize` so you can see how fragile the default behavior is.
    """
    )
    return


@app.cell
def _():
    def whitespace_tokenize(text):
      return text.split()

    whitespace_tokenize("Mary had a little lamb")
    return (whitespace_tokenize,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### When we index, we tokenize

    Indexing is where tokenization actually happens. `SearchArray.index(...)` takes a series of strings (here `msgs['msg']`), applies your tokenizer, and stores the tokens in a lightweight inverted index column named `msg_tokenized`. Every downstream search call depends on the choices you make at this step.
    """
    )
    return


@app.cell
def _(SearchArray, msgs, whitespace_tokenize):
    msgs['msg_tokenized'] = SearchArray.index(msgs['msg'],
                                              tokenizer=whitespace_tokenize)
    msgs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Then let's search

    Suppose a shopper types `"doug"`. With our current tokenizer the control flow is:

    1. compute the BM25 score for token `"doug"` across all rows,
    2. keep rows where the score is greater than zero (i.e., a token match), and
    3. inspect the matches to confirm intuition.

    If tokenization behaved the way we expect, **every** message mentioning Doug would appear.
    """
    )
    return


@app.cell
def _(msgs):
    _scores = msgs['msg_tokenized'].array.score('doug')
    _matches = msgs[_scores > 0]
    _matches
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Why so few matches?

    Surprise: only the lowercase `"doug"` tokens were returned. Lexical search is literal string matching—every character must align. `"Doug,"` and `"doug"` are different tokens because the comma and capitalization remain attached to the first instance.

    Whenever you see unexpected misses, look first at how the text was tokenized.

    """
    )
    return


@app.cell
def _(whitespace_tokenize):
    whitespace_tokenize("Hi this is Doug, I'd like to complain")
    return


@app.cell
def _(whitespace_tokenize):
    whitespace_tokenize("doug is a nice guy")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The strings above show why: `"Doug,"` includes punctuation and casing that keep it distinct from `"doug"`.

    ```
    Doug, != doug
    ```

    **Takeaway:** lexical search is about *extremely precise* string matching. You decide what to treat as equivalent tokens based on your domain.

    **Why this matters:** matching variants is vital for queries, but also for metadata such as product tags or taxonomy labels. Embedding-based retrieval offers fuzzy semantic matches, whereas lexical retrieval gives you exact control.

    To loosen the matching rules we will:

    1. lowercase tokens,
    2. strip punctuation, and
    3. split on whitespace.
    """
    )
    return


@app.cell
def _():
    from string import punctuation


    def better_tokenize(text):
        lowercased = text.lower()
        without_punctuation = lowercased.translate(str.maketrans('', '', punctuation))
        split = without_punctuation.split()
        return split

    better_tokenize("Doug, that weirdo?"), better_tokenize("Oh this is about doug?")
    return (better_tokenize,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Reindex and search

    After defining the smarter tokenizer, rebuild the index and rerun the same query. Nothing else in the retrieval code changes, so any difference in matches is attributable solely to tokenization.
    """
    )
    return


@app.cell
def _(SearchArray, better_tokenize, msgs):
    msgs['msg_tokenized'] = SearchArray.index(msgs['msg'],
                                              tokenizer=better_tokenize)
    msgs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### It worked!

    By normalizing tokens before indexing we recovered every message that mentions Doug. The retrieval code itself never changed—tokenization alone determined whether the query connected with the stored text.
    """
    )
    return


@app.cell
def _(msgs):
    _scores = msgs['msg_tokenized'].array.score('doug')
    _matches = msgs[_scores > 0]
    _matches
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Breadcrumbs for Elasticsearch, Vespa, etc

    Production engines expose these same knobs:

    - In the Lucene family (Solr, Elasticsearch, OpenSearch) you compose [analyzers](https://www.elastic.co/docs/reference/text-analysis/analyzer-reference) out of character filters, tokenizers, and token filters.
    - Vespa’s [linguistics module](https://docs.vespa.ai/en/linguistics.html) offers similar control via OpenNLP tokenizers.

    Some teams keep tokenization purely server-side, while others push it into client libraries for transparency. Either way, lexical relevance is only as good as your tokenizer.

    ---

    ### Key takeaways

    - Token boundaries are everything: punctuation, casing, and spacing decide whether two strings match.
    - Normalization steps dramatically change recall; test them in a notebook before updating production analyzers.
    - Lexical search gives you deliberate control, complementing (not replacing) semantic retrieval.

    ### Next steps

    - Continue to `1_AI_Introduction_to_Lexical_and_BM25_query_tokenization.py` to see how to tokenize queries symmetrically.
    - Experiment with custom token filters (e.g., stop-word removal) and track how they change the matches above.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
