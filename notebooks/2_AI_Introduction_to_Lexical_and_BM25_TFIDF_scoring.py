import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Lexical scoring – TF*IDF intuition

    After mastering tokenization, the next question is **how** a matched token contributes to a document’s relevance score.

    ### You

    You are a notebook-first engineer who wants to peek behind BM25’s curtain and understand the TF*IDF building blocks.

    ### Goal

    Rebuild scoring step by step—from raw term counts to TF/DF weighting—so that BM25 no longer feels like a black box.

    ## This notebook: TF*IDF scoring

    We [previously discussed controlling index and query time tokenization](https://colab.research.google.com/drive/1RGNkq4SOZMvlFvpHq3IKgNJdCTlHqiek). Here we assume tokens are aligned and focus purely on scoring. You will code tiny similarity functions, inspect their outputs, and see how they connect to BM25.
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

    As before we build a miniature corpus and index it with a normalized tokenizer (lowercasing, punctuation stripping, whitespace splitting). Keeping the input simple means any change in ranking comes purely from the scoring function.
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
      "Doug, this is Tom, support for Earth's Climate, how can we help you doug?",
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
    ## Search (again)

    We reuse the query from the previous notebook to keep the matching logic identical. The goal is to compare *scores*, not matches, as we swap similarity functions.
    """
    )
    return


@app.cell
def _(better_tokenize, msgs, np):
    _QUERY = 'doug complaint'
    _query_tokenized = better_tokenize(_QUERY)
    matches = np.zeros(len(msgs), dtype=bool)
    for _query_token in _query_tokenized:
        matches |= msgs['msg_tokenized'].array.score(_query_token) > 0
    msgs[matches]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Let's generate scores

    `SearchArray.array.score` uses BM25 by default. Instead of treating BM25 as magic, we'll plug in our own `similarity` functions and inspect the raw components passed to them:

    - `term_freqs`: frequency of the token in each document.
    - `doc_freqs`: number of documents containing the token.
    - `doc_lens`: token counts per document.
    - `avg_doc_lens`: average document length across the corpus.
    - `num_docs`: total number of documents.

    By experimenting with these ingredients we can recreate TF*IDF—and understand how BM25 tweaks it.
    """
    )
    return


@app.cell
def _(np):
    from searcharray.similarity import Similarity

    def term_counts(term_freqs: np.ndarray,        # TF array of every doc in the index
                    doc_freqs: np.ndarray,         # Doc freq array of every term (> 1 if a phrase)
                    doc_lens: np.ndarray,          # Every documents length (same shape as TF)
                    avg_doc_lens: int,             # avg doc length of corpus
                    num_docs: int) -> np.ndarray:     # total number of docs in corpus

        return term_freqs
    return (term_counts,)


@app.cell
def _(msgs, term_counts):
    msgs['msg_tokenized'].array.score('doug', similarity=term_counts)
    return


@app.cell
def _(better_tokenize, msgs, np, term_counts):
    _QUERY = 'doug complaint'
    _query_tokenized = better_tokenize(_QUERY)
    _scores = np.zeros(len(msgs))
    # ACCUMULATE SCORES
    for _query_token in _query_tokenized:
        _score = msgs['msg_tokenized'].array.score(_query_token, similarity=term_counts)
        print(f"Term '{_query_token}' score: {_score}")  # PASS SIMILARITY
        _scores += _score
    msgs['scores'] = _scores
    msgs.sort_values('scores', ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Notice – we're not preferring any particular term

    With raw term counts the score increases every time a token repeats, regardless of how informative that token is. In our example, multiple occurrences of `"doug"` can outweigh the single mention of `"complaint"` even though the complaint token is more discriminative.

    To rebalance the influence we introduce document frequency (DF)—the number of documents that contain the token. Dividing TF by DF (or multiplying by IDF) down-weights ubiquitous tokens and boosts rarer ones.
    """
    )
    return


@app.cell
def _(np):
    def tf_over_df(term_freqs: np.ndarray, doc_freqs: np.ndarray, doc_lens: np.ndarray, avg_doc_lens: int, num_docs: int) -> np.ndarray:
        return term_freqs / doc_freqs  # TF array of every doc in the index  # Doc freq array of every term (> 1 if a phrase)  # Every documents length (same shape as TF)  # avg doc length of corpus  # total number of docs in corpus
    return (tf_over_df,)


@app.cell
def _(better_tokenize, msgs, np, tf_over_df):
    _QUERY = 'doug complaint'
    _query_tokenized = better_tokenize(_QUERY)
    _scores = np.zeros(len(msgs))
    # ACCUMULATE SCORES
    for _query_token in _query_tokenized:
        _score = msgs['msg_tokenized'].array.score(_query_token, similarity=tf_over_df)
        print(f"Term '{_query_token}' score: {_score}")  # PASS SIMILARITY
        _scores += _score
    msgs['scores'] = _scores
    msgs.sort_values('scores', ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## TF divided by DF == TF * IDF ≈ BM25

    IDF stands for *inverse document frequency* (`1 / DF`). Multiplying TF by IDF yields the classic TF*IDF weighting. BM25 extends this idea with two key refinements:

    - saturation (`k1`) prevents TF from growing without bound, and
    - length normalization (`b`) prevents longer documents from dominating purely because they contain more tokens.

    By inspecting TF and TF/DF in isolation we can reason about why adjusting BM25’s parameters changes ranking behavior.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Breadcrumbs for Elasticsearch, Vespa, etc

    - Elasticsearch/Lucene let you pick or implement a [similarity](https://www.elastic.co/docs/reference/elasticsearch/index-settings/similarity) class (classic TF*IDF, BM25, DFR, or a plugin).
    - Vespa exposes the same term statistics in ranking expressions, giving you direct control to implement custom formulas.

    ---

    ### Key takeaways

    - Scoring functions receive TF, DF, document length, and corpus stats—perfect ingredients for experimentation.
    - TF*IDF rewards rarer terms but still lacks BM25’s saturation and length normalization.
    - Prototyping similarities in a notebook demystifies the parameters you tune in production engines.

    ### Next steps

    - Continue to `3_AI_Introduction_to_Lexical_and_BM25_Searching_multiple_fields.py` to combine field-specific scores.
    - Try authoring your own similarity that includes a length penalty or threshold and observe how rankings shift.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
