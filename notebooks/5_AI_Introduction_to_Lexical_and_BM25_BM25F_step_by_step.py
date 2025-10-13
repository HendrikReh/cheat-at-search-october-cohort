import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Lexical scoring – BM25F step by step

    BM25F extends BM25 to handle multiple fields gracefully. Instead of treating each field independently, it blends term statistics across them.

    ### You

    You have BM25 intuition and now want to understand how multi-field blending works under the hood.

    ### Goal

    Diagnose the pitfalls of independent field scoring and rebuild BM25F incrementally: blending document frequencies, normalizing per field, and applying BM25-style saturation.

    ## This notebook: BM25F

    We [previously examined the basics of BM25](https://colab.research.google.com/drive/1RGNkq4SOZMvlFvpHq3IKgNJdCTlHqiek). Here we show why field-specific document frequencies can mislead the ranker and how BM25F addresses the issue.
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

    We index both `msg` and `topics` fields. Each field captures a different aspect of the conversation, setting us up to blend their contributions. The dataset includes cases where a term is rare in one field but common in another—perfect for exposing BM25F’s motivation.
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
      "Doug, this is Tom, support for Earth's Climate, sorry to hear about your complaint, how can we help you doug?",
      "Tom, can I speak to your manager?",
      "Hi, this is Sue, Tom's boss. What can I do for you?",
      "I'd like to complain about the ski conditions in West Virginia",
      "Oh doug thats terrible, lets see what we can do.",
      "Thanks you guys are great.",
      "That's very sweet of you"

    ]

    topics = [
        "bad weather complaint climate",
        "earth climate",
        "escalation support",
        "boss asks",
        "West Virginia ski",
        "doug",
        "grattitude",
        "sweet"

    ]

    msgs = pd.DataFrame({"name": ["Doug", "Doug", "Tom", "Sue", "Doug", "Sue", "Doug", "Sue"],
                         "msg": chat_transcript,
                         "topics": topics})
    msgs['msg_tokenized'] = SearchArray.index(msgs['msg'],
                                              tokenizer=better_tokenize)

    msgs['topics_tokenized'] = SearchArray.index(msgs['topics'],
                                                  tokenizer=better_tokenize)
    msgs
    return better_tokenize, msgs


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Search each field independently

    First we compute BM25 per field and take the max across fields (a term-centric/dismax approach). This exposes how field-specific document frequencies can skew scores.
    """
    )
    return


@app.cell
def _(better_tokenize, msgs, np):
    _QUERY = 'doug complaint'
    _query_tokenized = better_tokenize(_QUERY)
    from searcharray.similarity import compute_idf
    _scores = np.zeros(len(msgs))
    # ACCUMULATE SCORES
    for _query_token in _query_tokenized:
        _impactA = msgs['msg_tokenized'].array.score(_query_token)
        _impactB = msgs['topics_tokenized'].array.score(_query_token)  # Score of each term
        _scores += np.maximum(_impactA, _impactB)
    msgs['score'] = _scores
    msgs.sort_values('score', ascending=False)
    return (compute_idf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Problem: single-field match too strong

    The `topics` field treats `"doug"` as extremely rare, so its BM25 score spikes—even though the word is common in the main message field. Document frequency is being measured per field, not across the whole document, so the specificity estimate is skewed.
    """
    )
    return


@app.cell
def _(msgs):
    print(f"`doug` matches in `topics` field (very high!): {msgs['topics_tokenized'].array.score('doug')}")
    print(f"`doug` matches in `msg` field (much lower): {msgs['msg_tokenized'].array.score('doug')}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Field blending

    BM25F blends per-field term frequencies but shares a single document frequency so the notion of “rarity” reflects the entire document, not an individual field. Conceptually we move from:

    ```
    score = msgs.TF*IDF + topics.TF*IDF
    ```

    to:

    ```
    score = (msgs.TF + topics.TF) * combined_IDF
    ```

    The next few cells rebuild this idea step by step.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Step 1: blend document frequencies

    Start by taking the maximum document frequency across fields before computing IDF. This approximates the intuition that `"doug"` is common overall, not just in one field.
    """
    )
    return


@app.cell
def _(np):
    from searcharray.similarity import Similarity

    def just_tfs(term_freqs: np.ndarray,        # TF array of every doc in the index
                 doc_freqs: np.ndarray,         # Doc freq array of every term (> 1 if a phrase)
                 doc_lens: np.ndarray,          # Every documents length (same shape as TF)
                 avg_doc_lens: int,             # avg doc length of corpus
                 num_docs: int) -> np.ndarray:     # total number of docs in corpus

        return term_freqs
    return (just_tfs,)


@app.cell
def _(better_tokenize, just_tfs, msgs, np):
    _QUERY = 'doug complaint'
    _query_tokenized = better_tokenize(_QUERY)
    _scores = np.zeros(len(msgs))
    for _query_token in _query_tokenized:
        _impactA = msgs['msg_tokenized'].array.score(_query_token, similarity=just_tfs)
        _impactB = msgs['topics_tokenized'].array.score(_query_token, similarity=just_tfs)  # Score of each term
        _docFreq = max(msgs['msg_tokenized'].array.docfreq(_query_token), msgs['topics_tokenized'].array.docfreq(_query_token))
        blended = (_impactA + _impactB) / _docFreq
        print(f"Term '{_query_token}' impactA: {_impactA}")
        print(f"Term '{_query_token}' impactB: {_impactB}")
        print(f"Term '{_query_token}' docFreq: {_docFreq}")
        _scores += blended  # Take doc freq as max of this terms doc freq across terms
    msgs['score'] = _scores
    msgs.sort_values('score', ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Term frequency attenuated by length

    Next we normalize term frequency by document length (controlled by the `b` parameter). Longer fields need more evidence to contribute the same impact as shorter ones.
    """
    )
    return


@app.cell
def _(np):
    b = 0.8
    k1 = 1.1
    # BM25 params

    def bm25_impact(term_freqs: np.ndarray, doc_freqs: np.ndarray, doc_lens: np.ndarray, avg_doc_lens: int, num_docs: int) -> np.ndarray:
        return term_freqs / (1 - b + b * doc_lens / avg_doc_lens)  # TF array of every doc in the index  # Doc freq array of every term (> 1 if a phrase)  # Every documents length (same shape as TF)  # avg doc length of corpus  # total number of docs in corpus
    return bm25_impact, k1


@app.cell
def _(better_tokenize, bm25_impact, msgs, np):
    _QUERY = 'doug complaint'
    _query_tokenized = better_tokenize(_QUERY)
    _scores = np.zeros(len(msgs))
    for _query_token in _query_tokenized:
        _impactA = msgs['msg_tokenized'].array.score(_query_token, similarity=bm25_impact)
        _impactB = msgs['topics_tokenized'].array.score(_query_token, similarity=bm25_impact)
        _docFreq = max(msgs['msg_tokenized'].array.docfreq(_query_token), msgs['topics_tokenized'].array.docfreq(_query_token))
        print(f"Term '{_query_token}' impactA: {_impactA}")
        print(f"Term '{_query_token}' impactB: {_impactB}")
        print(f"Term '{_query_token}' docFreq: {_docFreq}")
        _impact = (_impactA + _impactB) / _docFreq
        print(f"Term '{_query_token}' score: {_impact}")
        _scores += _impact
    msgs.sort_values('score', ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## BM25 term frequency saturation

    Finally we apply the BM25 saturation curve `tf / (tf + k1)` to the blended term frequency. This keeps repeated matches from growing without bound before we multiply by the global IDF.
    """
    )
    return


@app.cell
def _(better_tokenize, bm25_impact, k1, msgs, np):
    _QUERY = 'doug complaint'
    _query_tokenized = better_tokenize(_QUERY)
    _scores = np.zeros(len(msgs))
    for _query_token in _query_tokenized:
        _impactA = msgs['msg_tokenized'].array.score(_query_token, similarity=bm25_impact)
        _impactB = msgs['topics_tokenized'].array.score(_query_token, similarity=bm25_impact)
        _docFreq = max(msgs['msg_tokenized'].array.docfreq(_query_token), msgs['topics_tokenized'].array.docfreq(_query_token))
        print(f"Term '{_query_token}' impactA: {_impactA}")
        print(f"Term '{_query_token}' impactB: {_impactB}")
        print(f"Term '{_query_token}' docFreq: {_docFreq}")
        _impact = _impactA + _impactB
        _impact = _impact / (_impact + k1)
        _impact = _impact / _docFreq
        print(f"Term '{_query_token}' score: {_impact}")
        _scores += _impact
    msgs.sort_values('score', ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Full BM25F

    We're almost to full BM25F, but now we now change to the BM25 inverse document frequency, which is more logarithmic ~(1 / log(df))
    """
    )
    return


@app.cell
def _(better_tokenize, bm25_impact, compute_idf, k1, msgs, np):
    _QUERY = 'doug complaint'
    _query_tokenized = better_tokenize(_QUERY)
    _scores = np.zeros(len(msgs))
    for _query_token in _query_tokenized:
        _impactA = msgs['msg_tokenized'].array.score(_query_token, similarity=bm25_impact)
        _impactB = msgs['topics_tokenized'].array.score(_query_token, similarity=bm25_impact)
        _docFreq = max(msgs['msg_tokenized'].array.docfreq(_query_token), msgs['topics_tokenized'].array.docfreq(_query_token))
        print(f"Term '{_query_token}' impactA: {_impactA}")
        print(f"Term '{_query_token}' impactB: {_impactB}")
        print(f"Term '{_query_token}' docFreq: {_docFreq}")
        _impact = _impactA + _impactB
        _impact = _impact / (_impact + k1)
        idf = compute_idf(len(msgs), _docFreq)
        _impact = _impact * idf
        print(f"Term '{_query_token}' score: {_impact}")
        _scores += _impact
    msgs.sort_values('score', ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---

    ### Key takeaways

    - Blending field term frequencies while sharing a document frequency prevents rare-field artifacts.
    - BM25F reuses BM25’s `k1` (saturation) and `b` (length normalization) after blending—understand them before tuning.
    - Stepwise construction clarifies how BM25F differs from naïve sums, giving you confidence when configuring production search.

    ### Next steps

    - Apply BM25F ideas to your own schema: decide which fields contribute TF and how to weight them.
    - Experiment with per-field boosts or separate `b` values; BM25F supports nuanced weighting per field.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
