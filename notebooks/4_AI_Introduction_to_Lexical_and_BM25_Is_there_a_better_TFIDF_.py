import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Lexical scoring – from TF*IDF to BM25

    We now critique the naïve TF*IDF formulation and see how BM25 addresses its shortcomings.

    ### You

    You already understand TF*IDF and want to know *why* BM25’s tweaks (saturation, length normalization) matter in practice.

    ### Goal

    Observe where raw TF*IDF breaks, introduce incremental fixes, and arrive at the BM25 formula with intuition for each parameter.

    ## This notebook: better TF*IDF (hint: BM25)

    We [previously walked through text similarity scoring](https://colab.research.google.com/drive/1MOUa7u6kE_BWJEeueWjRuClctVxaJlJL#scrollTo=UevYFMMZmbp9) using a simplistic TF*IDF. Here we extend that prototype to handle term frequency saturation and document length bias—two core reasons BM25 performs better in production.
    """
    )
    return


@app.cell
def _():
    # (use marimo's built-in package management features instead) !pip install searcharray pystemmer

    from searcharray import SearchArray
    import pandas as pd
    import numpy as np
    import Stemmer
    return SearchArray, Stemmer, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Tokenize and index

    We again index both `name` and `msg` fields with a stemmed tokenizer. This dataset intentionally includes a long message stuffed with the word “Doug” so we can observe the effect of term frequency saturation and document length normalization.
    """
    )
    return


@app.cell
def _(SearchArray, Stemmer, pd):
    from string import punctuation
    stemmer = Stemmer.Stemmer('english')


    def even_better_tokenize(text):
        lowercased = text.lower()
        without_punctuation = lowercased.translate(str.maketrans('', '', punctuation))
        split = without_punctuation.split()
        return [stemmer.stemWord(tok) for tok in split]


    chat_transcript = [
      "Hi this is Doug, I have a complaint about the weather. My Doug Day is not Doug-tastic.",

      """
        Doug, we see you're having an issue with the climate. Doug, maybe you'd like to talk to the manager?
        Doug I think that'd be wise. What do you think Doug?
      """,
      "Tom, can I speak to your manager?",
      "Hi, this is Sue, Tom's boss. What can I do for you?",
      "I have complaints about the ski conditions in West Virginia",
      "Oh doug thats terrible, lets see what we can do."
    ]

    msgs = pd.DataFrame({"name": ["Doug", "Doug", "Tom", "Sue", "Doug", "Sue"],
                         "msg": chat_transcript})
    msgs['msg_tokenized'] = SearchArray.index(msgs['msg'],
                                              tokenizer=even_better_tokenize)
    msgs['name_tokenized'] = SearchArray.index(msgs['name'],
                                              tokenizer=even_better_tokenize)
    msgs
    return even_better_tokenize, msgs


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Search again with naive TF*IDF

    We reuse the earlier TF*IDF similarity to highlight its behavior on the longer document. Expect the Doug-heavy transcript to dominate because term frequency grows without bound.
    """
    )
    return


@app.cell
def _(np):
    from searcharray.similarity import Similarity

    def tf_idf(term_freqs: np.ndarray,        # TF array of every doc in the index
                   doc_freqs: np.ndarray,         # Doc freq array of every term (> 1 if a phrase)
                   doc_lens: np.ndarray,          # Every documents length (same shape as TF)
                   avg_doc_lens: int,             # avg doc length of corpus
                   num_docs: int) -> np.ndarray:     # total number of docs in corpus

        return term_freqs / (doc_freqs + 1)
    return (tf_idf,)


@app.cell
def _(even_better_tokenize, msgs, np, tf_idf):
    _QUERY = 'doug complaint'
    _FIELDS = ['msg_tokenized', 'name_tokenized']
    _query_tokenized = even_better_tokenize(_QUERY)
    _scores = np.zeros(len(msgs))
    # ACCUMULATE SCORES
    for _query_token in _query_tokenized:
        _field_scores = np.zeros(len(msgs))
        for _field in _FIELDS:  # PASS SIMILARITY
            _score = msgs[_field].array.score(_query_token, similarity=tf_idf)
            print(f"Field {_field}, Term '{_query_token}' score: {_score}")
            _field_scores = np.maximum(_field_scores, _score)
        print(f"Term '{_query_token}' score: {_field_scores}")
        _scores += _field_scores  # Take maximum between field_scores and this field's score
        print(f'Scores now: {_field_scores}')
    msgs['scores'] = _scores
    msgs.sort_values('scores', ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Problem – term frequency saturation

    Our long document repeats “Doug” many times, so raw TF*IDF gives it an outsized score. But repetition alone does not guarantee relevance.

    Early information retrieval research emphasized **aboutness**: after a few occurrences, additional mentions provide diminishing value. We need a scoring function that saturates instead of growing linearly with term frequency.


    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Take log of term freq?

    A quick fix is to take the logarithm of the term frequency so gains diminish as the count grows. This is crude but illustrates the direction BM25 takes.
    """
    )
    return


@app.cell
def _(even_better_tokenize, msgs, np):
    def tf_idf_saturate(term_freqs: np.ndarray, doc_freqs: np.ndarray, doc_lens: np.ndarray, avg_doc_lens: int, num_docs: int) -> np.ndarray:
        return np.log1p(term_freqs) / (doc_freqs + 1)
    _QUERY = 'doug complaint'
    _FIELDS = ['msg_tokenized', 'name_tokenized']
    _query_tokenized = even_better_tokenize(_QUERY)
    _scores = np.zeros(len(msgs))
    for _query_token in _query_tokenized:
        _field_scores = np.zeros(len(msgs))
        for _field in _FIELDS:
            _score = msgs[_field].array.score(_query_token, similarity=tf_idf_saturate)
            print(f"Field {_field}, Term '{_query_token}' score: {_score}")
            _field_scores = np.maximum(_field_scores, _score)
        print(f"Term '{_query_token}' score: {_field_scores}")
        _scores += _field_scores
        print(f'Scores now: {_field_scores}')
    msgs['scores'] = _scores
    msgs.sort_values('scores', ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Problem – field length

    Logging TF helps, but longer documents still accumulate more evidence. A tweet mentioning “Doug” once should count more than a novel mentioning it once. We need to normalize by document length so short fields get proportionally higher credit.
    """
    )
    return


@app.cell
def _(even_better_tokenize, msgs, np):
    def tf_idf_saturate_by_len(term_freqs: np.ndarray, doc_freqs: np.ndarray, doc_lens: np.ndarray, avg_doc_lens: int, num_docs: int) -> np.ndarray:
        tf_idf_sat = np.log1p(term_freqs) / (doc_freqs + 1)
        tf_idf_sat /= doc_lens + 1
        return tf_idf_sat
    _QUERY = 'doug complaint'
    _FIELDS = ['msg_tokenized', 'name_tokenized']
    _query_tokenized = even_better_tokenize(_QUERY)
    _scores = np.zeros(len(msgs))
    for _query_token in _query_tokenized:
        _field_scores = np.zeros(len(msgs))
        for _field in _FIELDS:
            _score = msgs[_field].array.score(_query_token, similarity=tf_idf_saturate_by_len)
            print(f"Field {_field}, Term '{_query_token}' score: {_score}")
            _field_scores = np.maximum(_field_scores, _score)
        print(f"Term '{_query_token}' score: {_field_scores}")
        _scores += _field_scores
        print(f'Scores now: {_field_scores}')
    msgs['scores'] = _scores
    msgs.sort_values('scores', ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## BM25 – TF*IDF tuned to the gills

    BM25 (“Best Match 25”) refines TF*IDF with two parameters:

    ```
    tf = term_freqs / (term_freqs + k1 * (1 - b + b * doc_lens / avg_doc_lens))
    ```

    - `k1` controls how quickly term frequency saturates.
    - `b` (0–1) controls how aggressively long documents are penalized.

    You can experiment with these parameters or explore [interactive graphs](https://www.desmos.com/calculator/lukbszx5oe) to understand their effect. BM25 ships as the default similarity in most modern lexical engines.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Breadcrumbs for Elasticsearch, Vespa, etc

    BM25 ships as the default similarity in Elasticsearch, OpenSearch, Solr, and Vespa. Most platforms expose `k1` and `b` so you can tune saturation and length normalization per field.

    ---

    ### Key takeaways

    - Raw TF*IDF over-rewards repeated terms and longer documents.
    - Adding saturation (`log1p`) and length normalization bridges the gap toward BM25.
    - BM25’s `k1` and `b` parameters map directly to these intuitions—tune them with purpose.

    ### Next steps

    - Move on to `5_AI_Introduction_to_Lexical_and_BM25_BM25F_step_by_step.py` to see how BM25 generalizes across multiple fields.
    - Visualize BM25 curves for your own `k1`/`b` choices using the linked Desmos graph or by plotting inside the notebook.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
