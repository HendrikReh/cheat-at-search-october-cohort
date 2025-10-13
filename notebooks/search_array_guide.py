import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## SearchArray Guide

    [SearchArray](http://github.com/softwaredoug/searcharray) wraps Lucene-style lexical search primitives in a light Pandas-friendly package. It lets us prototype indexing and scoring ideas without running a full Solr/Elasticsearch cluster.

    Think of it as "just enough" Lucene:

    - analyzers/tokenizers decide how text becomes tokens,
    - similarities compute BM25 (or your own scoring function) on those tokens, and
    - everything lives in-memory inside familiar dataframes.

    ### Why use it?

    - iterate on ranking ideas without provisioning infrastructure,
    - collaborate with teammates who know Python and Pandas but not search servers, and
    - bring BM25-style baseline experiments into notebooks where you already explore data.
    """
    )
    return


@app.cell
def _():
    from searcharray import SearchArray
    import pandas as pd
    import numpy as np
    return SearchArray, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Basic indexing

    SearchArray ships with a whitespace tokenizer by default. Indexing a column stores a tokenized view in the dataframe so we can call `.score(...)` later without reprocessing text.
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
      "I'd like to complain about the ski conditions in West Virginia"
    ]

    msgs = pd.DataFrame({"name": ["Doug", "Doug", "Tom", "Sue", "Doug"],
                         "msg": chat_transcript})
    msgs
    return (msgs,)


@app.cell
def _(SearchArray, msgs):
    msgs['msg_tokenized'] = SearchArray.index(msgs['msg'])
    msgs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Basic search (single term)

    Call `.array.score(<term>)` to get BM25 scores for every row. The return value is a vector aligned with the dataframe index, so you can sort or filter like any other column.
    """
    )
    return


@app.cell
def _(msgs):
    msgs['score'] = msgs['msg_tokenized'].array.score("ski")
    msgs.sort_values('score', ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Basic search (phrase)

    Pass a list of tokens to `.score(...)` to compute phrase matches. Under the hood, SearchArray requires all terms to appear in the document and uses the same BM25 similarity for multi-term queries.
    """
    )
    return


@app.cell
def _(msgs):
    msgs['score'] = msgs['msg_tokenized'].array.score(["ski", "conditions"])
    msgs.sort_values('score', ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Custom tokenization (text analysis)

    Most production systems customize analysis: lowercasing, stemming, removing punctuation, etc. SearchArray lets you plug in any callable that turns text into tokens.

    Below we wrap the `Stemmer` Snowball stemmer to show how you might normalize accents, strip punctuation, and stem each term before indexing.
    """
    )
    return


@app.cell
def _():
    import Stemmer
    import string

    stemmer = Stemmer.Stemmer('english')

    def snowball_tokenizer(text):
      fold_to_ascii = dict( [ (ord(x), ord(y)) for x,y in zip( u"‘’´“”–-",  u"'''\"\"--") ] )

      split = text.lower().split()
      folded = [token.translate(fold_to_ascii) for token in split]
      return [stemmer.stemWord(token.translate(str.maketrans('', '', string.punctuation)))
              for token in folded]

    snowball_tokenizer("Mary had a little lamb!")
    return (snowball_tokenizer,)


@app.cell
def _(SearchArray, msgs, snowball_tokenizer):
    msgs['msg_snowball'] = SearchArray.index(msgs['msg'], tokenizer=snowball_tokenizer)
    msgs['score'] = msgs['msg_snowball'].array.score(snowball_tokenizer("earths climate"))
    msgs.sort_values('score', ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Searching with a custom tokenizer

    When you index with a custom tokenizer, the stored `SearchArray` keeps a reference to it. Use `.array.tokenizer(query)` to ensure your query analysis matches the index, then feed the resulting token list into `.score(...)`.
    """
    )
    return


@app.cell
def _(msgs):
    query1 = "earths climate"
    tokenized_phrase = msgs['msg_snowball'].array.tokenizer(query1)
    tokenized_phrase
    return (tokenized_phrase,)


@app.cell
def _(msgs, tokenized_phrase):
    msgs['score'] = msgs['msg_snowball'].array.score(tokenized_phrase)
    msgs.sort_values('score', ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Adjusting similarities

    BM25 is the default similarity and tries to mirror Lucene's implementation. You can tweak it--or replace it entirely--by passing a similarity factory when scoring.

    Each factory returns a callable that accepts term frequencies, document frequencies, document lengths, etc., and outputs a score per document. The next cell shows how to override BM25's `k1` and `b` hyperparameters.
    """
    )
    return


@app.cell
def _(msgs, tokenized_phrase):
    from searcharray.similarity import bm25_similarity

    custom_bm25_sim = bm25_similarity(k1=10, b=0.01)
    msgs['score'] = msgs['msg_snowball'].array.score(tokenized_phrase, similarity=custom_bm25_sim)
    msgs.sort_values('score', ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Custom similarity

    You are not limited to BM25. If you return a callable that consumes the standard arguments (`term_freqs`, `doc_freqs`, etc.) and emits a NumPy array of scores, SearchArray will use it. The example below implements a raw TF * IDF-style scorer for demonstration.
    """
    )
    return


@app.cell
def _(np):
    from searcharray.similarity import Similarity

    def tf_idf_raw() -> Similarity:
        def raw(term_freqs: np.ndarray,        # TF array of every doc
                doc_freqs: np.ndarray,         # Doc freq array of every term (> 1 if a phrase)
                doc_lens: np.ndarray,          # Every documents length (same shape as TF)
                avg_doc_lens: int,             # avg doc length of corpus
                num_docs: int) -> np.ndarray:     # total number of docs in corpus

            phrase_doc_freq = np.sum(doc_freqs)     # In case of phrase
            return term_freqs * (1.0 / phrase_doc_freq)
        return raw

    raw = tf_idf_raw()
    raw(term_freqs=np.asarray([5.0, 3.0]),     # Two docs with term freqs 5 and 3
        doc_freqs=np.asarray([10.0]),          # Single term, df = 10
        doc_lens=np.asarray([10, 20]),
        avg_doc_lens=15,
        num_docs=2)
    return (raw,)


@app.cell
def _(msgs, raw, tokenized_phrase):
    msgs['score'] = msgs['msg_snowball'].array.score(tokenized_phrase, similarity=raw)
    msgs.sort_values('score', ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Advanced queries (edismax, multi-match)

    Complex query parsers such as Solr's `edismax` or Elasticsearch's `multi_match` boil down to combining per-field scores with weights. Because SearchArray exposes raw score arrays, we can rebuild those behaviors with a few lines of Pandas/NumPy.

    We start by tokenizing the query once per field so each analyzer gets the right view of the text.
    """
    )
    return


@app.cell
def _(msgs):
    query2 = "doug ski vacation conditions"

    query_as_snowball = msgs['msg_snowball'].array.tokenizer(query2)
    query_as_whitespace = msgs['msg_tokenized'].array.tokenizer(query2)
    query_as_snowball, query_as_whitespace
    return (query_as_snowball,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Then we compute per-term scores from each field. The resulting matrices are shaped `(num_terms, num_docs)`, which makes it easy to mix-and-match aggregation strategies.
    """
    )
    return


@app.cell
def _(msgs, np, query_as_snowball):
    snowball_scores = np.asarray([msgs['msg_snowball'].array.score(query_term)
                                  for query_term in query_as_snowball])

    whitespace_scores = np.asarray([msgs['msg_tokenized'].array.score(query_term)
                                    for query_term in query_as_snowball])

    snowball_scores, whitespace_scores
    return snowball_scores, whitespace_scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Take max-per-term (dismax)

    A disjunction maximum query picks the best field score per term, then sums the winners. The code here mirrors that logic across the two field-specific matrices we just built. This is effectively what Solr's `edismax` or Elasticsearch's `multi_match` with `type=best_fields` do under the hood.
    """
    )
    return


@app.cell
def _(np, snowball_scores, whitespace_scores):
    best_term_scores_per_doc = []
    for term_idx in range(len(snowball_scores)):
        this_term_scores = np.max([snowball_scores[term_idx], whitespace_scores[term_idx]], axis=0)
        best_term_scores_per_doc.append(this_term_scores)
    best_term_scores_per_doc
    return (best_term_scores_per_doc,)


@app.cell
def _(best_term_scores_per_doc, np):
    scores = np.sum(best_term_scores_per_doc, axis=0)
    scores
    return (scores,)


@app.cell
def _(msgs, scores):
    msgs['score'] = scores
    msgs.sort_values('score', ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Simulate edismax query parser

    If you do not want to hand-roll the math, `searcharray.solr.edismax` wraps the same idea into a helper. Provide the fields (`qf`) you want to search, their boosts if desired, and a query string; the function returns scores along with an optional explanation.
    """
    )
    return


@app.cell
def _(msgs):
    from searcharray.solr import edismax

    msgs['score'], explain = edismax(msgs, q="ski",
                                     qf=["msg_tokenized", "msg_snowball"])
    print(explain)
    msgs.sort_values('score', ascending=False)
    return


if __name__ == "__main__":
    app.run()
