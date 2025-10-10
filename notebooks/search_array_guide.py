import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## SearchArray Guide

    [SearchArray](http://github.com/softwaredoug/searcharray) is intended to be a very minmial API for lexical (ie BM25) search on top of a Pandas Dataframe.

    The API is inspired by Lucene, so if you're comfortable with core search concepts from Lucene-search engines (Solr, Elasticsearch, OpenSearch, you'll be fine). Just like Lucene we have analyzers/tokenizers and similarities.

    ### WHY!?!?

    * Help prototype ideas without standing up a search engine
    * To let people without Solr / Elasticsearch expertise propose ideas
    * Bring the lexical / BM25 into the normal Python data world
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
    ### Basic Indexing

    We start with basic / default tokenization that doesn't do anything special.
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

    Searching is just a matter of calling "score"
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

    Phrases are just lists of terms passed to score
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
    ## Custom tokenization (aka text analysis)

    You almost always want some kind of custom tokenization (stemming, etc).

    Luckily python comes with a rich array of stemmers, lematizers, and other functionality. SearchArray intentionally avoids creating its own library of tokenizers for this reason.

    Here's an example using snowball.
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
    ### Searching with custom tokenizer

    The `score` method expects pre-tokenized terms. You can use the `tokenizer` used at index time pretty easily.
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
    ## Changing similarities

    By default, we use BM25 that attempts to mirror Lucene's BM25 implementation. But this can be changed by simply passing similarity at query time.

    Each "similarity" is a factory function that itself returns a function. Notice below we customize bm25's k1 and b parameters.
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

    You can also just make your own similarity if you create a function that returns a function that satisfies the contract.

    Given an array of term_freqs for each doc, and other doc/term stats, you should return an array of similarity scores of the same length of term_freqs.

    See the comments below with an example of raw TF*IDF
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
    ## Advanced queries (edismax? multi-match?)

    What about things like Solr's edismax? Or a big Elasticsearch multi-match query?

    Well, in the end, these things are just math. And you know what Pandas good at? Math!

    So, for example, an Elasticsearch multi-match query searching different fields, multiplying them by a weight (ie boost), and then summing or taking the maximum score.

    First we tokenize the query according to each field's tokenizer
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
    Then we get a score for each field, for each query term.

    The resultiing arrays are shaped num_terms x num_docs
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
    ## Take max-per-term (ie 'dismax')

    In search, ["disjunction maximum"](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-dis-max-query.html) or "dismax" just means take the maximum score. That's pretty easy to do with these two arrays. It sits underneath the hood of many base queries like edismax or multi-match.
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
    mo.md(r"""## Simulate edismax query parser""")
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
