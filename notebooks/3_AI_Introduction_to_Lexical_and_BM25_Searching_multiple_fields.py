import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Lexical scoring – searching multiple fields

    Time to move beyond single-field scoring. Real search stacks juggle multiple fields (title, description, tags), and we need a mental model for how their scores combine.

    ### You

    You are a search practitioner exploring how to blend signals from multiple textual fields without losing control.

    ### Goal

    Compare field-centric (sum all fields) and term-centric (take the max per term) strategies, and understand when to stem or weight fields differently.

    ## This notebook: term-centric search

    We [previously discussed controlling index and query time tokenization](https://colab.research.google.com/drive/1RGNkq4SOZMvlFvpHq3IKgNJdCTlHqiek). Now we ask: what happens when a single query must search multiple fields, and how can we make sure the right ones dominate the score?
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

    We index two fields so we can experiment with multi-field scoring:

    1. `name` – who is speaking in the transcript.
    2. `msg` – what they said.

    Both fields share the same tokenizer so differences in score come purely from how we aggregate fields.
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
    _chat_transcript = ['Hi this is Doug, I have a complaint about the weather', "Doug, this is Tom, support for Earth's Climate, how can we help you doug?", 'Tom, can I speak to your manager?', "Hi, this is Sue, Tom's boss. What can I do for you?", 'I have complaints about the ski conditions in West Virginia', 'Oh doug thats terrible, lets see what we can do.']
    msgs = pd.DataFrame({'name': ['Doug', 'Doug', 'Tom', 'Sue', 'Doug', 'Sue'], 'msg': _chat_transcript})
    msgs['msg_tokenized'] = SearchArray.index(msgs['msg'], tokenizer=better_tokenize)
    msgs['name_tokenized'] = SearchArray.index(msgs['name'], tokenizer=better_tokenize)
    msgs
    return better_tokenize, msgs, punctuation


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Use naive TF\*IDF again

    Recall we created a naive TF\*IDF similarity function last time. Let's use that!
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Repeat our search from last time

    Before introducing field-specific logic we rerun the familiar TF*IDF scoring over the message field only. This anchors the results before we add extra signals.
    """
    )
    return


@app.cell
def _(better_tokenize, msgs, np, tf_idf):
    _QUERY = 'doug complaint'
    _query_tokenized = better_tokenize(_QUERY)
    _scores = np.zeros(len(msgs))
    for _query_token in _query_tokenized:
        _score = msgs['msg_tokenized'].array.score(_query_token, similarity=tf_idf)
        print(f"Term '{_query_token}' score: {_score}")
        _scores = _scores + _score
    msgs['scores'] = _scores
    msgs.sort_values('scores', ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Tokenization detour

    While reviewing results we notice `"complaint"` fails to match `"complaints"`. To close the gap we add a simple Snowball stemmer so both words reduce to the same root. This keeps the dataset tiny while illustrating how tokenization and field scoring interact.
    """
    )
    return


@app.cell
def _(Stemmer):
    stemmer = Stemmer.Stemmer('english')
    stemmer.stemWord("complaint"), stemmer.stemWord("complaints")
    return (stemmer,)


@app.cell
def _(punctuation, stemmer):
    def even_better_tokenize(text):
        lowercased = text.lower()
        without_punctuation = lowercased.translate(str.maketrans('', '', punctuation))
        split = without_punctuation.split()
        return [stemmer.stemWord(tok) for tok in split]

    even_better_tokenize("I have complaints about this complaint!")
    return (even_better_tokenize,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Reindex with stemming added

    We rebuild the indexes so both `name` and `msg` fields store stemmed tokens. From this point forward the query and all fields speak the same stemmed vocabulary.
    """
    )
    return


@app.cell
def _(SearchArray, even_better_tokenize, pd):
    _chat_transcript = ['Hi this is Doug, I have a complaint about the weather', "Doug, this is Tom, support for Earth's Climate, how can we help you doug?", 'Tom, can I speak to your manager?', "Hi, this is Sue, Tom's boss. What can I do for you?", 'I have complaints about the ski conditions in West Virginia', 'Oh doug thats terrible, lets see what we can do.']
    msgs_1 = pd.DataFrame({'name': ['Doug', 'Doug', 'Tom', 'Sue', 'Doug', 'Sue'], 'msg': _chat_transcript})
    msgs_1['msg_tokenized'] = SearchArray.index(msgs_1['msg'], tokenizer=even_better_tokenize)
    msgs_1['name_tokenized'] = SearchArray.index(msgs_1['name'], tokenizer=even_better_tokenize)
    msgs_1
    return (msgs_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Search again

    With stemming enabled we repeat the single-field search to confirm the tokenizer change fixes the plural/singular gap before layering in multiple fields.
    """
    )
    return


@app.cell
def _(better_tokenize, msgs_1, np, tf_idf):
    _QUERY = 'doug complaint'
    _query_tokenized = better_tokenize(_QUERY)
    _scores = np.zeros(len(msgs_1))
    for _query_token in _query_tokenized:
        _score = msgs_1['msg_tokenized'].array.score(_query_token, similarity=tf_idf)
        print(f"Term '{_query_token}' score: {_score}")
        _scores = _scores + _score
    msgs_1['scores'] = _scores
    msgs_1.sort_values('scores', ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## New problem: multi-term, multi-field

    Summing all field scores treats repeated hits in the same field the same as complementary hits across different fields. A document containing `doug` twice can outrank one containing both `doug` and `complaint`, even though the latter is closer to the user’s intent.

    We will explore two aggregation patterns:

    - **Field-centric** (sum across fields): good when every field contributes additive evidence.
    - **Term-centric / dismax** (max per term): good when we want one strong match per term regardless of field.
    """
    )
    return


@app.cell
def _(even_better_tokenize, msgs_1, np, tf_idf):
    _QUERY = 'doug complaint'
    _FIELDS = ['msg_tokenized', 'name_tokenized']
    _query_tokenized = even_better_tokenize(_QUERY)
    _scores = np.zeros(len(msgs_1))
    for _query_token in _query_tokenized:
        _field_scores = np.zeros(len(msgs_1))
        for _field in _FIELDS:
            _score = msgs_1[_field].array.score(_query_token, similarity=tf_idf)
            print(f"Field {_field}, Term '{_query_token}' score: {_score}")
            _field_scores = _field_scores + _score
        print(f"Term '{_query_token}' score: {_field_scores}")
        _scores = _scores + _field_scores
        print(f'Scores now: {_field_scores}')
    msgs_1['scores'] = _scores
    msgs_1.sort_values('scores', ascending=False)
    return


@app.cell
def _(even_better_tokenize, msgs_1, np, tf_idf):
    _QUERY = 'doug complaint'
    _FIELDS = ['msg_tokenized', 'name_tokenized']
    _query_tokenized = even_better_tokenize(_QUERY)
    _scores = np.zeros(len(msgs_1))
    for _query_token in _query_tokenized:
        _field_scores = np.zeros(len(msgs_1))
        for _field in _FIELDS:
            _score = msgs_1[_field].array.score(_query_token, similarity=tf_idf)
            print(f"Field {_field}, Term '{_query_token}' score: {_score}")
            _field_scores = np.maximum(_field_scores, _score)
        print(f"Term '{_query_token}' score: {_field_scores}")
        _scores = _scores + _field_scores
        print(f'Scores now: {_field_scores}')
    msgs_1['scores'] = _scores
    msgs_1.sort_values('scores', ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Term-centric (dismax) scoring

    "Dismax" (disjunction maximum) means: for each query term, take the maximum score across fields instead of summing them. This prevents one verbose field from overpowering the rest and ensures each term contributes at most once.

    Many engines call this a [**term-centric**](https://medium.com/@ansuaggarwal/elasticsearch-field-centric-vs-term-centric-approach-f754b6e7d51c) approach.
    """
    )
    return


@app.cell
def _(even_better_tokenize, msgs_1, np, tf_idf):
    _QUERY = 'doug complaint'
    _FIELDS = ['msg_tokenized', 'name_tokenized']
    _query_tokenized = even_better_tokenize(_QUERY)
    _scores = np.zeros(len(msgs_1))
    for _query_token in _query_tokenized:
        _field_scores = np.zeros(len(msgs_1))
        for _field in _FIELDS:
            _score = msgs_1[_field].array.score(_query_token, similarity=tf_idf)
            print(f"Field {_field}, Term '{_query_token}' score: {_score}")
            _field_scores = np.maximum(_field_scores, _score)
        print(f"Term '{_query_token}' score: {_field_scores}")
        _scores = _scores + _field_scores
        print(f'Scores now: {_field_scores}')
    msgs_1['scores'] = _scores
    msgs_1.sort_values('scores', ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Breadcrumbs for Elasticsearch, Vespa, etc

    - Elasticsearch’s [`multi_match`](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-multi-match-query) query supports both field-centric (`type: best_fields`) and term-centric (`type: cross_fields`, `dis_max`) modes.
    - Vespa lets you write ranking expressions directly, e.g., `bm25(title) + bm25(description)` or `max(bm25(title), bm25(tags))`.

    ---

    ### Key takeaways

    - Field-centric (sum) strategies boost documents that hit many fields; term-centric (max) strategies ensure each term contributes once.
    - Tokenization tweaks (stemming) still matter when aggregating across fields—align them before tuning weights.
    - The right aggregation depends on the domain: product search often prefers term-centric logic for head queries; knowledge bases may prefer additive evidence.

    ### Next steps

    - Explore `4_AI_Introduction_to_Lexical_and_BM25_Is_there_a_better_TFIDF_.py` to examine refined weighting schemes.
    - Experiment with custom field weights (e.g., boost `name_tokenized`) and observe how rankings shift.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
