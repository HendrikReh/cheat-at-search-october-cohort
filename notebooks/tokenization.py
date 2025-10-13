import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Basics of lexical search
    
        This series walks a Python-friendly audience through the primitives of classical (lexical) search engines such as Lucene, Elasticsearch, and Vespa.
    
        ### You
    
        You are an ML engineer who is comfortable with pandas, NumPy, and notebooks, and you want to understand how traditional search systems decide whether a document matches a query.
    
        ### This notebook
    
        Tokenization is the first, and arguably most important, dial you control in a lexical engine. Every decision about case, punctuation, and word boundaries affects whether a token in the query matches a token in the index. We will:
    
        - build a tiny corpus and index it with a naive tokenizer,
        - observe how brittle the matches are,
        - introduce a slightly smarter tokenizer that normalizes case and punctuation, and
        - draw parallels to how production systems expose analyzer settings.
    
        By the end you should see why lexical search feels like using a scalpel compared with the semantic sledgehammer of embeddings.
        """
    )
    return


@app.cell
def _():
    from searcharray import SearchArray
    return (SearchArray,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Tokenization and indexing
    
        We start with a toy conversation. Think of it as the smallest possible document collection. Each row becomes a "document" and the `msg` column is the text we tokenize. Keeping the dataset tiny makes it easy to inspect the effects of different tokenization strategies.
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


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Word based tokenization
    
        Out of the box, many search engines split text on whitespace and punctuation. We mimic that with a trivial tokenizer that simply calls `str.split()`. It is intentionally naive so we can see the pitfalls before adding normalization.
        """
    )
    return


@app.cell
def _():
    def whitespace_tokenize(text):
      return text.split()

    whitespace_tokenize("Mary had a little lamb")
    return (whitespace_tokenize,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ### When we index, we tokenize
    
        Indexing is where tokenization happens. `SearchArray.index(...)` consumes a series of strings, runs the tokenizer you provide, and stores the tokens in an efficient structure. Here we add a new column, `msg_tokenized`, that behaves like an inverted index: each row now has an `array.score(...)` method for later retrieval.
        """
    )
    return


@app.cell
def _(SearchArray, msgs, whitespace_tokenize):
    msgs['msg_tokenized'] = SearchArray.index(msgs['msg'],
                                              tokenizer=whitespace_tokenize)
    msgs
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## First search
    
        Suppose a user searches for "doug". With the naive tokenizer we just built, the workflow looks like:
    
        1. Ask `msg_tokenized` for the BM25 score of the token `"doug"`.
        2. Filter rows where the score is greater than zero.
        3. Inspect the matches.
    
        If our tokenizer behaved the way we hoped, every mention of Doug should appear in the results.
        """
    )
    return


@app.cell
def _(msgs):
    scores1 = msgs['msg_tokenized'].array.score("doug")
    matches1 = msgs[scores1 > 0]
    matches1
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Why so few matches?
    
        The results are disappointing: only the lowercase "doug" tokens matched. Lexical search is brutally literal. Every character in the stored token must line up with the query token. Case, punctuation, and even trailing commas create distinct tokens.
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


@app.cell
def _(mo):
    mo.md(
        r"""
        In the transcript, `"Doug,"` and `"doug"` are different tokens because the comma and capitalization stay attached to the first instance. Two strings that look similar to a human are entirely different to the tokenizer:
    
        ```
        Doug, != doug
        ```
    
        **Takeaway** Lexical search gives you surgical control over what counts as a match. You decide which variants collapse together and which remain distinct. That control is essential for text matching, but it is also critical when indexing product tags, categories, or any structured labels. In contrast, embedding based retrieval uses a semantic similarity threshold that trades precision for fuzziness.
    
        To make the tokenizer a little friendlier we will:
    
        1. lowercase the text,
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


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Reindex and search
    
        After defining the smarter tokenizer we rebuild the index and rerun the same query. Nothing else in the retrieval code changes; only the tokenization step determines whether `"doug"` and `"Doug,"` land in the same bucket.
        """
    )
    return


@app.cell
def _(SearchArray, better_tokenize, msgs):

    msgs['msg_tokenized'] = SearchArray.index(msgs['msg'],
                                              tokenizer=better_tokenize)
    msgs
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### It worked!
    
        Every mention of Doug now matches because we normalized the tokens before indexing. The retrieval code did not change at all; tokenization alone determined whether the query connected with the stored text.
        """
    )
    return


@app.cell
def _(msgs):
    scores2 = msgs['msg_tokenized'].array.score("doug")
    matches2 = msgs[scores2 > 0]
    matches2
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Breadcrumbs for Elasticsearch, Vespa, and friends
    
        Production engines expose similar controls under different names:
    
        - In the Lucene family (Solr, Elasticsearch, OpenSearch) you configure [analyzers](https://www.elastic.co/docs/reference/text-analysis/analyzer-reference) that chain together character filters, tokenizers, and token filters.
        - Vespa bundles comparable behavior in its [linguistics module](https://docs.vespa.ai/en/linguistics.html), which relies on OpenNLP tokenizers under the hood.
    
        The indexing pipeline often runs on the server when you send text over the wire, but there is an ongoing debate about pushing more of that logic to clients for transparency and reproducibility. Whichever side you take, the key idea remains: lexical relevance is only as good as your tokenization.
        """
    )
    return


if __name__ == "__main__":
    app.run()
