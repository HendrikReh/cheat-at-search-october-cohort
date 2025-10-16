import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Experiment with query semantic vector for caching

    <small>
    (from <a href="http://maven.com/softwaredoug/cheat-at-search">Cheat at Search with LLMs</a> training course by Doug Turnbull.)
    </small>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### A set of highly related queries

    These queries vary a couple of topics, but all are different strings for the same information needs

    * Color: red or purple
    * Item type: shoes or tennis shoes
    * Gender: men or womens

    Below we have two sets of test queries:
    * Queries that should NOT match - they have different, if similar, information needs
    * Queries that its OK if they match

    We will use an embedding model fine-tuned on e-commerce data: `intfloat/e5-small-v2`. You will need to find an embedding model that passes this test.
    """
    )
    return


@app.cell
def _():
    from sentence_transformers import SentenceTransformer
    query_eggshells = ['red tennis shoes', 'tennis shoes, the red ones', 'I want to buy red tennis shoes', 'please give me red tennis shoes', 'red shoes', 'tennis shoes', 'red shoes for men', 'red shoes for women', 'I want to buy red shoes', 'I want to buy red shoes for men', 'purple shoes', 'I want to buy purple shoes', 'purple tennis shoes', 'I need purple tennis shoes', 'tennis shoes in purple', 'can I get some purple tennis shoes?', "I'd like to get some red shoes", "red women's tennis shoes", "men's red tennis shoes", "red shoes men's", 'shoes, red, for women', 'give me tennis shoes in red', 'looking for red shoes for women', "shopping for red men's tennis shoes", "purple women's shoes", "purple men's tennis shoes", 'I’m shopping for tennis shoes, maybe in red', 'tennis shoes that are red', 'show me red tennis shoes for women', 'do you have red tennis shoes?', 'red colored shoes', 'red sneakers', 'purple sneakers for men', "men's purple shoes", "women's red sneakers", "I'd like women's tennis shoes in red", 'get me red tennis shoes', 'can I buy red shoes?', 'I need tennis shoes for men in red', "buy women's tennis shoes purple", "want to see red men's shoes", 'looking for purple tennis shoes for women', 'show me tennis shoes, red color', 'are there purple shoes for women?', 'tennis shoes for women in purple', 'ladies red tennis shoes', "guys' red shoes", 'purple tennis shoes for ladies', 'purple shoes for guys', 'red tennis sneakers', 'tennis shoes — red ones for women', 'searching for red sneakers', 'tennis shoes red male', 'red tennis shoes female', 'I’m looking for red sports shoes', "purple tennis shoes men's", 'tennis sneakers in purple for women', 'buy red sneakers', 'purple athletic shoes for men', 'purple gym shoes', 'red gym shoes', 'ladies tennis shoes, red color', 'purple tennis footwear for men', "men's gym shoes in red", "I'd like to see red shoes for women", 'I need new red tennis shoes', 'let’s buy red shoes', 'do you carry red tennis sneakers?', 'want red running shoes', 'get purple tennis sneakers', 'red tennis shoes for women', 'red shoes in men’s sizes', 'purple colored shoes for women', 'do you have men’s tennis shoes in red?', 'purple shoes for women please', 'can I find red women’s sneakers?', 'I want red tennis shoes for men', 'men’s shoes in red', 'show red tennis shoes', "women's red running shoes", 'purple tennis shoes women want', "buy men's purple shoes", 'buy women’s red sneakers', 'tennis shoes with red color', 'tennis shoes for ladies in red', 'give me red athletic shoes', 'I’m after red tennis sneakers', 'purple workout shoes', 'purple tennis shoes I can run in', "can I buy red women's shoes?", 'buy shoes in red for women', 'looking for tennis shoes, red ones', 'I like purple tennis shoes', 'tennis sneakers in red for men', "ladies' red shoes", 'running shoes, red, for women', 'men’s purple sports shoes', 'tennis shoes, purple, male', 'tennis sneakers purple female', 'purple shoes for sports', 'purple shoes for tennis', 'get me red shoes for gym', 'I’d like tennis shoes, in red']
    model = SentenceTransformer('intfloat/e5-small-v2')
    prefix = 'query: '
    query_embeddings = model.encode([prefix + _query for _query in query_eggshells])
    print(query_embeddings)
    return (
        SentenceTransformer,
        model,
        prefix,
        query_eggshells,
        query_embeddings,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Our 'caching' function

    This will return a hit if similarity with our embedding model exceeds a threshold, otherwise it will not be considered a match.
    """
    )
    return


@app.cell
def _(model, prefix, query_eggshells):
    import numpy as np

    def similar_query(other_query_embeddings, query, threshold=0.95):
        query_embedding = model.encode(prefix + _query)
        scores = np.dot(query_embedding, other_query_embeddings.T)
        max_idx = np.argmax(scores)
        if scores[max_idx] < threshold:
            return (None, None)
        return (query_eggshells[max_idx], scores[max_idx])
    return (similar_query,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### These should not match""")
    return


@app.cell
def _(query_embeddings, similar_query):
    should_not_match = ['show me brown shoes', "I'm looking for brown shoes for men", 'brown shoes for women please', "I'd like some brown tennis sneakers", 'where can I buy brown tennis shoes for women?', 'get me some tennis shoes, brown ones', 'tennis sneakers in brown for men', 'can I find brown sports shoes for women?', "looking for brown men's tennis shoes", "do you carry women's tennis shoes in brown?", 'shopping for brown tennis sneakers', 'brown sneakers for women', 'brown gym shoes for men', "I'd like tennis shoes in brown", 'please show me brown shoes', 'tennis shoes, brown color for ladies', 'can I get brown shoes?', 'brown colored sneakers', 'men’s shoes in brown', 'ladies’ brown tennis sneakers', 'tennis shoes in brown for women', 'give me brown sneakers', 'brown tennis footwear for men', 'brown athletic shoes for women', 'brown casual shoes for men', 'looking to buy brown sneakers', 'find me brown tennis sneakers', 'searching for brown tennis shoes for ladies', 'brown sports shoes', 'brown shoes for guys', 'brown shoes for ladies', 'I need tennis shoes in brown color', "where are brown men's shoes?", "I'd like women's brown shoes", 'can you show me brown sneakers?', 'tennis shoes for women in brown']
    _found = 0
    for _source_query in should_not_match:
        _query, _score = similar_query(query_embeddings, _source_query)
        if _query is not None:
            print(_source_query, similar_query(query_embeddings, _query))
            _found = _found + 1
    print(_found / len(should_not_match))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### These should match""")
    return


@app.cell
def _(query_embeddings, similar_query):
    _should_match = ['I’m shopping for tennis shoes, maybe in purple', 'show me purple tennis shoes for women', 'tennis sneakers that are red', 'do you have purple tennis shoes?', "I'd like to get some purple shoes", 'shopping for red tennis sneakers', 'purple women’s gym shoes', "red shoes in men's sizes", 'men’s purple workout sneakers', 'buy tennis shoes in red', 'can I find red women’s sneakers?', 'tennis shoes in purple for guys', 'do you sell red colored shoes?', 'get me purple sports shoes', 'ladies red athletic sneakers', 'red tennis sneakers for her', 'tennis shoes for women in red', "I'd like to purchase purple shoes", 'show purple workout shoes for men', 'tennis shoes in red for adults', 'purple tennis sneakers for adults', "red women's gym sneakers", 'get women’s tennis shoes, in purple', 'looking to buy red athletic shoes', 'tennis sneakers that are purple', 'purple running shoes for men', 'I’d love some purple tennis shoes', 'red sports sneakers for men', 'purple gym sneakers for women', 'find red tennis sneakers for guys', 'looking for red gym shoes', 'tennis shoes in red for her', 'please find purple athletic shoes', "I'd like tennis shoes for women in purple", "buy women's tennis shoes in red"]
    _found = 0
    for _source_query in _should_match:
        _query, _score = similar_query(query_embeddings, _source_query)
        if _query is not None:
            print(_source_query, similar_query(query_embeddings, _query))
            _found = _found + 1
    print(_found / len(_should_match))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## With sizes

    Numerical attributes can be tricky, here we'll add some shoe sizes to see how well caching performs.
    """
    )
    return


@app.cell
def _(SentenceTransformer):
    query_eggshells_1 = ['red tennis shoes size 9', 'tennis shoes, the red ones, size 10', 'I want to buy red tennis shoes, size 9', 'please give me red tennis shoes size 10', 'red shoes for men, size 9', 'tennis shoes, size 10', 'red shoes for women, size 9', 'I want to buy red shoes in size 10', 'I want to buy red shoes', 'I want to buy red shoes for men', 'purple shoes', 'I want to buy purple shoes', 'purple tennis shoes', 'I need purple tennis shoes', 'tennis shoes in purple', 'can I get some purple tennis shoes?', "I'd like to get some red shoes", "red women's tennis shoes", "men's red tennis shoes", "red shoes men's", 'shoes, red, for women', 'give me tennis shoes in red', 'looking for red shoes for women', "shopping for red men's tennis shoes", "purple women's shoes", "purple men's tennis shoes", 'I’m shopping for tennis shoes, maybe in red', 'tennis shoes that are red', 'show me red tennis shoes for women', 'do you have red tennis shoes?', 'red colored shoes', 'red sneakers', 'purple sneakers for men', "men's purple shoes", "women's red sneakers", "I'd like women's tennis shoes in red", 'get me red tennis shoes', 'can I buy red shoes?', 'I need tennis shoes for men in red', "buy women's tennis shoes purple", "want to see red men's shoes", 'looking for purple tennis shoes for women', 'show me tennis shoes, red color', 'are there purple shoes for women?', 'tennis shoes for women in purple', 'ladies red tennis shoes', "guys' red shoes", 'purple tennis shoes for ladies', 'purple shoes for guys', 'red tennis sneakers', 'tennis shoes — red ones for women', 'searching for red sneakers', 'tennis shoes red male', 'red tennis shoes female', 'I’m looking for red sports shoes', "purple tennis shoes men's", 'tennis sneakers in purple for women', 'buy red sneakers', 'purple athletic shoes for men', 'purple gym shoes', 'red gym shoes', 'ladies tennis shoes, red color', 'purple tennis footwear for men', "men's gym shoes in red", "I'd like to see red shoes for women", 'I need new red tennis shoes', 'let’s buy red shoes', 'do you carry red tennis sneakers?', 'want red running shoes', 'get purple tennis sneakers', 'red tennis shoes for women', 'red shoes in men’s sizes', 'purple colored shoes for women', 'do you have men’s tennis shoes in red?', 'purple shoes for women please', 'can I find red women’s sneakers?', 'I want red tennis shoes for men', 'men’s shoes in red', 'show red tennis shoes', "women's red running shoes", 'purple tennis shoes women want', "buy men's purple shoes", 'buy women’s red sneakers', 'tennis shoes with red color', 'tennis shoes for ladies in red', 'give me red athletic shoes', 'I’m after red tennis sneakers', 'purple workout shoes', 'purple tennis shoes I can run in', "can I buy red women's shoes?", 'buy shoes in red for women', 'looking for tennis shoes, red ones', 'I like purple tennis shoes', 'tennis sneakers in red for men', "ladies' red shoes", 'running shoes, red, for women', 'men’s purple sports shoes', 'tennis shoes, purple, male', 'tennis sneakers purple female', 'purple shoes for sports', 'purple shoes for tennis', 'get me red shoes for gym', 'I’d like tennis shoes, in red']
    model_1 = SentenceTransformer('intfloat/e5-small-v2')
    prefix_1 = 'query: '
    query_embeddings_1 = model_1.encode([prefix_1 + _query for _query in query_eggshells_1])
    print(query_embeddings_1)
    return (query_embeddings_1,)


@app.cell
def _(query_embeddings_1, similar_query):
    should_not_match_1 = ['red tennis shoes size 9show me brown shoes size 9', "I'm looking for brown shoes for men size 10", 'brown shoes for women please size 7', "I'd like some brown tennis sneakers", 'where can I buy brown tennis shoes for women?', 'get me some tennis shoes, brown ones', 'tennis sneakers in brown for men', 'can I find brown sports shoes for women?', "looking for brown men's tennis shoes", "do you carry women's tennis shoes in brown?", 'shopping for brown tennis sneakers', 'brown sneakers for women', 'brown gym shoes for men', "I'd like tennis shoes in brown", 'please show me brown shoes', 'tennis shoes, brown color for ladies', 'can I get brown shoes?', 'brown colored sneakers', 'men’s shoes in brown', 'ladies’ brown tennis sneakers', 'tennis shoes in brown for women size 12', 'give me brown sneakers', 'brown tennis footwear for men', 'brown athletic shoes for women', 'brown casual shoes for men', 'looking to buy brown sneakers', 'find me brown tennis sneakers', 'searching for brown tennis shoes for ladies', 'brown sports shoes', 'brown shoes for guys', 'brown shoes for ladies', 'I need tennis shoes in brown color', "where are brown men's shoes?", "I'd like women's brown shoes", 'can you show me brown sneakers?', 'tennis shoes for women in brown']
    _found = 0
    for _source_query in should_not_match_1:
        _query, _score = similar_query(query_embeddings_1, _source_query)
        if _query is not None:
            print(_source_query, similar_query(query_embeddings_1, _query))
            _found = _found + 1
    print(_found / len(should_not_match_1))
    return (should_not_match_1,)


@app.cell
def _(query_embeddings_1, should_not_match_1, similar_query):
    _should_match = ['red tennis shoes size 9 red tennis shoes', 'tennis shoes, the red ones, size 10', 'I want to buy red tennis shoes, size 9', 'please give me red tennis shoes size 10', 'red shoes for men, size 9', 'tennis shoes, size 10', 'red shoes for women, size 9', 'I want to buy red shoes in size 10']
    _found = 0
    for _source_query in _should_match:
        _query, _score = similar_query(query_embeddings_1, _source_query)
        if _query is not None:
            print(_source_query, similar_query(query_embeddings_1, _query))
            _found = _found + 1
    print(_found / len(should_not_match_1))
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

