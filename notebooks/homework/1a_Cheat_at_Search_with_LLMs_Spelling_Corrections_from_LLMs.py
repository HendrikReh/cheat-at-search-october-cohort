import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Spelling Correction with LLMs

    <small>
    (from <a href="http://maven.com/softwaredoug/cheat-at-search">Cheat at Search with LLMs</a> training course by Doug Turnbull.)
    </small>

    We'll use an LLM to correct some search queries. We'll try a first pass naive example, hit some walls, and see where we went wrong.

    **Note** If you haven't already, you may want te review [the first notebook](https://colab.research.google.com/drive/1aUCvcBa1YdmsbIgYc74jlknl9_iRotp1?authuser=2#scrollTo=ccUNd_mLZWdA) which goes through the helpers and other tools here in more detail.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Boilerplate

    Install deps, mount GDrive, prompt for your OpenAI Key (placed in your GDrive)
    """
    )
    return


@app.cell
def _():
    # (use marimo's built-in package management features instead) !pip install git+https://github.com/softwaredoug/cheat-at-search.git
    from cheat_at_search.data_dir import mount
    mount(use_gdrive=True)    # colab, share data across notebook runs on gdrive
    # mount(use_gdrive=False) # <- colab without gdrive
    # mount(use_gdrive=False, manual_path="/path/to/directory")  # <- force data path to specific directory, ie you're running locally.

    from cheat_at_search.search import run_strategy, graded_bm25, ndcgs, ndcg_delta, vs_ideal
    from cheat_at_search.wands_data import products

    products
    return graded_bm25, ndcg_delta, ndcgs, products, run_strategy


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Spelling correction types

    We setup a a pydantic type mapping the full query to a spelling corrected version.
    """
    )
    return


@app.cell
def _():
    from pydantic import BaseModel, Field
    from typing import List
    from cheat_at_search.enrich import AutoEnricher


    class SpellingCorrectedQuery(BaseModel):
        """
        Search query with spelling corrections applied
        """
        corrected_keywords: str = Field(
            ...,
            description="Spell-corrected search query per instructions"
        )
    return AutoEnricher, SpellingCorrectedQuery


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Spellchecking generation code

    We ask OpenAI to generate spelling corrections for a given query.
    """
    )
    return


@app.cell
def _(AutoEnricher, SpellingCorrectedQuery):
    spell_correct_enricher = AutoEnricher(
        model="openai/gpt-4.1-nano",
        # model="google/gemini-2.5-flash-lite",
        system_prompt="You are a helpful AI assistant that very lightly spell-checks furniture e-commerce queries.",
        response_model=SpellingCorrectedQuery
    )

    def get_prompt(query: str) -> str:
        prompt = f"""
            Take this furniture e-commerce query and correct any obvious spelling mistakes

            {query}
        """
        return prompt


    def corrector(query: str) -> SpellingCorrectedQuery:
        return spell_correct_enricher.enrich(get_prompt(query))

    corrector("raaack glass")
    return corrector, spell_correct_enricher


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Spelling corrected search strategy

    We **replace** the original query with spelling checked query.

    Ideas to try
    * What if we **boost** and keep the original?
    """
    )
    return


@app.cell
def _():
    from searcharray import SearchArray
    from cheat_at_search.tokenizers import snowball_tokenizer
    from cheat_at_search.strategy.strategy import SearchStrategy
    import numpy as np

    class SpellingCorrectedSearch(SearchStrategy):
        def __init__(self, products, corrector,
                     name_boost=9.3,
                     description_boost=4.1):
            super().__init__(products)
            self.index = products
            self.name_boost = name_boost
            self.description_boost = description_boost
            self.index['product_name_snowball'] = SearchArray.index(
                products['product_name'], snowball_tokenizer)
            self.index['product_description_snowball'] = SearchArray.index(
                products['product_description'], snowball_tokenizer)
            self.corrector = corrector


        def search(self, query, k=10):
            """Spellcheck lexical search"""
            bm25_scores = np.zeros(len(self.index))
            corrected = self.corrector(query)
            different = corrected.corrected_keywords.lower().split() != query.lower().split()
            asterisk = "*" if different else ""
            if different:
                print(f"Query: {query} -> Corrected: {corrected.corrected_keywords}{asterisk}")
                query = corrected.corrected_keywords
            tokenized = snowball_tokenizer(query)
            for token in tokenized:
                bm25_scores += self.index['product_name_snowball'].array.score(token) * self.name_boost
                bm25_scores += self.index['product_description_snowball'].array.score(token) * self.description_boost
            top_k = np.argsort(-bm25_scores)[:k]
            scores = bm25_scores[top_k]

            return top_k, scores
    return (SpellingCorrectedSearch,)


@app.cell
def _(SpellingCorrectedSearch, corrector, products, run_strategy):
    _spell_check_search = SpellingCorrectedSearch(products, corrector)
    corrected_results1 = run_strategy(_spell_check_search)
    corrected_results1
    return (corrected_results1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Compare NDCGs of our first attempt""")
    return


@app.cell
def _(corrected_results1, graded_bm25, ndcgs):
    ndcgs(graded_bm25).mean(), ndcgs(corrected_results1).mean()
    return


@app.cell
def _(corrected_results1, graded_bm25, ndcg_delta):
    ndcg_delta(corrected_results1, graded_bm25)
    return


@app.cell
def _(corrector):
    corrector("tye dye duvet cover")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Develop ground truth for spelling correction

    We have a manually curated list of spelling corrections to evaluate our corrector. We also add in queries we DONT want to change.

    **Ideas to try** -- how good is this ground truth, can it be improved?
    """
    )
    return


@app.cell
def _():
    def leave_alones(phrases):
        """Queries we don't want to change."""
        return {phrase: phrase for phrase in phrases}

    # These should be left alone
    spellcheck_ground_truth = leave_alones(["kohen 5 drawer dresser",
                                            "grantola wall mirror",
                                            "kisner",
                                            "malachi sled",
                                            "tressler rug",
                                            "bed side table",
                                            "pennfield playhouse",
                                            "platform bed side table",
                                            "liberty hardware francisco",
                                            "wood coffee table set by storage",
                                            "mahone porch rocking chair",
                                            "odum velvet",
                                            "mobley zero gravity adjustable bed with wireless remote",
                                            "fortunat coffee table",
                                            "alter furniture",
                                            "love seat wide faux leather tuxedo arm sofa",
                                            "regner power loom red",
                                            "gurney  slade 56",
                                            "mahone porch rocking chair",
                                            "golub dining table",
                                            "itchington butterfly"])

    actual_correction = {
        "outdoor sectional doning": "outdoor sectional dining",
        "pedistole sink": "pedestal sink",
        "biycicle plant stands": "bicycle plant stands",
        "7 draw white dresser": "7 drawer white dresser",
        "glass lsmp shades": "glass lamp shades",
        "desk for kids tjat ate 10 year old": "desk for kids that are 10 year old",
        "twin over full bunk beds cool desins": "twin over full bunk beds cool designs",
        "sheets for twinxl": "sheets for twin xl",
        "tye dye duvet cover": "tie dye duvet cover",
        "foutains with brick look": "fountains with brick look",
        "westling coffee table": "wesling coffee table",
        "trinaic towel rod": "trinsic towel rod",
        "blk 18x18 seat cushions": "black 18x18 seat cushions",
        "big basket for dirty cloths": "big basket for dirty clothes"
    }

    ground_truth = {**spellcheck_ground_truth, **actual_correction}
    ground_truth
    return (ground_truth,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Function to compute accuracy""")
    return


@app.cell
def _(corrector, ground_truth):
    from tqdm import tqdm

    def acc(corrector):
        hits = []
        misses = []
        for query, correction in tqdm(ground_truth.items()):
            corrected = corrector(query)
            corrected_keywords = corrected.corrected_keywords.strip().lower()
            expected_correction = correction.strip().lower()
            if corrected_keywords == expected_correction:
                hits.append(corrected)
            else:
                print(f"Bad correction: Query: {query} -> Corrected: {corrected.corrected_keywords}")
                misses.append(corrected)

        return len(hits) / (len(hits) + len(misses)), hits, misses

    accuracy, hits, misses = acc(corrector)
    accuracy, hits, misses
    return (acc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Build a better corrector

    We bake in some information to the prompt.

    **Thoughts** - is this overfit to our data? Is this appropriate guidance?
    """
    )
    return


@app.cell
def _(acc, spell_correct_enricher):
    def better_corrector_prompt(query):
        prompt = f"""
            You're a furniture expert, you know all about what Wayfair sells.

            Take the user's query and correct any spelling mistakes.

            * Dont compound words. Just leave the original form alone: IE don't turn anti scratch into anti scratch
            * Dont decompound words Just leave the original form alone: IE dont turn antiscratch into anti scratch
            * Dont add hyphens (ie "anti scratch" not "anti-scratch")
            * *Remember your Wayfair expertise* -- DO NOT correct stylized product names known from the wayfair product line or other furniture / home improvement brands

            Here's the users query:
            {query}
        """
        return prompt



    def better_corrector(query):
        return spell_correct_enricher.enrich(better_corrector_prompt(query))

    acc(better_corrector)
    return (better_corrector,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Rerun to see if NDCG improved""")
    return


@app.cell
def _(SpellingCorrectedSearch, better_corrector, products, run_strategy):
    _spell_check_search = SpellingCorrectedSearch(products, better_corrector)
    corrected_results2 = run_strategy(_spell_check_search)
    corrected_results2
    return (corrected_results2,)


@app.cell
def _(corrected_results2, graded_bm25, ndcgs):
    ndcgs(corrected_results2).mean(), ndcgs(graded_bm25).mean()
    return


@app.cell
def _(corrected_results2, graded_bm25, ndcg_delta):
    ndcg_delta(corrected_results2, graded_bm25)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Examine some of the differences

    **Ideas to try** - Can you improve this corrector further?

    * Produce a list of possible spell corrections
    * Trying a new / different model
    * Few-shotting by passing examples to the prompt
    """
    )
    return


@app.cell
def _(better_corrector):
    better_corrector('kisner')
    return


@app.cell
def _(better_corrector, corrector):
    better_corrector("sheets for twinxl"), corrector("sheets for twinxl")
    return


@app.cell
def _(better_corrector, corrector):
    better_corrector("desk for kids tjat ate 10 year old"), corrector("desk for kids tjat ate 10 year old")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

