import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Query -> Category Perfect Classification

    <small>
    (from <a href="http://maven.com/softwaredoug/cheat-at-search">Cheat at Search with LLMs</a> training course by Doug Turnbull.)
    </small>

    Refinement - **perfect categorization** we see the impact of directly using the ground truth to evaluate to see the theoretical maximum.

    1. Build a classifier that returns the ground truth for each query
    2. Essentially filter (boost very highly) those category / subcategories if present
    """
    )
    return


@app.cell
def _():
    # (use marimo's built-in package management features instead) !pip install git+https://github.com/softwaredoug/cheat-at-search.git
    from cheat_at_search.search import run_strategy, graded_bm25, ndcgs, ndcg_delta, vs_ideal
    from cheat_at_search.wands_data import products

    products
    return graded_bm25, ndcg_delta, ndcgs, products, run_strategy


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Create Cheating Classifier

    Below we use the ground truth to classify our queries perfectly from our labeled data.

    **The Goal** -- understand what MIGHT happen if our zero-shot LLM could always get the right answer. Then we can decide how much we want to invest here.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Recreate Category Ground Truth

    Our ground truth from previous notebooks mapping query -> relevant products -> the dominant categories for those products
    """
    )
    return


@app.cell
def _():
    from cheat_at_search.wands_data import labeled_query_products, queries

    def get_top_category(column, no_fit_label, cutoff=0.8):
        # Get relevant products per query
        top_products = labeled_query_products[labeled_query_products['grade'] == 2]

        # Aggregate top categories
        categories_per_query_ideal = top_products.groupby('query')[column].value_counts().reset_index()

        # Get as percentage of all categories for this query
        top_cat_proportion = categories_per_query_ideal.groupby(['query', column]).sum() / categories_per_query_ideal.groupby('query').sum()
        top_cat_proportion = top_cat_proportion.drop(columns=column).reset_index()

        # Only look at cases where the category is > 0.8
        top_cat_proportion = top_cat_proportion[top_cat_proportion['count'] > cutoff]
        top_cat_proportion[column].fillna(no_fit_label, inplace=True)
        ground_truth_cat = top_cat_proportion
        # Give No Category Fits to all others without dominant category
        ground_truth_cat = ground_truth_cat.merge(queries, how='right', on='query')[['query', column, 'count']]
        ground_truth_cat[column].fillna(no_fit_label, inplace=True)
        return ground_truth_cat

    def get_pred(cat, column):
        if column == 'category':
            return cat.category
        elif column == 'sub_category':
            return cat.sub_category
        else:
            raise ValueError(f"Unknown column {column}")


    def prec_cat(ground_truth, column, no_fit_label, categorized, N=500):
        hits = []
        misses = []
        for _, row in ground_truth.sample(frac=1).iterrows():
            query = row['query']
            expected_category = row[column]

            cat = categorized(query)
            pred = get_pred(cat, column)
            if pred == no_fit_label:
                print(f"Skipping {query}")
                continue
            if pred == expected_category.strip():
                hits.append((expected_category, cat))
            else:
                print("***")
                print(f"{query} -- predicted:{cat.category} != expected:{expected_category.strip()}")
                misses.append((expected_category, cat))
                num_so_far = len(hits) + len(misses)
                print(f"prec (N={num_so_far}) -- {len(hits) / (len(hits) + len(misses))}")
                print(f"coverage {num_so_far / len(ground_truth)}")

            if len(hits) + len(misses) > N:
                break
        return len(hits) / (len(hits) + len(misses)), num_so_far / len(ground_truth)
    return (get_top_category,)


@app.cell
def _(get_top_category):
    ground_truth_cat = get_top_category('category', 'No Category Fits')
    ground_truth_cat
    ground_truth_sub_cat = get_top_category('sub_category', 'No SubCategory Fits')
    ground_truth_sub_cat
    return ground_truth_cat, ground_truth_sub_cat


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Query -> Category classification  structure

    As taken from previous notebooks, our category + subcategory classifications.
    """
    )
    return


@app.cell
def _():
    from pydantic import BaseModel, Field
    from typing import List, Literal
    from cheat_at_search.enrich import AutoEnricher


    Categories = Literal['Furniture',
                         'Home Improvement',
                         'Décor & Pillows',
                         'Outdoor',
                         'Storage & Organization',
                         'Lighting',
                         'Rugs',
                         'Bed & Bath',
                         'Kitchen & Tabletop',
                         'Baby & Kids',
                         'School Furniture and Supplies',
                         'Appliances',
                         'Holiday Décor',
                         'Commercial Business Furniture',
                         'Pet',
                         'Contractor',
                         'Sale',
                         'Foodservice ',
                         'Reception Area',
                         'Clips',
                         'No Category Fits']

    SubCategories = Literal['Bedroom Furniture',
     'Small Kitchen Appliances',
     'All-Clad',
     'Doors & Door Hardware',
     'Bathroom Remodel & Bathroom Fixtures',
     'Home Accessories',
     'Living Room Furniture',
     'Outdoor Décor',
     'Flooring, Walls & Ceiling',
     'Garage & Outdoor Storage & Organization',
     'Cookware & Bakeware',
     'Bedding',
     'Kitchen Utensils & Tools',
     'Shower Curtains & Accessories',
     'Wall Shelving & Organization',
     'Clocks',
     'Bedding Essentials',
     'Kitchen & Dining Furniture',
     'Office Furniture',
     'Tableware & Drinkware',
     'Nursery Bedding',
     'Cat',
     'Outdoor Shades',
     'Outdoor & Patio Furniture',
     'Ceiling Lights',
     'Area Rugs',
     'Outdoor Lighting',
     'Window Treatments',
     'Garden',
     'Closet Storage & Organization',
     'Wall Décor',
     'Mirrors',
     'Shoe Storage',
     'Toddler & Kids Playroom',
     'Game Tables & Game Room Furniture',
     'Decorative Pillows & Blankets',
     'School Furniture',
     'Wall Lights',
     'Bathroom Storage & Organization',
     'Commercial Office Furniture',
     'Flowers & Plants',
     'Mattresses & Foundations',
     'Area Rugs',
     'Cleaning & Laundry Organization',
     'Kitchen Organization',
     'Candles & Holders',
     'Christmas',
     'Toddler & Kids Bedroom Furniture',
     'Front Door Décor & Curb Appeal',
     'Storage Furniture',
     'School Spaces',
     'Hardware',
     'Light Bulbs & Hardware',
     'Ceiling Fans',
     'Doormats',
     'Entry & Hallway',
     'Storage Containers & Drawers',
     'Holiday Lighting',
     'Kitchen Mats',
     'Facilities & Maintenance',
     'Table & Floor Lamps',
     'Bird',
     'Kitchen Appliances',
     'Building Equipment',
     'Art',
     'Picture Frames & Albums',
     'Outdoor Heating',
     'Outdoor Recreation',
     'Bathroom Accessories & Organization',
     'School Boards & Technology',
     'Closeout',
     'Reception Seating',
     'Foodservice Tables',
     'Kitchen Remodel & Kitchen Fixtures',
     'Hot Tubs & Saunas',
     'Teen Bedroom Furniture',
     'Outdoor Fencing & Flooring',
     'Chairs',
     'Bath Rugs & Towels',
     'Fish',
     'Dog',
     'Chicken',
     'Boards & Tech Accessories',
     'Commercial Contractor',
     'Clamps',
     'Jewelry Organization',
     'Entry & Mudroom Furniture',
     'Outdoor Cooking & Tableware',
     'Seasonal Décor',
     'Nursery Furniture',
     'Storage & Organization Sale',
     'Washers & Dryers',
     'Baby & Kids Décor & Lighting',
     'Outdoor Remodel',
     'Plumbing',
     'Birch Lane™',
     'Office Organization',
     'Kitchen & Dining Sale',
     'Holiday Lighting',
     'Baby & Kids Storage',
     'Shop All Characters',
     'Commercial Kitchen',
     'Guest Room Amenities',
     'Charlton Home',
     'Wade Logan®',
     'Heating, Cooling & Air Quality',
     'Thanksgiving',
     'Fourth of July',
     'Vacuums & Deep Cleaners',
     'Stair Tread Rugs',
     'Small Spaces',
     'Toddler & Kids Bedding & Bath',
     'Classroom Décor',
     'Early Education Play Area',
     'Zoomie Kids',
     'Fryers',
     'August Grove',
     'Dorm Décor & Back to School Essentials',
     'Symple Stuff',
     'Wayfair Basics®',
     'The Holiday Aisle',
     'Chair Pads & Cushions',
     'The Monogram Shop',
     'Wedding',
     'Wedding',
     'Reception Desks & Tables',
     'Rug Pads',
     'Latitude Run',
     'Accommodations Furniture',
     'Easter',
     'Furniture Sale',
     'Shop All Characters',
     'Novelty Lights',
     "Valentine's Day",
     'Outdoor Sale',
     'Classroom & Training Furniture',
     'Rebrilliant',
     'Rug Pads',
     'Commercial Kitchen Storage',
     'Teen Bedding',
     'Tommy Bahama Home',
     'Appliances Sale',
     'Massage Products',
     'No SubCategory Fits']

    class Query(BaseModel):
        """
        Base model for search queries, containing common query attributes.
        """
        keywords: str = Field(
            ...,
            description="The original search query keywords sent in as input"
        )

    class QueryCategory(Query):
        """
        Structured representation of a search query for furniture e-commerce.
        Inherits keywords from the base Query model and adds category and sub-category.
        """
        category: Categories = Field(
            description="Category of the product"
        )
        sub_category: SubCategories = Field(
            description="Sub-category of the product"
        )

        @property
        def classification(self) -> str:
            return f"{self.category} / {self.sub_category}"
    return (QueryCategory,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Query classification code (cheating)

    We lookup in the ground truth the category and subcategory and return the right answer
    """
    )
    return


@app.cell
def _(QueryCategory, ground_truth_cat, ground_truth_sub_cat):
    from re import sub
    def categorized(query):
        category = "No Category Fits"
        sub_category = "No SubCategory Fits"
        if query in ground_truth_cat['query'].values:
            cat_at_query = ground_truth_cat[ground_truth_cat['query'] == query]['category']
            cat_at_query = cat_at_query.values[0].strip()
            category = cat_at_query if cat_at_query else "No Category Fits"
        if query in ground_truth_sub_cat['query'].values:
            sub_cat_at_query = ground_truth_sub_cat[ground_truth_sub_cat['query'] == query]['sub_category']
            sub_cat_at_query = sub_cat_at_query.values[0].strip()
            sub_category = sub_cat_at_query if sub_cat_at_query else "No SubCategory Fits"
            print(sub_category)
        return QueryCategory(keywords=query, category=category, sub_category=sub_category)


    categorized("tv stand")
    return (categorized,)


@app.cell
def _(categorized, ground_truth_cat):
    TEST_QUERY = "island estate coffee table"
    ground_truth_cat[ground_truth_cat['query'] == TEST_QUERY], categorized(TEST_QUERY)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Category search strategy

    The same search strategy from before, but here we pass the cheating classifier
    """
    )
    return


@app.cell
def _():
    from searcharray import SearchArray
    from cheat_at_search.tokenizers import snowball_tokenizer
    from cheat_at_search.strategy.strategy import SearchStrategy
    import numpy as np


    class CategorySearch(SearchStrategy):
        def __init__(self, products, query_to_cat,
                     name_boost=9.3,
                     description_boost=4.1,
                     category_boost=100,
                     sub_category_boost=50):
            super().__init__(products)
            self.index = products
            self.index['product_name_snowball'] = SearchArray.index(
                products['product_name'], snowball_tokenizer)
            self.index['product_description_snowball'] = SearchArray.index(
                products['product_description'], snowball_tokenizer)

            cat_split = products['category hierarchy'].fillna('').str.split("/")

            products['category'] = cat_split.apply(
                lambda x: x[0].strip() if len(x) > 0 else ""
            )
            products['subcategory'] = cat_split.apply(
                lambda x: x[1].strip() if len(x) > 1 else ""
            )
            self.index['category_snowball'] = SearchArray.index(
                products['category'], snowball_tokenizer
            )
            self.index['subcategory_snowball'] = SearchArray.index(
                products['subcategory'], snowball_tokenizer
            )

            self.query_to_cat = query_to_cat
            self.name_boost = name_boost
            self.description_boost = description_boost
            self.category_boost = category_boost
            self.sub_category_boost = sub_category_boost

        def search(self, query, k=10):
            """Dumb baseline lexical search, but add a constant boost when
               the desired category or subcategory"""
            bm25_scores = np.zeros(len(self.index))
            structured = self.query_to_cat(query)
            tokenized = snowball_tokenizer(query)
            print(query, structured)

            # ****
            # Baseline BM25 search from before
            for token in tokenized:
                bm25_scores += self.index['product_name_snowball'].array.score(token) * self.name_boost
                bm25_scores += self.index['product_description_snowball'].array.score(
                    token) * self.description_boost

            # ****
            # If there's a subcategory, boost that by a constant amount
            if structured.sub_category and structured.sub_category != "No SubCategory Fits":
                tokenized_subcategory = snowball_tokenizer(structured.sub_category)
                subcategory_match = np.ones(len(self.index))
                if tokenized_subcategory:
                    subcategory_match = self.index['subcategory_snowball'].array.score(tokenized_subcategory) > 0
                bm25_scores[subcategory_match] += self.sub_category_boost

            # ****
            # If there's a category, boost that by a constant amount
            if structured.category and structured.category != "No Category Fits":
                tokenized_category = snowball_tokenizer(structured.category)
                category_match = np.ones(len(self.index))
                if tokenized_category:
                    category_match = self.index['category_snowball'].array.score(tokenized_category) > 0
                bm25_scores[category_match] += self.category_boost

            top_k = np.argsort(-bm25_scores)[:k]
            scores = bm25_scores[top_k]

            return top_k, scores
    return (CategorySearch,)


@app.cell
def _(CategorySearch, categorized, products, run_strategy):
    categorized_search = CategorySearch(products, categorized)
    graded_categorized = run_strategy(categorized_search)
    graded_categorized
    return (graded_categorized,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Analyze the results

    What's the upside / downside?
    """
    )
    return


@app.cell
def _(graded_bm25, graded_categorized, ndcgs):
    ndcgs(graded_bm25).mean(), ndcgs(graded_categorized).mean()
    return


@app.cell
def _(graded_bm25, graded_categorized, ndcg_delta):
    deltas = ndcg_delta(graded_categorized, graded_bm25)
    deltas
    return (deltas,)


@app.cell
def _(deltas):
    sig_improved = len(deltas[deltas > 0.1])
    print(f"Num Significatly Improved: {sig_improved}")
    deltas[deltas > 0.1]
    return (sig_improved,)


@app.cell
def _(deltas, sig_improved):
    sig_harmed = len(deltas[deltas < -0.1])
    print(f"Num Significatly Harmed: {sig_harmed}")
    print(f"Prop improved/harmed: {sig_improved / (sig_harmed + sig_improved)} | {sig_harmed / (sig_harmed + sig_improved)}")
    deltas[deltas < -0.1]
    return


@app.cell
def _(deltas):
    impacted_queries = deltas[deltas > 0.1].index.to_list() + deltas[deltas < -0.1].index.to_list()
    impacted_queries
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Look at a query""")
    return


@app.cell
def _(categorized):
    QUERY = "zodiac pillow"
    categorized(QUERY)
    return (QUERY,)


@app.cell
def _(QUERY, graded_categorized):
    graded_categorized[graded_categorized['query'] == QUERY][['product_name', 'category hierarchy', 'grade']]
    return


@app.cell
def _(QUERY, graded_bm25):
    graded_bm25[graded_bm25['query'] == QUERY][['product_name', 'category hierarchy', 'grade']]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Discussion points / questions

    * Why do we still have a few negative 'deltas'?
    * Is this truly the "upper bound" of having a category / subcategory in our ranking function?
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
