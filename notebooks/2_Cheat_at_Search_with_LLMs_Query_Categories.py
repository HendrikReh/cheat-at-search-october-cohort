import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Query -> Category

    <small>
    (from <a href="http://maven.com/softwaredoug/cheat-at-search">Cheat at Search with LLMs</a> training course by Doug Turnbull.)
    </small>

    We learned that we want to try to model aspects of the user's _information need_ not just make queries better.

    One such common dimension is query -> category classification.

    The Wayfair dataset has an existing classification scheme. We'll use that. It has entries such as:

    ```
    Furniture / Living Room Furniture / Chairs & Seating / Accent Chairs,
    Rugs / Area Rugs,
    ...
    ```

    The top level we'll call "category" the second level we'll call "subcategory"


    In this notebook, we'll use OpenAI to classify our queries into a category and subcategory per query.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Boilerplate

    Install deps, mount GDrive, prompt for your OpenAI Key (placed in your GDrive), and import needed cheat at search helpers.

    We cover this extensively in the [synonyms notebook](https://colab.research.google.com/drive/1aUCvcBa1YdmsbIgYc74jlknl9_iRotp1) walkthrough
    """
    )
    return


@app.cell
def _():
    # (use marimo's built-in package management features instead) !pip install git+https://github.com/softwaredoug/cheat-at-search.git
    from cheat_at_search.data_dir import mount
    mount(use_gdrive=True)
    from cheat_at_search.search import run_strategy, graded_bm25, ndcgs, ndcg_delta, vs_ideal
    from cheat_at_search.wands_data import products

    products
    return graded_bm25, ndcg_delta, ndcgs, products, run_strategy


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Query -> Category classification baseline

    We'll setup a task of classifying query to category and subcategory. Here we have a first-pass baseline that might make sense.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Define allowed output


    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""You'll notice we constrain the allowed output of the LLM to the allowed categories. We do this in Pydantic with a `Literal` type.""")
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
                         'Clips']

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
     'Massage Products']


    class Query(BaseModel):
        """
        Base model for search queries, containing common query attributes.
        """
        keywords: str = Field(
            ...,
            description="The original search query keywords sent in as input"
        )
        query_intent: str = Field(
            description="Explain the intent of this query"
        )

    class QueryCategory(Query):
        """
        Representation of the category and subcategory classification of a search query
        """
        category: Categories = Field(
            description="Category of the product"
        )
        sub_category: SubCategories = Field(
            description="Sub-category of the product"
        )
        labeling_explanation: str = Field(
            description="Why did you label this the way you did?"
        )

        @property
        def classification(self) -> str:
            return f"{self.category} / {self.sub_category}"
    return AutoEnricher, QueryCategory


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Query classification code

    Here we define a prompt and setup the enricher.
    """
    )
    return


@app.cell
def _(AutoEnricher, QueryCategory):
    enricher = AutoEnricher(
         model="openai/gpt-4o",
         system_prompt="You are a helpful furniture shopping agent that helps users construct search queries.",
         response_model=QueryCategory
    )

    def get_prompt(query):
        prompt = f"""
            As a helpful agent, you'll recieve requests from users looking for furniture products.

            Your task is to search with a structured query against a furniture product catalog.

            Here is the users request:

            {query}

            Return Category / Subcategory:

            * Category - the allowed categories (as listed in schema) for the product.
            * SubCategory - the allowed subcategories (as listed in schema) for the product.
        """
        return prompt


    def categorized(query):
        prompt = get_prompt(query)
        return enricher.enrich(prompt)


    categorized("tv stand")
    return (categorized,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Category search strategy

    Below is how we'll use category in search, we

    1. Tokenize category / sub category into their own fields
    2. Predict the category / subcategory for the query
    3. Apply a constant boost (`category_boost` / `sub_category_boost`) to the results below
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

        def __init__(self, products, query_to_cat, name_boost=9.3, description_boost=4.1, category_boost=10, sub_category_boost=5):
            super().__init__(products)
            self.index = products
            self.index['product_name_snowball'] = SearchArray.index(products['product_name'], snowball_tokenizer)
            self.index['product_description_snowball'] = SearchArray.index(products['product_description'], snowball_tokenizer)
            cat_split = products['category hierarchy'].fillna('').str.split('/')
            products['category'] = cat_split.apply(lambda x: x[0].strip() if len(x) > 0 else '')
            products['subcategory'] = cat_split.apply(lambda x: x[1].strip() if len(x) > 1 else '')
            self.index['category_snowball'] = SearchArray.index(products['category'], snowball_tokenizer)
            self.index['subcategory_snowball'] = SearchArray.index(products['subcategory'], snowball_tokenizer)
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
            for token in tokenized:
                bm25_scores = bm25_scores + self.index['product_name_snowball'].array.score(token) * self.name_boost
                bm25_scores = bm25_scores + self.index['product_description_snowball'].array.score(token) * self.description_boost
            if structured.sub_category and structured.sub_category != 'No SubCategory Fits':
                tokenized_subcategory = snowball_tokenizer(structured.sub_category)
                subcategory_match = np.ones(len(self.index))
                if tokenized_subcategory:
                    subcategory_match = self.index['subcategory_snowball'].array.score(tokenized_subcategory) > 0
                bm25_scores[subcategory_match] = bm25_scores[subcategory_match] + self.sub_category_boost
            if structured.category and structured.category != 'No Category Fits':
                tokenized_category = snowball_tokenizer(structured.category)
                category_match = np.ones(len(self.index))
                if tokenized_category:
                    category_match = self.index['category_snowball'].array.score(tokenized_category) > 0
                bm25_scores[category_match] = bm25_scores[category_match] + self.category_boost
            top_k = np.argsort(-bm25_scores)[:k]
            scores = bm25_scores[top_k]
            return (top_k, scores)  # ****  # Baseline BM25 search from before  # ****  # If there's a subcategory, boost that by a constant amount  # ****  # If there's a category, boost that by a constant amount
    return (CategorySearch,)


@app.cell
def _(CategorySearch, categorized, products, run_strategy):
    categorized_search = CategorySearch(products, categorized)
    graded_categorized = run_strategy(categorized_search)
    graded_categorized
    return (graded_categorized,)


@app.cell
def _(graded_categorized):
    graded_categorized['query'].unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Analyze the change

    1. Look at the mean NDCG between the BM25 baseline and this search strategy. We note a nice improvement.

    2. We then take a look at what changes were significantly improved or harmed to ask whether we would ship this to prod?

    3. An we look at individual queries
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
def _(categorized):
    categorized("chair pillow cushion")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Look at a query""")
    return


@app.cell
def _(categorized):
    QUERY = "bathroom vanity knobs"
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
    ## Define a ground truth for categories / subcategories

    We're going to need to focus in on our classifier's specific performance, so we can understand how its performance relates to NDCG improvements, etc.

    This will let us debug our classifier's errors more carefully.
    """
    )
    return


@app.cell
def _():
    CUTOFF = 0.8

    from cheat_at_search.wands_data import labeled_query_products, queries

    # Get relevant products per query
    top_products = labeled_query_products[labeled_query_products['grade'] == 2]
    top_products
    return CUTOFF, labeled_query_products, queries, top_products


@app.cell
def _(top_products):
    # Aggregate top categories
    categories_per_query_ideal = top_products.groupby('query')['category'].value_counts().reset_index()
    categories_per_query_ideal
    return (categories_per_query_ideal,)


@app.cell
def _(CUTOFF, categories_per_query_ideal):
    # Get as percentage of all categories for this query
    top_cat_proportion = categories_per_query_ideal.groupby(['query', 'category']).sum() / categories_per_query_ideal.groupby('query').sum()
    top_cat_proportion = top_cat_proportion.drop(columns='category').reset_index()

    # Only look at cases where the category is > 0.8
    top_cat_proportion = top_cat_proportion[top_cat_proportion['count'] > CUTOFF]
    top_cat_proportion['category'].fillna('No Category Fits', inplace=True)
    ground_truth_cat = top_cat_proportion
    ground_truth_cat
    return (ground_truth_cat,)


@app.cell
def _(ground_truth_cat, queries):
    # Give No Category Fits to all others without dominant category
    ground_truth_cat_1 = ground_truth_cat.merge(queries, how='right', on='query')[['query', 'category', 'count']]
    ground_truth_cat_1['category'].fillna('No Category Fits', inplace=True)
    ground_truth_cat_1
    return (ground_truth_cat_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Category prediction prec of baseline""")
    return


@app.cell
def _(categorized, ground_truth_cat_1):
    def get_pred(cat, column):
        if column == 'category':
            return cat.category
        elif column == 'sub_category':
            return cat.sub_category
        else:
            raise ValueError(f'Unknown column {column}')

    def prec_cat(ground_truth, column, no_fit_label, categorized, N=500):
        _hits = []
        _misses = []
        for _, row in ground_truth.sample(frac=1).iterrows():
            query = row['query']
            expected_category = row[column]
            cat = categorized(query)
            pred = get_pred(cat, column)
            if pred == no_fit_label:
                print(f'Skipping {query}')
                continue
            if pred == expected_category.strip():
                _hits.append((expected_category, cat))
            else:
                print('***')
                print(f'{query} -- predicted:{cat.category} != expected:{expected_category.strip()}')
                _misses.append((expected_category, cat))
                num_so_far = len(_hits) + len(_misses)
                print(f'prec (N={num_so_far}) -- {len(_hits) / (len(_hits) + len(_misses))}')
                print(f'coverage {num_so_far / len(ground_truth)}')
            if len(_hits) + len(_misses) > N:
                break
        return (len(_hits) / (len(_hits) + len(_misses)), _hits, _misses)
    _prec, _hits, _misses = prec_cat(ground_truth_cat_1, 'category', 'No Category Fits', categorized, N=500)
    _prec
    return (prec_cat,)


@app.cell
def _(CUTOFF, labeled_query_products, queries):
    def get_top_category(column, no_fit_label, cutoff=0.8):
        top_products = labeled_query_products[labeled_query_products['grade'] == 2]
        categories_per_query_ideal = top_products.groupby('query')[column].value_counts().reset_index()
        top_cat_proportion = categories_per_query_ideal.groupby(['query', column]).sum() / categories_per_query_ideal.groupby('query').sum()  # Get relevant products per query
        top_cat_proportion = top_cat_proportion.drop(columns=column).reset_index()
        top_cat_proportion = top_cat_proportion[top_cat_proportion['count'] > CUTOFF]
        top_cat_proportion[column].fillna(no_fit_label, inplace=True)  # Aggregate top categories
        ground_truth_cat = top_cat_proportion
        ground_truth_cat = ground_truth_cat.merge(queries, how='right', on='query')[['query', column, 'count']]
        ground_truth_cat[column].fillna(no_fit_label, inplace=True)  # Get as percentage of all categories for this query
        return ground_truth_cat
    ground_truth_sub_cat = get_top_category('sub_category', 'No SubCategory Fits')
    ground_truth_sub_cat  # Only look at cases where the category is > 0.8  # Give No Category Fits to all others without dominant category
    return (ground_truth_sub_cat,)


@app.cell
def _(categorized, ground_truth_sub_cat, prec_cat):
    _prec, _hits, _misses = prec_cat(ground_truth_sub_cat, 'sub_category', 'No SubCategory Fits', categorized, N=500)
    _prec
    return


@app.cell
def _(categorized, ground_truth_cat_1, prec_cat):
    impacted_queries = ['drum picture', 'bathroom freestanding cabinet', 'outdoor lounge chair', 'wood rack wide', 'outdoor light fixtures', 'bathroom vanity knobs', 'door jewelry organizer', 'beds that have leds', 'non slip shower floor tile', 'turquoise chair', 'modern outdoor furniture', 'podium with locking cabinet', 'closet storage with zipper', 'barstool patio sets', 'ayesha curry kitchen', 'led 60', 'wisdom stone river 3-3/4', 'liberty hardware francisco', 'french molding', 'glass doors for bath', 'accent leather chair', 'dark gray dresser', 'wainscoting ideas', 'floating bed', 'dining table vinyl cloth', 'entrance table', 'storage dresser', 'almost heaven sauna', 'toddler couch fold out', 'outdoor welcome rug', 'wooden chair outdoor', 'emma headboard', 'outdoor privacy wall', 'driftwood mirror', 'white abstract', 'bedroom accessories', 'bathroom lighting', 'light and navy blue decorative pillow', 'gnome fairy garden', 'medium size chandelier', 'above toilet cabinet', 'odum velvet', 'ruckus chair', 'modern farmhouse lighting semi flush mount', 'teal chair', 'bedroom wall decor floral, multicolored with some teal (prints)', 'big basket for dirty cloths', 'milk cow chair', 'small wardrobe grey', 'glow in the dark silent wall clock', 'medium clips', 'desk for kids tjat ate 10 year old', 'industrial pipe dining  table', 'itchington butterfly', 'midcentury tv unit', 'gas detector', 'fleur de lis living candle wall sconce bronze', 'zodiac pillow', 'papasan chair frame only', 'bed side table']
    _prec, _hits, _misses = prec_cat(ground_truth_cat_1[ground_truth_cat_1['query'].isin(impacted_queries)], 'category', 'No Category Fits', categorized, N=500)
    _prec
    return


@app.cell
def _(products):
    products['category hierarchy'].unique()
    return


@app.cell
def _(labeled_query_products):
    labeled_query_products[labeled_query_products['grade'] == 2][['query', 'category hierarchy', 'grade']]
    # What category / subcategory occurs in > 80% of the relevant results for each query
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

