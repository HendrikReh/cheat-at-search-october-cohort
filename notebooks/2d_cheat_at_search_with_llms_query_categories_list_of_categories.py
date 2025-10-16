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

    **Refinement** -- Use the -full list- of classifications, and allow a list to be returned

    We learned that we want to try to model aspects of the user's _information need_ not just make queries better.

    One such common dimension is query -> category classification
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Import helpers

    Helpers to compute ndcg of each query and other comparison functions
    """
    )
    return


@app.cell
def _():
    from cheat_at_search.data_dir import ensure_data_subdir
    from cheat_at_search.search import run_strategy, graded_bm25, ndcgs, ndcg_delta, vs_ideal
    return graded_bm25, ndcg_delta, ndcgs, run_strategy


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Import WANDS data

    Import [Wayfair Annotated Dataset](https://github.com/wayfair/WANDS) a labeled furniture e-commerce dataset
    """
    )
    return


@app.cell
def _():
    from cheat_at_search.wands_data import products

    products
    return (products,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Query -> Full classification

    We'll first setup the model of query -> category and subcategory. We've precurated categories / subcategories for you to help.
    """
    )
    return


@app.cell
def _():
    from typing import Literal, List, get_args
    from pydantic import BaseModel, Field
    from cheat_at_search.enrich import AutoEnricher
    FullyQualifiedClassifications = Literal['Furniture / Bedroom Furniture / Beds & Headboards / Beds', 'Furniture / Living Room Furniture / Chairs & Seating / Accent Chairs', 'Rugs / Area Rugs', 'Furniture / Office Furniture / Desks', 'Furniture / Living Room Furniture / Coffee Tables & End Tables / Coffee Tables', 'Furniture / Living Room Furniture / Coffee Tables & End Tables / End & Side Tables', 'Décor & Pillows / Decorative Pillows & Blankets / Throw Pillows', 'Furniture / Bedroom Furniture / Dressers & Chests', 'Outdoor / Outdoor & Patio Furniture / Patio Furniture Sets / Patio Conversation Sets', 'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Bathroom Vanities / All Bathroom Vanities', 'Furniture / Living Room Furniture / Console Tables', 'Décor & Pillows / Art / All Wall Art', 'Furniture / Kitchen & Dining Furniture / Bar Furniture / Bar Stools & Counter Stools / All Bar Stools & Counter Stools', 'Furniture / Kitchen & Dining Furniture / Dining Tables & Seating / Kitchen & Dining Chairs', 'Furniture / Office Furniture / Office Chairs', 'Décor & Pillows / Mirrors / All Mirrors', 'Bed & Bath / Bedding / All Bedding', 'Décor & Pillows / Wall Décor / Wall Accents', 'Furniture / Living Room Furniture / Chairs & Seating / Recliners', 'Furniture / Kitchen & Dining Furniture / Dining Tables & Seating / Kitchen and Dining Sets', 'Décor & Pillows / Window Treatments / Curtains & Drapes', 'Furniture / Living Room Furniture / Sectionals', 'Baby & Kids / Toddler & Kids Bedroom Furniture / Kids Beds', 'Furniture / Living Room Furniture / TV Stands & Media Storage Furniture / TV Stands & Entertainment Centers', 'Lighting / Ceiling Lights / Chandeliers', 'Furniture / Bedroom Furniture / Nightstands', 'Baby & Kids / Toddler & Kids Bedroom Furniture / Kids Desks', 'Décor & Pillows / Home Accessories / Decorative Objects', 'Furniture / Bedroom Furniture / Beds & Headboards / Headboards', 'Furniture / Living Room Furniture / Sofas', 'Furniture / Living Room Furniture / Cabinets & Chests', 'Décor & Pillows / Clocks / Wall Clocks', 'Storage & Organization / Bathroom Storage & Organization / Bathroom Cabinets & Shelving', 'Lighting / Table & Floor Lamps / Table Lamps', 'Furniture / Living Room Furniture / Ottomans & Poufs', 'Furniture / Kitchen & Dining Furniture / Kitchen Islands & Carts', 'Furniture / Living Room Furniture / Bookcases', 'Outdoor / Outdoor & Patio Furniture / Outdoor Seating & Patio Chairs / Patio Seating / Patio Sofas & Sectionals', 'Furniture / Office Furniture / Office Storage Cabinets', 'Furniture / Kitchen & Dining Furniture / Dining Tables & Seating / Kitchen & Dining Tables', 'Contractor / Entry & Hallway / Coat Racks & Umbrella Stands', 'Bed & Bath / Bedding Essentials / Mattress Pads & Toppers', 'Home Improvement / Hardware / Home Hardware / Switch Plates', 'Baby & Kids / Toddler & Kids Playroom / Playroom Furniture / Toddler & Kids Chairs & Seating', 'Storage & Organization / Garage & Outdoor Storage & Organization / Outdoor Covers / Patio Furniture Covers', 'Rugs / Doormats', 'Rugs / Kitchen Mats', 'Furniture / Bedroom Furniture / Beds & Headboards / Beds / Queen Size Beds', 'Furniture / Bedroom Furniture / Daybeds', 'Furniture / Living Room Furniture / Living Room Sets', 'Outdoor / Outdoor & Patio Furniture / Patio Furniture Sets / Patio Dining Sets', 'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Bathroom Sinks & Faucet Components / Bathroom Sink Faucets / Single Hole Bathroom Sink Faucets', 'Outdoor / Outdoor Décor / Statues & Sculptures', 'Décor & Pillows / Art / All Wall Art / Green Wall Art', 'Furniture / Living Room Furniture / Coffee Tables & End Tables / Coffee Table Sets', 'Furniture / Living Room Furniture / Chairs & Seating / Chaise Lounge Chairs', 'Storage & Organization / Wall Shelving & Organization / Wall and Display Shelves', 'Furniture / Living Room Furniture / Coffee Tables & End Tables / Coffee Tables / Rectangle Coffee Tables', 'Décor & Pillows / Art / All Wall Art / Brown Wall Art', 'Furniture / Kitchen & Dining Furniture / Bar Furniture / Bar Stools & Counter Stools / All Bar Stools & Counter Stools / Counter (24-27) Bar Stools & Counter Stools', 'Furniture / Living Room Furniture / Coffee Tables & End Tables / Plant Stands & Tables', 'Décor & Pillows / Window Treatments / Curtain Hardware & Accessories', 'Furniture / Kitchen & Dining Furniture / Dining Tables & Seating / Kitchen & Dining Chairs / Side Kitchen & Dining Chairs', 'Outdoor / Outdoor & Patio Furniture / Outdoor Seating & Patio Chairs / Patio Seating / Outdoor Club Chairs', 'Furniture / Living Room Furniture / Chairs & Seating / Benches', 'Home Improvement / Kitchen Remodel & Kitchen Fixtures / Kitchen Sinks & Faucet Components / Kitchen Sinks / Farmhouse & Apron Kitchen Sinks', 'Kitchen & Tabletop / Kitchen Organization / Food Pantries', 'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Towel Storage / Towel & Robe Hooks / Black Towel & Robe Hooks', 'Storage & Organization / Garage & Outdoor Storage & Organization / Deck Boxes & Patio Storage', 'Outdoor / Garden / Planters', 'Lighting / Wall Lights / Bathroom Vanity Lighting', 'Furniture / Kitchen & Dining Furniture / Sideboards & Buffets', 'Storage & Organization / Garage & Outdoor Storage & Organization / Storage Racks & Shelving Units', 'Home Improvement / Hardware / Cabinet Hardware / Cabinet & Drawer Pulls / Bronze Cabinet & Drawer Pulls', 'Storage & Organization / Storage Containers & Drawers / All Storage Containers', 'Bed & Bath / Shower Curtains & Accessories / Shower Curtains & Shower Liners', 'Storage & Organization / Bathroom Storage & Organization / Hampers & Laundry Baskets', 'Lighting / Light Bulbs & Hardware / Light Bulbs / All Light Bulbs / LED Light Bulbs', 'Décor & Pillows / Art / All Wall Art / Blue Wall Art', 'Bed & Bath / Mattresses & Foundations / Innerspring Mattresses', 'Lighting / Outdoor Lighting / Outdoor Wall Lighting', 'Storage & Organization / Garage & Outdoor Storage & Organization / Natural Material Storage / Log Storage', 'Bed & Bath / Bathroom Accessories & Organization / Countertop Bath Accessories', 'Storage & Organization / Shoe Storage / All Shoe Storage', 'Home Improvement / Flooring, Walls & Ceiling / Floor Tiles & Wall Tiles / Ceramic Floor Tiles & Wall Tiles', 'Home Improvement / Hardware / Cabinet Hardware / Cabinet & Drawer Pulls / Black Cabinet & Drawer Pulls', 'Bed & Bath / Mattresses & Foundations / Adjustable Beds', "Rugs / Area Rugs / 2' x 3' Area Rugs", 'Commercial Business Furniture / Commercial Office Furniture / Office Storage & Filing / Office Carts & Stands / All Carts & Stands', 'Furniture / Bedroom Furniture / Beds & Headboards / Beds / Twin Beds', 'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Bathroom Sinks & Faucet Components / Bathroom Sink Faucets / Widespread Bathroom Sink Faucets', "Rugs / Area Rugs / 4' x 6' Area Rugs", 'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Bathroom Sinks & Faucet Components / Bathroom Sink Faucets', 'Kitchen & Tabletop / Tableware & Drinkware / Table & Kitchen Linens / All Table Linens', 'Kitchen & Tabletop / Kitchen Organization / Food Storage & Canisters / Food Storage Containers', 'Décor & Pillows / Flowers & Plants / Faux Flowers', 'Bed & Bath / Bedding / All Bedding / Twin Bedding', 'Furniture / Bedroom Furniture / Dressers & Chests / White Dressers & Chests', 'Home Improvement / Flooring, Walls & Ceiling / Floor Tiles & Wall Tiles / Porcelain Floor Tiles & Wall Tiles', 'Home Improvement / Flooring, Walls & Ceiling / Flooring Installation & Accessories / Molding & Millwork / Wall Molding & Millwork', 'Home Improvement / Doors & Door Hardware / Door Hardware & Accessories / Barn Door Hardware', 'Bed & Bath / Bedding / Sheets & Pillowcases', 'Furniture / Office Furniture / Chair Mats / Hard Floor Chair Mats', 'Outdoor / Outdoor Fencing & Flooring / All Fencing', 'Storage & Organization / Closet Storage & Organization / Clothes Racks & Garment Racks', 'Kitchen & Tabletop / Kitchen Utensils & Tools / Colanders, Strainers, & Salad Spinners', 'Outdoor / Hot Tubs & Saunas / Saunas', 'Décor & Pillows / Decorative Pillows & Blankets / Throw Pillows / Blue Throw Pillows', 'Bed & Bath / Bedding Essentials / Bed Pillows', 'Lighting / Wall Lights / Wall Sconces', 'Outdoor / Front Door Décor & Curb Appeal / Mailboxes', 'Outdoor / Garden / Greenhouses', 'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Showers & Bathtubs / Showers & Bathtubs Plumbing / Shower Faucets & Systems', 'Bed & Bath / Mattresses & Foundations / Queen Mattresses', 'Furniture / Bedroom Furniture / Jewelry Armoires', 'Outdoor / Outdoor Shades / Awnings', 'Baby & Kids / Nursery Bedding / Crib Bedding Sets', 'Home Improvement / Hardware / Cabinet Hardware / Cabinet & Drawer Knobs / Brass Cabinet & Drawer Knobs', 'Décor & Pillows / Art / All Wall Art / Red Wall Art', 'Lighting / Ceiling Lights / All Ceiling Lights', 'Lighting / Light Bulbs & Hardware / Lighting Components', 'Furniture / Game Tables & Game Room Furniture / Poker & Card Tables', 'Appliances / Kitchen Appliances / Range Hoods / All Range Hoods', 'Home Improvement / Flooring, Walls & Ceiling / Floor Tiles & Wall Tiles / Natural Stone Floor Tiles & Wall Tiles', 'Furniture / Kitchen & Dining Furniture / Bar Furniture / Bar Stools & Counter Stools / All Bar Stools & Counter Stools / Bar (28-33) Bar Stools & Counter Stools', 'Outdoor / Outdoor Cooking & Tableware / Outdoor Serving & Tableware / Coolers, Baskets & Tubs / Picnic Baskets & Backpacks', 'Décor & Pillows / Picture Frames & Albums / All Picture Frames', 'Bed & Bath / Shower Curtains & Accessories / Shower Curtain Hooks', 'Outdoor / Outdoor Shades / Outdoor Umbrellas / Patio Umbrella Stands & Bases', 'Outdoor / Outdoor & Patio Furniture / Patio Bar Furniture / Patio Bar Stools', 'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Toilets & Bidets / Toilet Paper Holders / Free Standing Toilet Paper Holders', 'Storage & Organization / Garage & Outdoor Storage & Organization / Bike & Sport Racks', 'Appliances / Kitchen Appliances / Refrigerators & Freezers / All Refrigerators / French Door Refrigerators', 'Décor & Pillows / Home Accessories / Decorative Trays', 'School Furniture and Supplies / School Spaces / Computer Lab Furniture / Podiums & Lecterns', 'Lighting / Light Bulbs & Hardware / Lighting Shades', 'Furniture / Kitchen & Dining Furniture / Bar Furniture / Home Bars & Bar Sets', 'Lighting / Table & Floor Lamps / Floor Lamps', 'Décor & Pillows / Wall Décor / Wall Accents / Brown Wall Accents', 'Kitchen & Tabletop / Small Kitchen Appliances / Pressure & Slow Cookers / Slow Cookers / Slow Slow Cookers', 'Décor & Pillows / Window Treatments / Curtains & Drapes / 90 Inch Curtains & Drapes', 'Furniture / Bedroom Furniture / Armoires & Wardrobes', 'Kitchen & Tabletop / Tableware & Drinkware / Flatware & Cutlery / Serving Utensils', 'Baby & Kids / Baby & Kids Décor & Lighting / All Baby & Kids Wall Art', 'Furniture / Office Furniture / Desks / Writing Desks', 'Furniture / Office Furniture / Office Chairs / Task Office Chairs', 'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Showers & Bathtubs / Shower & Bathtub Doors', 'Outdoor / Outdoor & Patio Furniture / Outdoor Seating & Patio Chairs / Patio Seating / Patio Rocking Chairs & Gliders', 'Home Improvement / Flooring, Walls & Ceiling / Walls & Ceilings / Wall Paneling', 'Outdoor / Garden / Plant Stands & Accessories', 'Furniture / Kitchen & Dining Furniture / Dining Tables & Seating / Kitchen & Dining Tables / 4 Seat Kitchen & Dining Tables', 'Décor & Pillows / Home Accessories / Vases, Urns, Jars & Bottles', 'Lighting / Wall Lights / Under Cabinet Lighting / Strip Under Cabinet Lighting', 'Furniture / Bedroom Furniture / Bedroom and Makeup Vanities', 'Pet / Dog / Dog Bowls & Feeding Supplies / Pet Bowls & Feeders', 'Décor & Pillows / Candles & Holders / Candle Holders', 'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Showers & Bathtubs / Shower & Bathtub Accessories', 'Furniture / Office Furniture / Office Chair Accessories / Seat Cushion Office Chair Accessories', 'Furniture / Office Furniture / Chair Mats', 'Furniture / Living Room Furniture / Chairs & Seating / Massage Chairs', 'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Bathroom Vanities / All Bathroom Vanities / Modern & Contemporary Bathroom Vanities', 'Lighting / Ceiling Fans / All Ceiling Fans', 'Home Improvement / Kitchen Remodel & Kitchen Fixtures / Kitchen Sinks & Faucet Components / Kitchen Faucets / Black Kitchen Faucets', 'Lighting / Light Bulbs & Hardware / Light Bulbs / All Light Bulbs / Incandescent Light Bulbs', 'Home Improvement / Flooring, Walls & Ceiling / Flooring Installation & Accessories / Molding & Millwork', 'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Showers & Bathtubs / Bathtubs', 'Décor & Pillows / Art / All Wall Art / Yellow Wall Art', 'Pet / Dog / Pet Gates, Fences & Doors / Pet Gates', 'Furniture / Bedroom Furniture / Beds & Headboards / Bed Frames / Twin Bed Frames', 'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Towel Storage / Towel Bars, Racks, and Stands / Metal Towel Bars, Racks, and Stands', 'Décor & Pillows / Art / All Wall Art / Pink Wall Art', 'Home Improvement / Kitchen Remodel & Kitchen Fixtures / Smoke Detectors / Wall & Ceiling Mounted Smoke Detectors', 'Outdoor / Garden / Planters / Plastic Planters', 'Décor & Pillows / Mirrors / All Mirrors / Accent Mirrors', 'Appliances / Kitchen Appliances / Range Hoods / All Range Hoods / Wall Mount Range Hoods', 'Outdoor / Garden / Garden Décor / Lawn & Garden Accents', 'Furniture / Living Room Furniture / Coffee Tables & End Tables / Coffee Tables / Round Coffee Tables', 'Kitchen & Tabletop / Tableware & Drinkware / Dinnerware / Dining Bowls', 'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Showers & Bathtubs / Showers & Bathtubs Plumbing / Shower Heads / Dual Shower Heads', 'Home Improvement / Flooring, Walls & Ceiling / Floor Tiles & Wall Tiles / Glass Floor Tiles & Wall Tiles', 'School Furniture and Supplies / Facilities & Maintenance / Trash & Recycling', 'Home Improvement / Hardware / Cabinet Hardware / Cabinet & Drawer Pulls / Nickel Cabinet & Drawer Pulls', 'Storage & Organization / Closet Storage & Organization / Closet Systems', 'Furniture / Bedroom Furniture / Beds & Headboards / Beds / Full & Double Beds', 'Commercial Business Furniture / Commercial Office Furniture / Office Storage & Filing / Office Carts & Stands / All Carts & Stands / Printer Carts & Stands', 'Storage & Organization / Closet Storage & Organization / Closet Accessories', 'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Bathroom Vanities / All Bathroom Vanities / Traditional Bathroom Vanities', 'Home Improvement / Plumbing / Core Plumbing / Parts & Components', 'Holiday Décor / Christmas / Christmas Trees / All Christmas Trees', 'Décor & Pillows / Decorative Pillows & Blankets / Throw Pillows / Black Throw Pillows', 'Furniture / Game Tables & Game Room Furniture / Sports Team Fan Shop & Memorabillia / Life Size Cutouts', 'Lighting / Ceiling Lights / Pendant Lighting', 'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Towel Storage / Towel & Robe Hooks', 'Appliances / Washers & Dryers / Dryers / All Dryers / Gas Dryers', 'Outdoor / Outdoor Recreation / Backyard Play / Kids Cars & Ride-On Toys', 'Kitchen & Tabletop / Small Kitchen Appliances / Coffee, Espresso, & Tea / Coffee Makers', 'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Showers & Bathtubs / Showers & Bathtubs Plumbing / Shower Heads', 'Outdoor / Outdoor & Patio Furniture / Outdoor Seating & Patio Chairs / Patio Seating / Patio Sofas & Sectionals / Sectional Patio Sofas & Sectionals', 'Lighting / Wall Lights / Under Cabinet Lighting', 'Foodservice / Foodservice Tables / Table Parts', 'Lighting / Outdoor Lighting / Landscape Lighting / All Landscape Lighting / Fence Post Cap Landscape Lighting', 'Lighting / Outdoor Lighting / Landscape Lighting / All Landscape Lighting', 'Outdoor / Outdoor & Patio Furniture / Outdoor Tables / All Patio Tables', 'Commercial Business Furniture / Commercial Office Furniture / Office Storage & Filing / Office Carts & Stands / All Carts & Stands / Utility Carts & Stands', 'Outdoor / Outdoor & Patio Furniture / Outdoor Seating & Patio Chairs / Outdoor Chaise & Lounge Chairs', 'Furniture / Living Room Furniture / Chairs & Seating / Recliners / Brown Recliners', 'Pet / Bird / Bird Perches & Play Gyms', 'Décor & Pillows / Picture Frames & Albums / All Picture Frames / Single Picture Picture Frames', 'Lighting / Outdoor Lighting / Outdoor Lanterns & Lamps', 'Home Improvement / Hardware / Cabinet Hardware / Cabinet & Drawer Pulls', 'Bed Accessories', 'Clips/Clamps', 'Décor & Pillows / Wall Décor / Wall Decals', 'Home Improvement / Flooring, Walls & Ceiling / Floor Tiles & Wall Tiles', 'Bed & Bath / Bedding / Sheets & Pillowcases / Twin XL Sheets & Pillowcases', 'Kitchen & Tabletop / Tableware & Drinkware / Serveware / Serving Trays & Boards / Serving Trays & Platters / Serving Serving Trays & Platters', 'Holiday Décor / Holiday Lighting', 'Décor & Pillows / Wall Décor / Memo Boards', 'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Toilets & Bidets / Toilet Paper Holders / Wall Mounted Toilet Paper Holders', 'Décor & Pillows / Window Treatments / Curtains & Drapes / 63 Inch and Less Curtains & Drapes', 'Home Improvement / Doors & Door Hardware / Door Hardware & Accessories / Door Knobs / Egg Door Knobs', 'Décor & Pillows / Clocks / Wall Clocks / Analog Wall Clocks', 'Home Improvement / Doors & Door Hardware / Interior Doors / Sliding Interior Doors', 'Outdoor / Outdoor Recreation / Outdoor Games / All Outdoor Games', 'Home Improvement / Doors & Door Hardware / Door Hardware & Accessories / Door Levers / Round Door Levers', 'Storage & Organization / Garage & Outdoor Storage & Organization / Sheds / Storage Sheds', 'Home Improvement / Doors & Door Hardware / Door Hardware & Accessories / Door Levers', 'School Furniture and Supplies / School Furniture / School Tables / Folding Tables / Wood Folding Tables', 'Décor & Pillows / Wall Décor / Wall Accents / Green Wall Accents', 'School Furniture and Supplies / Facilities & Maintenance / Commercial Signage', 'Storage & Organization / Garage & Outdoor Storage & Organization / Garage Storage Cabinets', 'Furniture / Bedroom Furniture / Dressers & Chests / Beige Dressers & Chests', 'Storage & Organization / Wall Shelving & Organization / Wall & Display Shelves', 'Furniture / Game Tables & Game Room Furniture / Dartboards & Cabinets', 'Outdoor / Outdoor Décor / Outdoor Pillows & Cushions / Patio Furniture Cushions / Lounge Chair Patio Furniture Cushions', 'Outdoor / Outdoor & Patio Furniture / Patio Furniture Sets / Patio Dining Sets / Two Person Patio Dining Sets', 'Décor & Pillows / Decorative Pillows & Blankets / Throw Pillows / Ivory & Cream Throw Pillows', 'Appliances / Washers & Dryers / Washer & Dryer Sets / Black Washer & Dryer Sets', 'School Furniture and Supplies / School Furniture / School Chairs & Seating / Stackable Chairs', 'Home Improvement / Hardware / Cabinet Hardware / Cabinet & Drawer Pulls / Brass Cabinet & Drawer Pulls', 'School Furniture and Supplies / School Boards & Technology / AV, Mounts & Tech Accessories / Electronic Mounts & Stands / Computer Mounts', 'Furniture / Living Room Furniture / Chairs & Seating / Accent Chairs / Papasan Accent Chairs', 'Storage & Organization / Shoe Storage / All Shoe Storage / Rack Shoe Storage', 'Storage & Organization / Shoe Storage / All Shoe Storage / Cabinet Shoe Storage', 'Storage & Organization / Storage Containers & Drawers / Storage Drawers', 'Appliances / Kitchen Appliances / Wine & Beverage Coolers / Water Coolers', 'Furniture / Living Room Furniture / Chairs & Seating / Rocking Chairs', 'Kitchen & Tabletop / Tableware & Drinkware / Serveware / Serving Bowls & Baskets / Serving Bowls / NA Serving Bowls', 'Furniture / Living Room Furniture / TV Stands & Media Storage Furniture / Projection Screens / Inflatable Projection Screens', 'Appliances / Kitchen Appliances / Large Appliance Parts & Accessories', 'Storage & Organization / Bathroom Storage & Organization / Hampers & Laundry Baskets / Laundry Hampers & Laundry Baskets', 'Furniture / Office Furniture / Office Stools', 'Outdoor / Outdoor & Patio Furniture / Outdoor Seating & Patio Chairs / Patio Seating / Outdoor Club Chairs / Metal Outdoor Club Chairs', 'School Furniture and Supplies / School Furniture / School Tables / Folding Tables', 'Lighting / Wall Lights / Bathroom Vanity Lighting / Traditional Bathroom Vanity Lighting', 'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Bathroom Sinks & Faucet Components / Bathroom Sink Faucets / Centerset Bathroom Sink Faucets', 'Décor & Pillows / Flowers & Plants / Faux Flowers / Orchid Faux Flowers', 'Home Improvement / Flooring, Walls & Ceiling / Floor Tiles & Wall Tiles / Metal Floor Tiles & Wall Tiles', 'Home Improvement / Kitchen Remodel & Kitchen Fixtures / Kitchen Sinks & Faucet Components / Kitchen Sinks', 'Storage & Organization / Garage & Outdoor Storage & Organization / Outdoor Covers / Grill Covers / Charcoal Grill Grill Covers', 'Outdoor / Outdoor Décor / Outdoor Wall Décor', 'Storage & Organization / Cleaning & Laundry Organization / Laundry Room Organizers', 'Reception Area / Reception Seating / Reception Sofas & Loveseats', 'Kitchen & Tabletop / Cookware & Bakeware / Baking Sheets & Pans / Bread & Loaf Pans / Steel Bread & Loaf Pans', 'Furniture / Living Room Furniture / Chairs & Seating / Accent Chairs / Wingback Accent Chairs', 'Home Improvement / Bathroom Remodel & Bathroom Fixtures / Showers & Bathtubs / Showers & Bathtubs Plumbing / Shower Heads / Fixed Shower Heads', 'Kitchen & Tabletop / Kitchen Utensils & Tools / Kitchen Gadgets / Pasta Makers & Accessories', 'School Furniture and Supplies / School Furniture / School Chairs & Seating / Classroom Chairs / High School & College Classroom Chairs', 'Furniture / Living Room Furniture / Sectionals / Stationary Sectionals', 'Furniture / Kitchen & Dining Furniture / Sideboards & Buffets / Drawer Equipped Sideboards & Buffets', 'Kitchen & Tabletop / Cookware & Bakeware / Baking Sheets & Pans / Bread & Loaf Pans', 'Kitchen & Tabletop / Kitchen Utensils & Tools / Cooking Utensils / All Cooking Utensils / Kitchen Cooking Utensils', 'Décor & Pillows / Flowers & Plants / Live Plants', 'Furniture / Living Room Furniture / TV Stands & Media Storage Furniture / Projection Screens / Folding Frame Projection Screens', 'Kitchen & Tabletop / Kitchen Organization / Food Storage & Canisters / Kitchen Canisters & Jars / Metal Kitchen Canisters & Jars', 'Outdoor / Outdoor Décor / Outdoor Fountains', 'Outdoor / Outdoor Shades / Pergolas / Wood Pergolas', 'Décor & Pillows / Candles & Holders / Candle Holders / Sconce Candle Holders', 'Kitchen & Tabletop / Tableware & Drinkware / Serveware / Cake & Tiered Stands', 'Home Improvement / Kitchen Remodel & Kitchen Fixtures / Kitchen Sinks & Faucet Components / Kitchen Faucets / Chrome Kitchen Faucets', 'Décor & Pillows / Decorative Pillows & Blankets / Throw Pillows / White Throw Pillows', 'Outdoor / Outdoor Fencing & Flooring / Turf', 'Décor & Pillows / Window Treatments / Valances & Kitchen Curtains', 'Home Improvement / Hardware / Cabinet Hardware / Cabinet & Drawer Knobs / Black Cabinet & Drawer Knobs', 'Home Improvement / Kitchen Remodel & Kitchen Fixtures / Kitchen Sinks & Faucet Components / Kitchen Faucets / Bronze Kitchen Faucets', 'Appliances / Washers & Dryers / Washer & Dryer Sets', 'Décor & Pillows / Clocks / Mantel & Tabletop Clocks', 'Home Improvement / Doors & Door Hardware / Interior Doors', 'Storage & Organization / Wall Shelving & Organization / Wall & Display Shelves / Floating Wall & Display Shelves', 'Outdoor / Outdoor Recreation / Backyard Play / Climbing Toys & Slides', 'Home Improvement / Building Equipment / Dollies / Hand Truck Dollies', 'Baby & Kids / Toddler & Kids Bedroom Furniture / Baby & Kids Dressers', 'Décor & Pillows / Mirrors / All Mirrors / Leaning & Floor Mirrors', 'Kitchen & Tabletop / Tableware & Drinkware / Drinkware / Mugs & Teacups', 'Décor & Pillows / Flowers & Plants / Wreaths', 'Outdoor / Outdoor Shades / Pergolas / Metal Pergolas', 'Bed & Bath / Bedding / Sheets & Pillowcases / Twin Sheets & Pillowcases', 'Outdoor / Outdoor Shades / Pergolas', 'Reception Area / Reception Seating / Office Sofas & Loveseats', 'Décor & Pillows / Home Accessories / Indoor Fountains', 'Kitchen & Tabletop / Kitchen Organization / Food Storage & Canisters / Kitchen Canisters & Jars / Ceramic Kitchen Canisters & Jars', 'Décor & Pillows / Window Treatments / Curtain Hardware & Accessories / Bracket Curtain Hardware & Accessories', 'Home Improvement / Flooring, Walls & Ceiling / Walls & Ceilings / Accent Tiles / Ceramic Accent Tiles', 'Home Improvement / Flooring, Walls & Ceiling / Walls & Ceilings / Accent Tiles', 'Furniture / Living Room Furniture / Chairs & Seating / Accent Chairs / Arm Accent Chairs', 'Furniture / Living Room Furniture / Coffee Tables & End Tables / Coffee Tables / Free Form Coffee Tables', 'Décor & Pillows / Flowers & Plants / Faux Flowers / Rose Faux Flowers', 'Bed & Bath / Mattresses & Foundations / Innerspring Mattresses / Twin Innerspring Mattresses', 'Outdoor / Outdoor Décor / Outdoor Pillows & Cushions / Patio Furniture Cushions / Dining Chair Patio Furniture Cushions', 'Furniture / Living Room Furniture / TV Stands & Media Storage Furniture / TV Stands & Entertainment Centers / Traditional TV Stands & Entertainment Centers', 'Furniture / Living Room Furniture / Coffee Tables & End Tables / Plant Stands & Tables / Square Plant Stands & Tables', 'Storage & Organization / Wall Shelving & Organization / Wall & Display Shelves / Corner Wall & Display Shelves', "Rugs / Area Rugs / 3' x 5' Area Rugs", 'Kitchen & Tabletop / Tableware & Drinkware / Drinkware / Mugs & Teacups / Coffee Mugs & Teacups', 'Contractor / Entry & Hallway / Coat Racks & Umbrella Stands / Wall Mounted Coat Racks & Umbrella Stands', "Baby & Kids / Toddler & Kids Playroom / Indoor Play / Kids' Playhouses", 'Furniture / Living Room Furniture / Coffee Tables & End Tables / Coffee Tables / Square Coffee Tables', 'Baby & Kids / Toddler & Kids Playroom / Indoor Play / Dollhouses & Accessories', 'Bed & Bath / Bedding / All Bedding / Queen Bedding', 'No Classification Fits']
    classifications_list = sorted(get_args(FullyQualifiedClassifications))
    known_categories = set([c.split(' / ')[0].strip() for c in classifications_list])
    known_sub_categories = set([c.split(' / ')[1].strip() for c in classifications_list if len(c.split(' / ')) > 1])
    known_sub_categories

    class Query(BaseModel):
        """
        Base model for search queries, containing common query attributes.
        """
        keywords: str = Field(..., description='The original search query keywords sent in as input')

    class QueryClassification(Query):
        """
        Structured representation of a search query for furniture e-commerce.
        Inherits keywords from the base Query model and adds category and sub-category.
        """
        classifications: list[FullyQualifiedClassifications] = Field(description='A possible classification for the product.')

        @property
        def categories(self):
            return set([c.split(' / ')[0] for c in self.classifications])

        @property
        def sub_categories(self):
            return set([c.split(' / ')[1] for c in self.classifications if len(c.split(' / ')) > 1])
    return AutoEnricher, QueryClassification


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Query classification code""")
    return


@app.cell
def _(AutoEnricher, QueryClassification):
    enricher = AutoEnricher(
         model="openai/gpt-4o",
         system_prompt="You are a helpful furniture shopping agent that helps users construct search queries.",
         response_model=QueryClassification
    )

    def get_prompt_fully_qualified(query):
            prompt = f"""
            As a helpful agent, you'll recieve requests from users looking for furniture products.

            Your task is to search with a structured query against a furniture product catalog.

            Here is the users request:

            {query}

            Return the best classifications for this user's query.

            Try to pick as diverse a set of possible to ensure the customer finds what they need.
            (IE different top level categories are better than very similar classifications that share most of their tree / subtree)

            Return an empty list if no classification fits, or its too ambiguous.

            """

            return prompt

    def fully_classified(query):
        prompt = get_prompt_fully_qualified(query)
        classification = enricher.enrich(prompt).model_copy()
        if "No Classification Fits" in classification.classifications:
            classification.classifications = []
        return classification

    fully_classified("dinooosaur"), fully_classified("sofa loveseat")
    return enricher, fully_classified, get_prompt_fully_qualified


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Redefine ground truth""")
    return


@app.cell
def _():
    from cheat_at_search.wands_data import labeled_query_products, queries

    def get_top_categories(column, no_fit_label, cutoff=0.8):
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

    # The original ground truth by category
    ground_truth_cat = get_top_categories('category', 'No Category Fits', cutoff=0.8)


    ground_truth_cat_list = get_top_categories('category', 'No Category Fits', cutoff=0.05)

    # Group by query, collect category into list
    ground_truth_cat_list = ground_truth_cat_list.groupby('query').agg({'category': list}).reset_index()
    # Remove empty items on "No Category Fits" from lists
    ground_truth_cat_list['category'] = ground_truth_cat_list['category'].apply(lambda x: [y.strip() for y in x if y and y != 'No Category Fits'])
    ground_truth_cat_list
    return ground_truth_cat, ground_truth_cat_list


@app.cell
def _(fully_classified, ground_truth_cat_list):
    def get_preds(cat, column):
        if column == 'category':
            return cat.categories
        elif column == 'sub_category':
            return cat.sub_categories
        else:
            raise ValueError(f'Unknown column {column}')

    def jaccard_sim(ground_truth, column, no_fit_label, classifier_fn):
        jaccard_sum = 0
        num_preds = 0
        for _, row in ground_truth.iterrows():
            query = row['query']
            expected_category = set(row[column])
            cat = classifier_fn(query)
            preds = set(get_preds(cat, column))
            if len(preds) == 0 or preds == [no_fit_label]:
                print(f'Skipping {query}')
                continue
            jaccard = len(preds.intersection(expected_category)) / len(preds.union(expected_category))
            print(f'{query} -- pred:{preds} == expected:{expected_category}')
            print(f'Jaccard: {jaccard}')
            jaccard_sum = jaccard_sum + jaccard
            num_preds = num_preds + 1
        return (jaccard_sum / num_preds, num_preds / len(ground_truth))
    jaccard, coverage = jaccard_sim(ground_truth_cat_list, 'category', 'No Category Fits', fully_classified)
    (jaccard, coverage)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Maybe a better question -- do we recall the most important prediction?""")
    return


@app.cell
def _(fully_classified, ground_truth_cat):
    def recall_when_pred(categorized, ground_truth_cat):
        """When we retrieve a category, is it correct?"""
        _hits = []
        _misses = []
        idx = 0
        for _, row in ground_truth_cat.iterrows():
            query = row['query']
            expected_category = row['category']
            cat = categorized(query)
            if len(cat.classifications) == 0:
                print(f'{idx} Skipping {query}')
                continue
            if expected_category != 'No Category Fits':
                if expected_category.strip() in cat.categories:
                    print(f'{idx} q:{query} -- pred:{cat.categories} == expected:{expected_category.strip()}')
                    _hits.append((expected_category, cat))
                else:
                    print('***')
                    print(f'{query} -- pred:{cat.categories} != expected:{expected_category.strip()}')
                    print(cat.classifications)
                    _misses.append((expected_category, cat))
                    num_so_far = len(_hits) + len(_misses)
                    print(f'{idx} recall (N={num_so_far}) -- {len(_hits) / (len(_hits) + len(_misses))}')
            idx = idx + 1
        return (len(_hits) / (len(_hits) + len(_misses)), _hits, _misses)
    prec, _hits, _misses = recall_when_pred(fully_classified, ground_truth_cat)
    prec
    return


@app.cell
def _(fully_classified, ground_truth_cat):
    def recall_all(categorized, ground_truth):
        """When we retrieve a category, is it correct?"""
        _hits = []
        _misses = []
        idx = 0
        for _, row in ground_truth.sample(frac=1).iterrows():
            query = row['query']
            expected_category = row['category']
            cat = categorized(query)
            if len(cat.classifications) == 0:
                print(f'{idx} Skipping {query}')
                continue
            if expected_category == 'No Category Fits' and len(cat.categories) > 0:
                print('!**')
                print(f'{query} -- pred:{cat.categories} != expected:{expected_category.strip()}')
                print(cat.classifications)
                _misses.append((expected_category, cat))
                num_so_far = len(_hits) + len(_misses)
                print(f'{idx} recall (N={num_so_far}) -- {len(_hits) / (len(_hits) + len(_misses))}')
            if expected_category.strip() in cat.categories:
                print(f'{idx} q:{query} -- pred:{cat.categories} == expected:{expected_category.strip()}')
                _hits.append((expected_category, cat))
            else:
                print('***')
                print(f'{query} -- pred:{cat.categories} != expected:{expected_category.strip()}')
                print(cat.classifications)
                _misses.append((expected_category, cat))
                num_so_far = len(_hits) + len(_misses)
                print(f'{idx} recall (N={num_so_far}) -- {len(_hits) / (len(_hits) + len(_misses))}')
            idx = idx + 1
        return (len(_hits) / (len(_hits) + len(_misses)), _hits, _misses)
    recall, _hits, _misses = recall_all(fully_classified, ground_truth_cat)
    recall
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Run Category search strategy with classifier""")
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
            for sub_category in structured.sub_categories:
                tokenized_subcategory = snowball_tokenizer(sub_category)
                subcategory_match = np.zeros(len(self.index))
                if tokenized_subcategory:
                    subcategory_match = self.index['subcategory_snowball'].array.score(tokenized_subcategory) > 0
                bm25_scores[subcategory_match] = bm25_scores[subcategory_match] + self.sub_category_boost
            for category in structured.categories:
                tokenized_category = snowball_tokenizer(category)
                category_match = np.zeros(len(self.index))
                if tokenized_category:
                    category_match = self.index['category_snowball'].array.score(tokenized_category) > 0
                bm25_scores[category_match] = bm25_scores[category_match] + self.category_boost
            top_k = np.argsort(-bm25_scores)[:k]
            scores = bm25_scores[top_k]
            return (top_k, scores)  # ****  # Baseline BM25 search from before  # ****  # If there's a subcategory, boost that by a constant amount  # ****  # If there's a category, boost that by a constant amount
    return CategorySearch, SearchArray, SearchStrategy, np, snowball_tokenizer


@app.cell
def _(CategorySearch, fully_classified, products, run_strategy):
    _categorized_search = CategorySearch(products, fully_classified)
    graded_categorized = run_strategy(_categorized_search)
    graded_categorized
    return (graded_categorized,)


@app.cell
def _(graded_bm25, graded_categorized, ndcgs):
    ndcgs(graded_bm25).mean(), ndcgs(graded_categorized).mean()
    return


@app.cell
def _(graded_bm25, graded_categorized, ndcg_delta):
    deltas = ndcg_delta(graded_categorized, graded_bm25)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Look at a query""")
    return


@app.cell
def _(fully_classified):
    QUERY = "sugar canister"
    fully_classified(QUERY)
    return (QUERY,)


@app.cell
def _(QUERY, ground_truth_cat):
    ground_truth_cat[ground_truth_cat['query'] == QUERY]
    return


@app.cell
def _(QUERY, graded_categorized):
    graded_categorized[graded_categorized['query'] == QUERY][['product_name', 'category hierarchy', 'grade']]
    return


@app.cell
def _(QUERY, graded_bm25):
    graded_bm25[graded_bm25['query'] == QUERY][['product_name', 'category hierarchy', 'grade']]
    return


@app.cell
def _(SearchArray, SearchStrategy, np, snowball_tokenizer):
    class CategorySearch_1(SearchStrategy):

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
            num_tokens_matched = np.zeros(len(self.index))
            for token in tokenized:
                name_score = self.index['product_name_snowball'].array.score(token) * self.name_boost
                desc_score = self.index['product_description_snowball'].array.score(token) * self.description_boost
                bm25_scores = bm25_scores + (name_score + desc_score)
                num_tokens_matched[name_score + desc_score > 0] = num_tokens_matched[name_score + desc_score > 0] + 1
            for sub_category in structured.sub_categories:
                tokenized_subcategory = snowball_tokenizer(sub_category)
                subcategory_match = np.zeros(len(self.index))
                if tokenized_subcategory:
                    subcategory_match = self.index['subcategory_snowball'].array.score(tokenized_subcategory) > 0
                bm25_scores[subcategory_match] = bm25_scores[subcategory_match] + self.sub_category_boost
            for category in structured.categories:
                tokenized_category = snowball_tokenizer(category)
                category_match = np.zeros(len(self.index))
                if tokenized_category:
                    category_match = self.index['category_snowball'].array.score(tokenized_category) > 0
                bm25_scores[category_match] = bm25_scores[category_match] + self.category_boost
            min_token_match = max(len(tokenized) / 2, 1)
            bm25_scores[num_tokens_matched < min_token_match] = bm25_scores[num_tokens_matched < min_token_match] * 0.8
            top_k = np.argsort(-bm25_scores)[:k]
            scores = bm25_scores[top_k]
            return (top_k, scores)  # ****  # Baseline BM25 search from before  # ****  # If there's a subcategory, boost that by a constant amount  # ****  # If there's a category, boost that by a constant amount  # ***  # Require all tokens to match
    return (CategorySearch_1,)


@app.cell
def _(CategorySearch_1, fully_classified, products, run_strategy):
    _categorized_search = CategorySearch_1(products, fully_classified)
    graded_categorized2 = run_strategy(_categorized_search)
    graded_categorized2
    return (graded_categorized2,)


@app.cell
def _(graded_bm25, graded_categorized2, ndcgs):
    ndcgs(graded_bm25).mean(), ndcgs(graded_categorized2).mean()
    return


@app.cell
def _(CategorySearch_1, fully_classified, products, run_strategy):
    _categorized_search = CategorySearch_1(products, fully_classified, category_boost=100, sub_category_boost=50)
    graded_categorized_1 = run_strategy(_categorized_search)
    graded_categorized_1
    return


@app.cell
def _(enricher, get_prompt_fully_qualified):
    def get_num_tokens(query):
        prompt = get_prompt_fully_qualified(query)
        return enricher.get_num_tokens(prompt)

    get_num_tokens("loveseat sofa")
    return (get_num_tokens,)


@app.cell
def _(get_num_tokens):
    get_num_tokens("loveseat")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
