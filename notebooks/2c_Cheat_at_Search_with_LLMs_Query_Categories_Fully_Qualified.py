import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Query -> Full classification

    <small>
    (from <a href="http://maven.com/softwaredoug/cheat-at-search">Cheat at Search with LLMs</a> training course by Doug Turnbull.)
    </small>

    In this refinement on [past examples](https://colab.research.google.com/drive/1AfK3uGV3Lbrv5henj995YV4XEpVmLpBf) we will get away from classifying to category / subcategory independently and classify to the full classification, ie a query to one of the following


    ```
    Furniture / Living Room Furniture / Chairs & Seating / Accent Chairs,
    Rugs / Area Rugs,
    ...
    ```

    We will recompute the previous stats to see how well this works.
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
    from cheat_at_search.search import run_strategy, graded_bm25, ndcgs, ndcg_delta, vs_ideal
    from cheat_at_search.wands_data import products

    products
    return graded_bm25, ndcg_delta, ndcgs, products, run_strategy


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Query -> Full classification

    We'll first setup the model of query -> the full classification.

    Here we provide a long list of valid full classifications for the LLM to use.

    **Warning** this gets expensive. It'll cost $5 or so to run. We'll quickly switch to some ways of saving costs, so no worries
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
    # Ameer says to try:
    # 'Furniture / Bedroom Furniture / Beds & Headboards / Beds',
    #-> ' Beds / Beds & Headboards / Bedroom Furniture / Furniture'
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
        classification: FullyQualifiedClassifications = Field(description="A classification for the product. Use 'No Classification Fits' if not a clear best classification.")

        @property
        def category(self):
            if self.classification == 'No Classification Fits':
                return 'No Category Fits'
            return self.classification.split(' / ')[0]

        @property
        def sub_category(self):
            if self.classification == 'No Classification Fits':
                return 'No SubCategory Fits'
            if len(self.classification.split(' / ')) < 2:
                return 'No SubCategory Fits'
            return self.classification.split(' / ')[1]
    return AutoEnricher, QueryClassification


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Query classification code

    This code is quite similar to previous iterations, **change** -- we're returning the classification now. Then we use this to parse out the category (top level) and sub category (second level)
    """
    )
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

            Return the 'Classification': the best classification in the schema for this user's query.
            Return 'No Classification Fits' if no classification fits or if its ambiguous
            """

            return prompt

    def fully_classified(query):
        prompt = get_prompt_fully_qualified(query)
        return enricher.enrich(prompt)

    fully_classified("dinosaur"), fully_classified("sofa loveseat")
    return (fully_classified,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Redefine ground truth

    This is the same ground truth from previous notebooks.

    We map the query to relevant products, then look to see the dominant category is for that query
    """
    )
    return


@app.cell
def _(fully_classified):
    from cheat_at_search.wands_data import labeled_query_products, queries

    def get_top_category(column, no_fit_label, cutoff=0.8):
        top_products = labeled_query_products[labeled_query_products['grade'] == 2]  # Get relevant products per query
        categories_per_query_ideal = top_products.groupby('query')[column].value_counts().reset_index()
        top_cat_proportion = categories_per_query_ideal.groupby(['query', column]).sum() / categories_per_query_ideal.groupby('query').sum()
        top_cat_proportion = top_cat_proportion.drop(columns=column).reset_index()  # Aggregate top categories
        top_cat_proportion = top_cat_proportion[top_cat_proportion['count'] > cutoff]
        top_cat_proportion[column].fillna(no_fit_label, inplace=True)
        ground_truth_cat = top_cat_proportion  # Get as percentage of all categories for this query
        ground_truth_cat = ground_truth_cat.merge(queries, how='right', on='query')[['query', column, 'count']]
        ground_truth_cat[column].fillna(no_fit_label, inplace=True)
        return ground_truth_cat
      # Only look at cases where the category is > 0.8
    def get_pred(cat, column):
        if column == 'category':
            return cat.category
        elif column == 'sub_category':  # Give No Category Fits to all others without dominant category
            return cat.sub_category
        else:
            raise ValueError(f'Unknown column {column}')

    def prec_cat(ground_truth, column, no_fit_label, categorized):
        hits = []
        misses = []
        for _, row in ground_truth.sample(frac=1).iterrows():
            query = row['query']
            expected_category = row[column]
            cat = categorized(query)
            pred = get_pred(cat, column)
            if pred == no_fit_label:
                print(f'Skipping {query}')
                continue
            if pred == expected_category.strip():
                hits.append((expected_category, cat))
            else:
                print('***')
                print(f'{query} -- predicted:{cat.category} != expected:{expected_category.strip()}')
                misses.append((expected_category, cat))
                num_so_far = len(hits) + len(misses)
                print(f'prec (N={num_so_far}) -- {len(hits) / (len(hits) + len(misses))}')
                print(f'coverage {num_so_far / len(ground_truth)}')
        return (len(hits) / (len(hits) + len(misses)), num_so_far / len(ground_truth))
    ground_truth_cat = get_top_category('category', 'No Category Fits')
    ground_truth_sub_cat = get_top_category('sub_category', 'No SubCategory Fits')
    _prec, _coverage = prec_cat(ground_truth_cat, 'category', 'No Category Fits', fully_classified)
    (_prec, _coverage)
    return ground_truth_cat, ground_truth_sub_cat, prec_cat


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### On impacted queries

    We know from the "perfect classifier" these queries would most benefit from this change. Let's see how we perform on these queries.

    """
    )
    return


@app.cell
def _(fully_classified, ground_truth_cat, prec_cat):
    impacted_queries = ['drum picture', 'bathroom freestanding cabinet', 'outdoor lounge chair', 'wood rack wide', 'outdoor light fixtures', 'bathroom vanity knobs', 'door jewelry organizer', 'beds that have leds', 'non slip shower floor tile', 'turquoise chair', 'modern outdoor furniture', 'podium with locking cabinet', 'closet storage with zipper', 'barstool patio sets', 'ayesha curry kitchen', 'led 60', 'wisdom stone river 3-3/4', 'liberty hardware francisco', 'french molding', 'glass doors for bath', 'accent leather chair', 'dark gray dresser', 'wainscoting ideas', 'floating bed', 'dining table vinyl cloth', 'entrance table', 'storage dresser', 'almost heaven sauna', 'toddler couch fold out', 'outdoor welcome rug', 'wooden chair outdoor', 'emma headboard', 'outdoor privacy wall', 'driftwood mirror', 'white abstract', 'bedroom accessories', 'bathroom lighting', 'light and navy blue decorative pillow', 'gnome fairy garden', 'medium size chandelier', 'above toilet cabinet', 'odum velvet', 'ruckus chair', 'modern farmhouse lighting semi flush mount', 'teal chair', 'bedroom wall decor floral, multicolored with some teal (prints)', 'big basket for dirty cloths', 'milk cow chair', 'small wardrobe grey', 'glow in the dark silent wall clock', 'medium clips', 'desk for kids tjat ate 10 year old', 'industrial pipe dining  table', 'itchington butterfly', 'midcentury tv unit', 'gas detector', 'fleur de lis living candle wall sconce bronze', 'zodiac pillow', 'papasan chair frame only', 'bed side table']
    _prec, _coverage = prec_cat(ground_truth_cat[ground_truth_cat['query'].isin(impacted_queries)], 'category', 'No Category Fits', fully_classified)
    (_prec, _coverage)
    return (impacted_queries,)


@app.cell
def _(fully_classified, ground_truth_cat, impacted_queries, prec_cat):
    _prec, _coverage = prec_cat(ground_truth_cat[~ground_truth_cat['query'].isin(impacted_queries)], 'category', 'No Category Fits', fully_classified)
    (_prec, _coverage)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Run Category search strategy with classifier

    Here we have an identical strategy as before to boost category / sub category

    Notice in the `search` method we go through the following flow:

    1. Classify query -> classification, using the passed function `query_to_cat`
    2. Perform a normal BM25 boost
    3. Boost category matches by category_boost
    4. Boost subcategory matches by subcategory boost

    **Idea to try** -- try REQUIRING a BM25 match before applying the category boost instead of just adding to the BM25 score -- which would include anything with BM25 score of 0.
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
                     category_boost=10,
                     sub_category_boost=5):
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
                subcategory_match = np.zeros(len(self.index))
                if tokenized_subcategory:
                    subcategory_match = self.index['subcategory_snowball'].array.score(tokenized_subcategory) > 0
                bm25_scores[subcategory_match] += self.sub_category_boost

            # ****
            # If there's a category, boost that by a constant amount
            if structured.category and structured.category != "No Category Fits":
                tokenized_category = snowball_tokenizer(structured.category)
                category_match = np.zeros(len(self.index))
                if tokenized_category:
                    category_match = self.index['category_snowball'].array.score(tokenized_category) > 0
                bm25_scores[category_match] += self.category_boost

            top_k = np.argsort(-bm25_scores)[:k]
            scores = bm25_scores[top_k]

            return top_k, scores
    return (CategorySearch,)


@app.cell
def _(CategorySearch, fully_classified, products, run_strategy):
    categorized_search = CategorySearch(products, fully_classified)
    graded_fully_classified = run_strategy(categorized_search)
    graded_fully_classified
    return (graded_fully_classified,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Analyze the results

    We notice
    * good NDCG change
    * limited downside impact to other queries
    """
    )
    return


@app.cell
def _(graded_bm25, graded_fully_classified, ndcgs):
    ndcgs(graded_bm25).mean(), ndcgs(graded_fully_classified).mean()
    return


@app.cell
def _(graded_bm25, graded_fully_classified, ndcg_delta):
    deltas = ndcg_delta(graded_fully_classified, graded_bm25)
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
    QUERY = "chair pillow cushion"
    fully_classified(QUERY)
    return (QUERY,)


@app.cell
def _(QUERY, ground_truth_cat):
    ground_truth_cat[ground_truth_cat['query'] == QUERY]
    return


@app.cell
def _(QUERY, ground_truth_sub_cat):
    ground_truth_sub_cat[ground_truth_sub_cat['query'] == QUERY]
    return


@app.cell
def _(QUERY, graded_fully_classified):
    graded_fully_classified[graded_fully_classified['query'] == QUERY][['product_name', 'category hierarchy', 'grade', 'score']]
    return


@app.cell
def _(QUERY, graded_bm25):
    graded_bm25[graded_bm25['query'] == QUERY][['product_name', 'category hierarchy', 'grade', 'score']]
    return


@app.cell
def _(products):
    products[products['product_name'] == 'gem paper clips , plastic , medium size , 500/box']
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
