"""Structured food reference data.

This is deliberately hand-curated, not model-generated. Storage advice, typical
shelf life and preservation methods are the kind of thing a language model will
happily invent, so FreshCO AI keeps them as data and lets the LLM *synthesise
from* these records rather than author them (see app/ai/assistant.py).

Shelf-life figures are typical domestic ranges for the storage mode named, drawn
from general consumer food-storage guidance. They are starting estimates for a
visually-assessed item, not regulatory limits, and they are always surfaced to
the user as an estimated window.

`healthy_hue` is the OpenCV input that matters: the hue band (OpenCV HSV, so
0-179) a sound example of this food occupies. The discoloration metric measures
drift away from it. `browning_penalty` scales how much visible browning should
count against the score -- high for a cut apple or leafy green where browning
means deterioration, low for a banana where it largely means ripening.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ShelfLife:
    """Typical days of usable life, by storage mode."""

    counter: tuple[int, int] | None = None
    pantry: tuple[int, int] | None = None
    refrigerated: tuple[int, int] | None = None
    frozen: tuple[int, int] | None = None

    def for_storage(self, storage: str) -> tuple[int, int] | None:
        return getattr(self, storage, None)


@dataclass(frozen=True)
class FoodRecord:
    name: str
    category: str
    shelf_life: ShelfLife
    preferred_storage: str
    storage_headline: str
    storage_details: list[str]
    preservation: list[str]
    nutrition: str
    recipe_uses: list[str]
    #: OpenCV HSV hue band (0-179) of a sound example. None for foods with no
    #: meaningful single hue (bread, milk), which skips the discoloration term.
    healthy_hue: tuple[int, int] | None = None
    #: 0..1 multiplier on the browning term of the freshness formula.
    browning_penalty: float = 1.0
    #: Foods where softening is the dominant spoilage cue.
    texture_sensitive: bool = True
    #: Extra caution shown for higher-risk categories.
    high_risk: bool = False
    aliases: list[str] = field(default_factory=list)


# --- Fruit ----------------------------------------------------------------

_FRUIT: list[FoodRecord] = [
    FoodRecord(
        name="Apple",
        category="fruit",
        shelf_life=ShelfLife(counter=(5, 7), refrigerated=(21, 42)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate in the crisper drawer",
        storage_details=[
            "Keep apples in the fridge crisper to slow softening.",
            "Store away from leafy greens, as apples release ethylene.",
        ],
        preservation=["Slice and freeze for baking", "Cook down into apple sauce"],
        nutrition="A source of dietary fibre and vitamin C.",
        recipe_uses=["Salad", "Apple sauce", "Baked apple", "Smoothie"],
        healthy_hue=(0, 20),
        browning_penalty=1.0,
    ),
    FoodRecord(
        name="Banana",
        category="fruit",
        shelf_life=ShelfLife(counter=(2, 5), refrigerated=(5, 8), frozen=(60, 90)),
        preferred_storage="counter",
        storage_headline="Keep on the counter, out of direct sunlight",
        storage_details=[
            "Store at room temperature until the skin reaches the ripeness you want.",
            "Separating the bunch slows ripening slightly.",
            "Refrigeration darkens the skin but keeps the flesh usable for longer.",
        ],
        preservation=["Peel and freeze for smoothies", "Mash into banana bread batter"],
        nutrition="A source of potassium, vitamin B6 and quick carbohydrate.",
        recipe_uses=["Smoothie", "Banana bread", "Pancakes", "Oatmeal topping"],
        healthy_hue=(20, 35),
        # Browning skin on a banana is mostly ripening, not spoilage.
        browning_penalty=0.45,
    ),
    FoodRecord(
        name="Orange",
        category="fruit",
        shelf_life=ShelfLife(counter=(7, 10), refrigerated=(21, 30)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate loose, not sealed in plastic",
        storage_details=[
            "Keep in the fridge in an open bag or loose in the drawer.",
            "Trapped moisture on the peel encourages mould.",
        ],
        preservation=["Juice and freeze", "Zest and dry the peel"],
        nutrition="High in vitamin C, with fibre in the segments.",
        recipe_uses=["Juice", "Salad", "Marmalade", "Dressing"],
        healthy_hue=(5, 25),
        browning_penalty=1.0,
    ),
    FoodRecord(
        name="Mango",
        category="fruit",
        shelf_life=ShelfLife(counter=(2, 5), refrigerated=(5, 8), frozen=(150, 180)),
        preferred_storage="counter",
        storage_headline="Ripen on the counter, then refrigerate",
        storage_details=[
            "Leave at room temperature until it yields slightly to a gentle press.",
            "Once ripe, refrigeration buys a few more days.",
        ],
        preservation=["Cube and freeze", "Cook into chutney"],
        nutrition="A source of vitamin A and vitamin C.",
        recipe_uses=["Smoothie", "Salsa", "Chutney", "Lassi"],
        healthy_hue=(10, 32),
        browning_penalty=0.7,
    ),
    FoodRecord(
        name="Guava",
        category="fruit",
        shelf_life=ShelfLife(counter=(2, 4), refrigerated=(5, 8)),
        preferred_storage="counter",
        storage_headline="Ripen on the counter, then chill",
        storage_details=[
            "Keep at room temperature until fragrant and slightly soft.",
            "Refrigerate once ripe to slow further softening.",
        ],
        preservation=["Cook into jam", "Blend and freeze as pulp"],
        nutrition="Notably high in vitamin C and dietary fibre.",
        recipe_uses=["Juice", "Jam", "Fruit salad"],
        healthy_hue=(25, 45),
        browning_penalty=0.8,
    ),
    FoodRecord(
        name="Pear",
        category="fruit",
        shelf_life=ShelfLife(counter=(3, 5), refrigerated=(10, 14)),
        preferred_storage="refrigerated",
        storage_headline="Chill once it gives at the neck",
        storage_details=[
            "Pears ripen from the inside out; check the neck, not the belly.",
            "Refrigerate as soon as the neck yields to slow it down.",
        ],
        preservation=["Poach and refrigerate", "Slice and freeze"],
        nutrition="A source of dietary fibre and vitamin C.",
        recipe_uses=["Salad", "Poached pear", "Crumble"],
        healthy_hue=(25, 45),
        browning_penalty=1.0,
    ),
    FoodRecord(
        name="Strawberry",
        category="fruit",
        shelf_life=ShelfLife(refrigerated=(3, 7), frozen=(180, 240)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate dry and unwashed",
        storage_details=[
            "Wash only just before eating; surface moisture accelerates mould.",
            "Line the container with paper and leave the lid slightly open.",
            "Remove any single mouldy berry immediately.",
        ],
        preservation=["Freeze whole on a tray", "Cook into a quick compote"],
        nutrition="High in vitamin C, with manganese and fibre.",
        recipe_uses=["Smoothie", "Compote", "Salad", "Dessert topping"],
        healthy_hue=(0, 12),
        browning_penalty=1.2,
    ),
    FoodRecord(
        name="Avocado",
        category="fruit",
        shelf_life=ShelfLife(counter=(2, 4), refrigerated=(4, 7)),
        preferred_storage="counter",
        storage_headline="Ripen on the counter, refrigerate when soft",
        storage_details=[
            "Leave at room temperature until it yields to gentle pressure.",
            "Refrigerate once ripe to hold it for a few more days.",
            "Cut surfaces brown quickly; press cling film against the flesh.",
        ],
        preservation=["Mash with lime juice and freeze", "Make guacamole"],
        nutrition="High in monounsaturated fat, potassium and folate.",
        recipe_uses=["Toast", "Guacamole", "Salad", "Smoothie"],
        healthy_hue=(35, 75),
        browning_penalty=1.3,
    ),
    FoodRecord(
        name="Grapes",
        category="fruit",
        shelf_life=ShelfLife(refrigerated=(7, 14), frozen=(90, 120)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate unwashed on the stem",
        storage_details=[
            "Keep on the stem in a perforated bag in the fridge.",
            "Wash just before serving.",
        ],
        preservation=["Freeze whole as a snack", "Roast for sauces"],
        nutrition="A source of vitamin K and antioxidants.",
        recipe_uses=["Snack", "Salad", "Roasted with chicken"],
        healthy_hue=(35, 90),
        browning_penalty=1.0,
    ),
    FoodRecord(
        name="Pineapple",
        category="fruit",
        shelf_life=ShelfLife(counter=(2, 3), refrigerated=(5, 7), frozen=(150, 180)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate once cut",
        storage_details=[
            "Whole pineapple keeps a couple of days at room temperature.",
            "Once cut, store the flesh in a sealed container in the fridge.",
        ],
        preservation=["Freeze chunks", "Grill and refrigerate"],
        nutrition="High in vitamin C and manganese.",
        recipe_uses=["Juice", "Salsa", "Grilled pineapple", "Curry"],
        healthy_hue=(18, 38),
        browning_penalty=1.0,
    ),
    FoodRecord(
        name="Lemon",
        category="fruit",
        shelf_life=ShelfLife(counter=(7, 10), refrigerated=(21, 30)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate in a sealed bag",
        storage_details=[
            "A sealed bag in the fridge keeps lemons juicy far longer than the counter.",
        ],
        preservation=["Freeze juice in ice cube trays", "Preserve in salt"],
        nutrition="High in vitamin C.",
        recipe_uses=["Dressing", "Marinade", "Lemonade", "Baking"],
        healthy_hue=(20, 35),
        browning_penalty=1.0,
    ),
    FoodRecord(
        name="Pomegranate",
        category="fruit",
        shelf_life=ShelfLife(counter=(5, 7), refrigerated=(21, 60)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate whole",
        storage_details=[
            "Whole fruit keeps for weeks in the fridge.",
            "Loose seeds keep about 3 days refrigerated.",
        ],
        preservation=["Freeze the seeds", "Reduce juice into molasses"],
        nutrition="A source of vitamin C, vitamin K and polyphenols.",
        recipe_uses=["Salad", "Juice", "Raita", "Dessert topping"],
        healthy_hue=(0, 12),
        browning_penalty=1.0,
    ),
]

# --- Vegetables -----------------------------------------------------------

_VEGETABLE: list[FoodRecord] = [
    FoodRecord(
        name="Tomato",
        category="vegetable",
        shelf_life=ShelfLife(counter=(3, 5), refrigerated=(7, 10)),
        preferred_storage="counter",
        storage_headline="Keep at room temperature, stem side down",
        storage_details=[
            "Room temperature preserves flavour and texture better than the fridge.",
            "Keep out of direct sunlight and do not stack them.",
            "Refrigerate only once fully ripe, to buy a few extra days.",
        ],
        preservation=["Roast and freeze", "Cook into passata", "Sun-dry"],
        nutrition="A source of vitamin C, potassium and lycopene.",
        recipe_uses=["Salad", "Tomato rice", "Soup", "Sandwich", "Curry base"],
        healthy_hue=(0, 12),
        browning_penalty=1.0,
    ),
    FoodRecord(
        name="Carrot",
        category="vegetable",
        shelf_life=ShelfLife(refrigerated=(21, 35), frozen=(240, 300)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate in a sealed bag",
        storage_details=[
            "Remove the leafy tops, which draw moisture out of the root.",
            "A sealed bag or container in the crisper keeps them crisp for weeks.",
        ],
        preservation=["Blanch and freeze", "Pickle in vinegar"],
        nutrition="High in beta-carotene (vitamin A) and fibre.",
        recipe_uses=["Salad", "Soup", "Stir fry", "Carrot halwa"],
        healthy_hue=(5, 22),
        browning_penalty=1.0,
    ),
    FoodRecord(
        name="Potato",
        category="vegetable",
        shelf_life=ShelfLife(pantry=(30, 60), refrigerated=(60, 90)),
        preferred_storage="pantry",
        storage_headline="Store cool, dark and dry — not in the fridge",
        storage_details=[
            "A dark, ventilated cupboard is ideal; light turns the skin green.",
            "Do not refrigerate raw potatoes: cold converts starch to sugar.",
            "Keep away from onions, which speed up sprouting.",
        ],
        preservation=["Cook then freeze", "Par-boil and chill"],
        nutrition="A source of potassium, vitamin C and starch.",
        recipe_uses=["Curry", "Roast potatoes", "Mash", "Soup"],
        healthy_hue=(15, 35),
        browning_penalty=0.9,
        texture_sensitive=True,
    ),
    FoodRecord(
        name="Onion",
        category="vegetable",
        shelf_life=ShelfLife(pantry=(30, 60), refrigerated=(14, 21)),
        preferred_storage="pantry",
        storage_headline="Store dry and ventilated, away from potatoes",
        storage_details=[
            "A mesh bag in a cool, dry cupboard is best.",
            "Refrigerate only once cut, sealed in a container.",
        ],
        preservation=["Dice and freeze", "Caramelise and refrigerate", "Pickle"],
        nutrition="A source of vitamin C and prebiotic fibre.",
        recipe_uses=["Curry base", "Salad", "Sambar", "Caramelised onions"],
        healthy_hue=(10, 30),
        browning_penalty=0.8,
    ),
    FoodRecord(
        name="Spinach",
        category="vegetable",
        shelf_life=ShelfLife(refrigerated=(3, 7), frozen=(240, 300)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate dry, loosely wrapped",
        storage_details=[
            "Keep leaves dry: moisture is what turns spinach slimy.",
            "Wrap loosely in a paper towel inside a container.",
            "Wash only just before cooking.",
        ],
        preservation=["Blanch and freeze", "Cook into palak and refrigerate"],
        nutrition="High in vitamin K, vitamin A, folate and iron.",
        recipe_uses=["Palak paneer", "Dal", "Smoothie", "Sautéed greens", "Soup"],
        healthy_hue=(35, 85),
        # Yellowing and browning are the primary spoilage cue for leafy greens.
        browning_penalty=1.4,
    ),
    FoodRecord(
        name="Broccoli",
        category="vegetable",
        shelf_life=ShelfLife(refrigerated=(7, 14), frozen=(240, 300)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate unwashed in an open bag",
        storage_details=[
            "Broccoli needs airflow; a loosely open bag in the crisper works well.",
            "Yellowing florets mean it is past its best.",
        ],
        preservation=["Blanch and freeze", "Roast and refrigerate"],
        nutrition="High in vitamin C, vitamin K and fibre.",
        recipe_uses=["Stir fry", "Soup", "Roasted broccoli", "Pasta"],
        healthy_hue=(35, 85),
        browning_penalty=1.3,
    ),
    FoodRecord(
        name="Capsicum",
        category="vegetable",
        shelf_life=ShelfLife(refrigerated=(7, 14), frozen=(180, 240)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate whole and dry",
        storage_details=[
            "Store whole in the crisper drawer; cut peppers soften quickly.",
        ],
        preservation=["Slice and freeze", "Roast and store in oil"],
        nutrition="High in vitamin C and vitamin A.",
        recipe_uses=["Stir fry", "Salad", "Stuffed peppers", "Curry"],
        healthy_hue=(0, 90),
        browning_penalty=1.0,
        aliases=["Bell pepper", "Capsicum"],
    ),
    FoodRecord(
        name="Brinjal",
        category="vegetable",
        shelf_life=ShelfLife(counter=(2, 3), refrigerated=(5, 7)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate, but use within the week",
        storage_details=[
            "Brinjal bruises easily; store it where nothing presses on it.",
            "Very cold temperatures cause pitting, so avoid the coldest shelf.",
        ],
        preservation=["Roast and freeze the pulp", "Pickle"],
        nutrition="A source of dietary fibre and manganese.",
        recipe_uses=["Baingan bharta", "Curry", "Grilled aubergine", "Moussaka"],
        healthy_hue=(120, 160),
        browning_penalty=1.1,
        aliases=["Eggplant", "Aubergine"],
    ),
    FoodRecord(
        name="Cucumber",
        category="vegetable",
        shelf_life=ShelfLife(refrigerated=(5, 10)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate wrapped, away from the coldest shelf",
        storage_details=[
            "Wrap in a paper towel to absorb condensation.",
            "Keep away from tomatoes and bananas, which speed up softening.",
        ],
        preservation=["Quick-pickle in vinegar", "Blend into a chilled soup"],
        nutrition="Mostly water, with some vitamin K.",
        recipe_uses=["Salad", "Raita", "Sandwich", "Pickle"],
        healthy_hue=(35, 85),
        browning_penalty=1.2,
    ),
    FoodRecord(
        name="Cabbage",
        category="vegetable",
        shelf_life=ShelfLife(refrigerated=(21, 60)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate whole and unwashed",
        storage_details=[
            "A whole head keeps for weeks; peel off outer leaves as needed.",
            "Once cut, wrap the exposed face tightly.",
        ],
        preservation=["Ferment into sauerkraut", "Blanch and freeze"],
        nutrition="A source of vitamin C, vitamin K and fibre.",
        recipe_uses=["Slaw", "Stir fry", "Thoran", "Soup"],
        healthy_hue=(35, 90),
        browning_penalty=1.2,
    ),
    FoodRecord(
        name="Cauliflower",
        category="vegetable",
        shelf_life=ShelfLife(refrigerated=(7, 14), frozen=(240, 300)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate stem down in an open bag",
        storage_details=[
            "Storing stem-side down keeps moisture off the florets.",
            "Brown speckling on the curd means it is deteriorating.",
        ],
        preservation=["Blanch and freeze", "Roast and refrigerate"],
        nutrition="A source of vitamin C, vitamin K and fibre.",
        recipe_uses=["Roasted cauliflower", "Curry", "Cauliflower rice", "Soup"],
        healthy_hue=(15, 40),
        browning_penalty=1.4,
    ),
    FoodRecord(
        name="Mushroom",
        category="vegetable",
        shelf_life=ShelfLife(refrigerated=(4, 7)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate in paper, never sealed plastic",
        storage_details=[
            "A paper bag lets mushrooms breathe; plastic makes them slimy.",
            "Do not wash until you are ready to cook.",
        ],
        preservation=["Sauté and freeze", "Dry thoroughly"],
        nutrition="A source of B vitamins, selenium and vitamin D.",
        recipe_uses=["Stir fry", "Risotto", "Soup", "Pasta"],
        healthy_hue=(10, 30),
        browning_penalty=1.2,
    ),
]

# --- Protein --------------------------------------------------------------

_PROTEIN: list[FoodRecord] = [
    FoodRecord(
        name="Chicken",
        category="poultry",
        shelf_life=ShelfLife(refrigerated=(1, 2), frozen=(270, 365)),
        preferred_storage="refrigerated",
        storage_headline="Keep refrigerated below 4 °C and cook within 2 days",
        storage_details=[
            "Store on the lowest shelf in a sealed container so nothing drips onto other food.",
            "Freeze immediately if you will not cook it within 2 days.",
            "Never refreeze poultry that has already been thawed in the fridge and left out.",
        ],
        preservation=["Freeze in portions", "Cook fully, then refrigerate up to 3 days"],
        nutrition="High in protein, with B vitamins and phosphorus.",
        recipe_uses=["Curry", "Grilled chicken", "Soup", "Stir fry"],
        healthy_hue=(0, 20),
        browning_penalty=1.5,
        high_risk=True,
    ),
    FoodRecord(
        name="Fish",
        category="seafood",
        shelf_life=ShelfLife(refrigerated=(1, 2), frozen=(180, 240)),
        preferred_storage="refrigerated",
        storage_headline="Keep on ice and cook the same day if possible",
        storage_details=[
            "Fresh fish is best cooked the day you buy it.",
            "Store on ice or on the coldest shelf, sealed.",
            "Smell is a far better guide than appearance for fish.",
        ],
        preservation=["Freeze immediately in portions", "Cure or smoke"],
        nutrition="High in protein and, in oily fish, omega-3 fatty acids.",
        recipe_uses=["Fish curry", "Grilled fish", "Fish fry", "Soup"],
        healthy_hue=(0, 25),
        browning_penalty=1.5,
        high_risk=True,
    ),
    FoodRecord(
        name="Beef",
        category="meat",
        shelf_life=ShelfLife(refrigerated=(2, 4), frozen=(180, 365)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate sealed on the lowest shelf",
        storage_details=[
            "Keep sealed and separate from ready-to-eat food.",
            "Surface darkening alone is often oxidation, not spoilage — but smell and texture matter more.",
        ],
        preservation=["Freeze in portions", "Cook fully, then refrigerate"],
        nutrition="High in protein, iron, zinc and vitamin B12.",
        recipe_uses=["Curry", "Stew", "Steak", "Mince dishes"],
        healthy_hue=(0, 15),
        browning_penalty=1.3,
        high_risk=True,
    ),
]

# --- Dairy ----------------------------------------------------------------

_DAIRY: list[FoodRecord] = [
    FoodRecord(
        name="Milk",
        category="dairy",
        shelf_life=ShelfLife(refrigerated=(5, 7), frozen=(60, 90)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate at the back, not in the door",
        storage_details=[
            "The fridge door is the warmest spot; the back shelf is the coldest.",
            "Always follow the printed date on the carton over any visual assessment.",
        ],
        preservation=["Freeze in a rigid container", "Make paneer or yoghurt"],
        nutrition="A source of calcium, protein and vitamin B12.",
        recipe_uses=["Curd", "Paneer", "Kheer", "Coffee"],
        healthy_hue=None,
        browning_penalty=0.6,
        texture_sensitive=False,
        high_risk=True,
    ),
    FoodRecord(
        name="Cheese",
        category="dairy",
        shelf_life=ShelfLife(refrigerated=(14, 28), frozen=(120, 180)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate wrapped in paper, then loose plastic",
        storage_details=[
            "Hard cheese wants to breathe: paper first, then a loose bag.",
            "On hard cheese, a small surface mould spot can be cut away with a margin; "
            "on soft cheese, discard the whole piece.",
        ],
        preservation=["Freeze grated", "Store hard cheese in wax paper"],
        nutrition="A source of calcium, protein and fat.",
        recipe_uses=["Sandwich", "Pasta", "Toast", "Salad"],
        healthy_hue=(15, 35),
        browning_penalty=0.8,
        high_risk=True,
    ),
    FoodRecord(
        name="Yogurt",
        category="dairy",
        shelf_life=ShelfLife(refrigerated=(7, 14)),
        preferred_storage="refrigerated",
        storage_headline="Keep refrigerated and sealed",
        storage_details=[
            "Some clear whey on top is normal — stir it back in.",
            "Follow the printed date; pink or fuzzy growth means discard it.",
        ],
        preservation=["Strain into hung curd", "Freeze into a frozen yoghurt"],
        nutrition="A source of calcium, protein and live cultures.",
        recipe_uses=["Raita", "Lassi", "Marinade", "Breakfast bowl"],
        healthy_hue=None,
        browning_penalty=0.6,
        texture_sensitive=False,
        high_risk=True,
        aliases=["Curd", "Dahi", "Thayir", "Perugu", "Mosaru"],
    ),
]

# --- Bakery ---------------------------------------------------------------

_BAKERY: list[FoodRecord] = [
    FoodRecord(
        name="Bread",
        category="bakery",
        shelf_life=ShelfLife(pantry=(2, 4), refrigerated=(7, 14), frozen=(60, 90)),
        preferred_storage="pantry",
        storage_headline="Keep in a sealed bag at room temperature",
        storage_details=[
            "Room temperature in a sealed bag keeps the crumb soft.",
            "Refrigeration makes bread stale faster, though it delays mould.",
            "Freezing sliced bread is the best way to keep it long term.",
            "Visible mould means discard the whole loaf, not just the slice.",
        ],
        preservation=["Freeze sliced", "Blitz stale bread into crumbs", "Make croutons"],
        nutrition="A source of carbohydrate, with fibre in wholegrain loaves.",
        recipe_uses=["Toast", "Sandwich", "Breadcrumbs", "Croutons", "Bread upma"],
        healthy_hue=(10, 30),
        browning_penalty=0.7,
        texture_sensitive=False,
    ),
    FoodRecord(
        name="Pastry",
        category="bakery",
        shelf_life=ShelfLife(pantry=(1, 2), refrigerated=(3, 5), frozen=(30, 60)),
        preferred_storage="pantry",
        storage_headline="Eat fresh; refrigerate anything with a cream filling",
        storage_details=[
            "Plain pastry keeps a day or two in a tin at room temperature.",
            "Cream- or custard-filled pastry must be refrigerated.",
        ],
        preservation=["Freeze unfilled pastry", "Refresh in a hot oven"],
        nutrition="Typically high in refined carbohydrate and fat.",
        recipe_uses=["Dessert", "Breakfast", "Bread pudding"],
        healthy_hue=(10, 30),
        browning_penalty=0.7,
        texture_sensitive=False,
    ),
]

# --- Regional foods -------------------------------------------------------
#
# Produce and staples common in Indian kitchens, which is who FreshCO AI's six
# languages serve. Each name and alias is the English or transliterated name a
# person or the vision model is likely to use. All are four characters or more,
# because ``KnowledgeService.resolve`` falls back to substring matching, where a
# short alias such as "pea" would silently claim "pear".
#
# Same sourcing rule as everything above: general consumer storage guidance,
# stated as typical domestic ranges, conservative where sources differ.

_REGIONAL_FRUIT: list[FoodRecord] = [
    FoodRecord(
        name="Papaya",
        category="fruit",
        shelf_life=ShelfLife(counter=(2, 4), refrigerated=(4, 7), frozen=(150, 180)),
        preferred_storage="counter",
        storage_headline="Ripen on the counter, then refrigerate",
        storage_details=[
            "Leave at room temperature until the skin is mostly yellow.",
            "Once ripe, refrigerate whole and eat within a few days.",
            "Cover cut papaya and refrigerate; use within two days.",
        ],
        preservation=["Cube and freeze for smoothies", "Use raw papaya in curry"],
        nutrition="A source of vitamin C, vitamin A and folate.",
        recipe_uses=["Fruit salad", "Smoothie", "Raw papaya thoran", "Papaya curry"],
        healthy_hue=(5, 45),
        browning_penalty=0.8,
        aliases=["Pappali", "Papita", "Boppayi"],
    ),
    FoodRecord(
        name="Jackfruit",
        category="fruit",
        shelf_life=ShelfLife(counter=(2, 4), refrigerated=(3, 5), frozen=(30, 60)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate the pods once cut",
        storage_details=[
            "A whole ripe jackfruit keeps a few days at room temperature.",
            "Store separated pods in an airtight container in the fridge.",
        ],
        preservation=["Freeze deseeded pods", "Cook raw jackfruit as a vegetable"],
        nutrition="A source of carbohydrate, fibre, vitamin C and potassium.",
        recipe_uses=["Chakka varatti", "Jackfruit curry", "Halwa", "Payasam"],
        healthy_hue=(20, 50),
        browning_penalty=0.8,
        aliases=["Kathal", "Chakka", "Palapazham", "Halasina"],
    ),
    FoodRecord(
        name="Sapota",
        category="fruit",
        shelf_life=ShelfLife(counter=(2, 4), refrigerated=(5, 7)),
        preferred_storage="counter",
        storage_headline="Ripen on the counter, then refrigerate",
        storage_details=[
            "Ripe sapota gives slightly to a gentle press.",
            "Refrigerate ripe fruit to slow it down for a few more days.",
        ],
        preservation=["Blend into a milkshake and freeze"],
        nutrition="A source of carbohydrate and dietary fibre.",
        recipe_uses=["Milkshake", "Fruit salad", "Ice cream"],
        healthy_hue=(5, 20),
        # The skin is brown when ripe, so browning means little here.
        browning_penalty=0.4,
        aliases=["Chikoo", "Chiku", "Sapodilla"],
    ),
    FoodRecord(
        name="Watermelon",
        category="fruit",
        shelf_life=ShelfLife(counter=(7, 10), refrigerated=(14, 21)),
        preferred_storage="counter",
        storage_headline="Keep whole at room temperature; refrigerate once cut",
        storage_details=[
            "A whole watermelon keeps about a week at room temperature.",
            "Once cut, cover the surface and refrigerate; use within 3-5 days.",
        ],
        preservation=["Freeze cubes for drinks", "Juice and freeze"],
        nutrition="Mostly water, with some vitamin C and vitamin A.",
        recipe_uses=["Juice", "Fruit salad", "Granita"],
        healthy_hue=(35, 85),
        aliases=["Tarbooz", "Tarbuj", "Kalingad"],
    ),
    FoodRecord(
        name="Muskmelon",
        category="fruit",
        shelf_life=ShelfLife(counter=(2, 4), refrigerated=(5, 7)),
        preferred_storage="counter",
        storage_headline="Ripen on the counter, then refrigerate",
        storage_details=[
            "A ripe melon smells sweet at the stem end.",
            "Cover cut melon and refrigerate; use within 3 days.",
        ],
        preservation=["Freeze cubes for smoothies"],
        nutrition="A source of vitamin A and vitamin C.",
        recipe_uses=["Juice", "Fruit salad", "Milkshake"],
        healthy_hue=(10, 35),
        # "Melon" resolves here exactly, so it never reaches the substring
        # stage, where it would match "Bitter melon" or "Watermelon".
        aliases=["Melon", "Cantaloupe", "Kharbuja", "Kharbooja"],
    ),
    FoodRecord(
        name="Custard apple",
        category="fruit",
        shelf_life=ShelfLife(counter=(2, 3), refrigerated=(2, 4), frozen=(60, 90)),
        preferred_storage="counter",
        storage_headline="Ripen on the counter and eat soon after",
        storage_details=[
            "Ripe fruit is soft and the segments begin to separate.",
            "Refrigerate ripe fruit only briefly; the skin blackens in the cold.",
        ],
        preservation=["Freeze the deseeded pulp"],
        nutrition="A source of carbohydrate, fibre and vitamin C.",
        recipe_uses=["Milkshake", "Basundi", "Ice cream"],
        healthy_hue=(30, 60),
        browning_penalty=0.8,
        aliases=["Sitaphal", "Sugar apple", "Seethapalam", "Seethaphal"],
    ),
    FoodRecord(
        name="Amla",
        category="fruit",
        shelf_life=ShelfLife(counter=(3, 5), refrigerated=(7, 14), frozen=(180, 240)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate in a dry container",
        storage_details=["Keep dry; trapped moisture brings soft spots and mould."],
        preservation=["Pickle", "Make murabba", "Dry slices in the sun"],
        nutrition="Very high in vitamin C.",
        recipe_uses=["Pickle", "Murabba", "Chutney", "Juice"],
        healthy_hue=(25, 45),
        aliases=["Indian gooseberry", "Nellikai", "Nellikka", "Usirikaya", "Amalaki"],
    ),
    FoodRecord(
        name="Jamun",
        category="fruit",
        shelf_life=ShelfLife(counter=(1, 2), refrigerated=(2, 4)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate and eat within a few days",
        storage_details=[
            "Jamun bruises and ferments quickly; keep it cold and unwashed.",
            "Wash just before eating.",
        ],
        preservation=["Cook into jam", "Juice and freeze"],
        nutrition="A source of vitamin C and fibre.",
        recipe_uses=["Eaten fresh with salt", "Jam", "Juice"],
        healthy_hue=(120, 165),
        aliases=["Java plum", "Indian blackberry", "Naval pazham", "Neredu"],
    ),
    FoodRecord(
        name="Coconut",
        category="fruit",
        shelf_life=ShelfLife(counter=(14, 30), refrigerated=(4, 7), frozen=(60, 90)),
        preferred_storage="counter",
        storage_headline="Keep whole at room temperature; refrigerate once opened",
        storage_details=[
            "A whole mature coconut keeps for weeks in a cool, dry place.",
            "Grated or opened coconut goes rancid quickly: refrigerate it and "
            "use within a few days, or freeze it.",
            "Discard flesh that smells sour or shows pink or grey patches.",
        ],
        preservation=["Freeze grated coconut in portions", "Toast and store dry"],
        nutrition="High in fat, with some fibre.",
        recipe_uses=["Chutney", "Avial", "Thoran", "Coconut rice"],
        # Brown husk outside, white flesh inside: no single hue describes it.
        healthy_hue=None,
        texture_sensitive=False,
        aliases=["Thengai", "Nariyal", "Thenga", "Kobbari"],
    ),
]

_REGIONAL_VEGETABLE: list[FoodRecord] = [
    FoodRecord(
        name="Okra",
        category="vegetable",
        shelf_life=ShelfLife(refrigerated=(2, 4), frozen=(180, 240)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate dry and unwashed",
        storage_details=[
            "Moisture makes okra slimy and spotted; wash only before cooking.",
            "Store in a paper bag in the crisper and use within a few days.",
        ],
        preservation=["Blanch and freeze", "Pickle"],
        nutrition="A source of fibre, vitamin C, vitamin K and folate.",
        recipe_uses=["Bhindi fry", "Vendakkai poriyal", "Sambar", "Okra curry"],
        healthy_hue=(35, 75),
        browning_penalty=1.1,
        aliases=["Lady finger", "Ladies finger", "Bhindi", "Vendakkai", "Bendakaya"],
    ),
    FoodRecord(
        name="Bitter gourd",
        category="vegetable",
        shelf_life=ShelfLife(refrigerated=(4, 7), frozen=(180, 240)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate and use while green",
        storage_details=[
            "Store in the crisper, away from ripening fruit.",
            "It turns yellow-orange as it ripens and becomes more bitter.",
        ],
        preservation=["Slice, salt and sun-dry", "Blanch and freeze"],
        nutrition="A source of vitamin C and folate.",
        recipe_uses=["Pavakkai fry", "Karela sabzi", "Stuffed karela", "Pitlai"],
        healthy_hue=(35, 75),
        aliases=["Bitter melon", "Karela", "Pavakkai", "Kakarakaya", "Hagalakai"],
    ),
    FoodRecord(
        name="Bottle gourd",
        category="vegetable",
        shelf_life=ShelfLife(counter=(3, 5), refrigerated=(5, 10)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate whole",
        storage_details=[
            "Choose firm, unblemished gourds; soft spots spread quickly.",
            "Wrap cut gourd and use within two days.",
        ],
        preservation=["Cook into halwa and freeze"],
        nutrition="Mostly water, with some fibre.",
        recipe_uses=["Lauki dal", "Kootu", "Halwa", "Kofta"],
        healthy_hue=(30, 60),
        aliases=["Lauki", "Calabash", "Sorakkai", "Dudhi", "Sorakaya"],
    ),
    FoodRecord(
        name="Ridge gourd",
        category="vegetable",
        shelf_life=ShelfLife(refrigerated=(3, 5)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate and use within a few days",
        storage_details=["It softens and turns spongy as it ages."],
        preservation=["Use the peel in chutney"],
        nutrition="Mostly water, with some fibre.",
        recipe_uses=["Kootu", "Peerkangai chutney", "Dal", "Stir fry"],
        healthy_hue=(35, 75),
        aliases=["Turai", "Peerkangai", "Beerakaya", "Luffa"],
    ),
    FoodRecord(
        name="Snake gourd",
        category="vegetable",
        shelf_life=ShelfLife(refrigerated=(3, 5)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate and use within a few days",
        storage_details=["Wrap loosely; it turns limp and yellow as it ages."],
        preservation=["Blanch and freeze"],
        nutrition="Mostly water, with some fibre.",
        recipe_uses=["Kootu", "Poriyal", "Raita"],
        healthy_hue=(35, 75),
        aliases=["Pudalangai", "Potlakaya", "Chichinda", "Padavalanga"],
    ),
    FoodRecord(
        name="Ivy gourd",
        category="vegetable",
        shelf_life=ShelfLife(refrigerated=(4, 7)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate unwashed",
        storage_details=["Use while green; red flesh inside means it is overripe."],
        preservation=["Pickle"],
        nutrition="A source of fibre.",
        recipe_uses=["Kovakkai fry", "Tindora sabzi", "Pickle"],
        healthy_hue=(35, 75),
        aliases=["Tindora", "Kovakkai", "Dondakaya", "Tendli", "Kundru"],
    ),
    FoodRecord(
        # Named with the qualifier on purpose. With a bare "Drumstick", whether
        # "Chicken drumstick" resolved to Chicken or to this vegetable -- and so
        # whether raw poultry was capped -- would depend only on list order.
        name="Drumstick (moringa)",
        category="vegetable",
        shelf_life=ShelfLife(refrigerated=(4, 7), frozen=(60, 90)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate wrapped",
        storage_details=[
            "Wrap in paper and refrigerate; pods dry out and turn woody.",
            "Cut into pieces before freezing.",
        ],
        preservation=["Cut and freeze for sambar"],
        nutrition="A source of vitamin C and fibre.",
        recipe_uses=["Sambar", "Drumstick curry", "Avial"],
        healthy_hue=(35, 75),
        texture_sensitive=False,
        aliases=["Moringa", "Moringa pods", "Murungakkai", "Sahjan", "Munagakaya"],
    ),
    FoodRecord(
        name="Raw banana",
        category="vegetable",
        shelf_life=ShelfLife(counter=(3, 6)),
        preferred_storage="counter",
        storage_headline="Keep at room temperature, not in the fridge",
        storage_details=[
            "Refrigeration blackens the skin.",
            "It ripens and sweetens if left too long.",
        ],
        preservation=["Make chips", "Cook and freeze"],
        nutrition="A source of resistant starch and potassium.",
        recipe_uses=["Vazhakkai fry", "Chips", "Avial", "Kofta"],
        healthy_hue=(30, 60),
        browning_penalty=0.8,
        aliases=["Plantain", "Green banana", "Vazhakkai", "Kacha kela"],
    ),
    FoodRecord(
        name="Curry leaves",
        category="vegetable",
        shelf_life=ShelfLife(refrigerated=(7, 14), frozen=(60, 90)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate dry in an airtight container",
        storage_details=[
            "Strip from the stem, pat dry and store in a lined airtight box.",
            "Black spots and a limp texture mean they are past their best.",
        ],
        preservation=["Freeze on the stem", "Dry and powder"],
        nutrition="Used as an aromatic and eaten in small amounts.",
        recipe_uses=["Tempering", "Curry leaf chutney", "Podi"],
        healthy_hue=(35, 85),
        aliases=["Kariveppilai", "Kadi patta", "Karivepaku", "Kari patta"],
    ),
    FoodRecord(
        name="Coriander",
        category="vegetable",
        shelf_life=ShelfLife(refrigerated=(5, 10), frozen=(60, 90)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate stems-down in water, or wrapped",
        storage_details=[
            "Stand the stems in a jar of water in the fridge, or wrap in a "
            "damp towel.",
            "Remove yellowing or slimy stems so they do not spread.",
        ],
        preservation=["Blend into chutney and freeze in portions"],
        nutrition="A source of vitamin K.",
        recipe_uses=["Garnish", "Chutney", "Coriander rice"],
        healthy_hue=(35, 85),
        aliases=["Cilantro", "Coriander leaves", "Kothamalli", "Dhania"],
    ),
    FoodRecord(
        name="Fenugreek leaves",
        category="vegetable",
        shelf_life=ShelfLife(refrigerated=(2, 4), frozen=(60, 90)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate wrapped and use quickly",
        storage_details=["Pick off the leaves and wrap them in paper."],
        preservation=["Dry into kasuri methi", "Blanch and freeze"],
        nutrition="A source of vitamin K and fibre.",
        recipe_uses=["Methi thepla", "Aloo methi", "Methi paratha"],
        healthy_hue=(35, 85),
        aliases=["Methi", "Methi leaves", "Vendhaya keerai", "Menthi kura"],
    ),
    FoodRecord(
        name="Green chilli",
        category="vegetable",
        shelf_life=ShelfLife(refrigerated=(7, 14), frozen=(90, 180)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate with the stems removed",
        storage_details=[
            "Removing the stems slows rotting.",
            "Store dry in an airtight container lined with paper.",
        ],
        preservation=["Freeze whole", "Pickle"],
        nutrition="A source of vitamin C.",
        recipe_uses=["Tempering", "Chutney", "Pickle"],
        healthy_hue=(35, 80),
        aliases=["Green chili", "Hari mirch", "Pachai milagai", "Pachimirchi"],
    ),
    FoodRecord(
        name="French beans",
        category="vegetable",
        shelf_life=ShelfLife(refrigerated=(5, 7), frozen=(180, 240)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate unwashed in a bag",
        storage_details=["Beans that bend instead of snapping are past their best."],
        preservation=["Blanch and freeze"],
        nutrition="A source of fibre, vitamin K and vitamin C.",
        recipe_uses=["Beans poriyal", "Pulao", "Stir fry"],
        healthy_hue=(35, 75),
        aliases=["Green beans", "String beans"],
    ),
    FoodRecord(
        name="Cluster beans",
        category="vegetable",
        shelf_life=ShelfLife(refrigerated=(3, 5)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate unwashed",
        storage_details=["Use while firm and green."],
        preservation=["Sun-dry salted beans"],
        nutrition="A source of fibre.",
        recipe_uses=["Kothavarangai poriyal", "Guar ki sabzi", "Paruppu usili"],
        healthy_hue=(35, 75),
        aliases=["Guar phali", "Gawar", "Kothavarangai", "Goru chikkudu"],
    ),
    FoodRecord(
        name="Radish",
        category="vegetable",
        shelf_life=ShelfLife(refrigerated=(10, 14)),
        preferred_storage="refrigerated",
        storage_headline="Remove the leaves and refrigerate",
        storage_details=[
            "Leaves draw moisture from the root; cut them off and cook them "
            "separately.",
        ],
        preservation=["Pickle"],
        nutrition="Mostly water, with some vitamin C.",
        recipe_uses=["Mullangi sambar", "Mooli paratha", "Salad"],
        # White radish has no meaningful hue.
        healthy_hue=None,
        aliases=["Mooli", "Mullangi", "White radish", "Daikon"],
    ),
    FoodRecord(
        name="Beetroot",
        category="vegetable",
        shelf_life=ShelfLife(refrigerated=(14, 21)),
        preferred_storage="refrigerated",
        storage_headline="Trim the leaves and refrigerate",
        storage_details=["Leave an inch of stem so the root does not bleed."],
        preservation=["Pickle", "Roast and freeze"],
        nutrition="A source of folate and fibre.",
        recipe_uses=["Beetroot poriyal", "Halwa", "Salad", "Cutlet"],
        healthy_hue=(140, 179),
        aliases=["Beet", "Chukandar"],
    ),
    FoodRecord(
        name="Sweet potato",
        category="vegetable",
        shelf_life=ShelfLife(pantry=(14, 28)),
        preferred_storage="pantry",
        storage_headline="Keep in a cool, dark, dry place",
        storage_details=["Do not refrigerate raw sweet potato; cold hardens the core."],
        preservation=["Bake and freeze the flesh"],
        nutrition="A source of vitamin A and fibre.",
        recipe_uses=["Roasted", "Chaat", "Kheer"],
        healthy_hue=(0, 25),
        aliases=["Shakarkandi", "Sakkaravalli kizhangu", "Genasu"],
    ),
    FoodRecord(
        name="Elephant foot yam",
        category="vegetable",
        shelf_life=ShelfLife(pantry=(14, 30)),
        preferred_storage="pantry",
        storage_headline="Keep whole in a cool, dry place",
        storage_details=[
            "Wrap cut yam and refrigerate; use within a few days.",
            "Always cook it thoroughly -- raw yam irritates the mouth and throat.",
        ],
        preservation=["Par-boil and freeze"],
        nutrition="A source of carbohydrate, fibre and potassium.",
        recipe_uses=["Senai roast", "Kootu", "Suran sabzi"],
        healthy_hue=None,
        texture_sensitive=False,
        aliases=["Suran", "Jimikand", "Senai kizhangu", "Kanda gadda"],
    ),
    FoodRecord(
        name="Taro root",
        category="vegetable",
        shelf_life=ShelfLife(pantry=(7, 14)),
        preferred_storage="pantry",
        storage_headline="Keep in a cool, dry place, not the fridge",
        storage_details=["Always cook it -- raw taro irritates the mouth and throat."],
        preservation=["Boil, peel and freeze"],
        nutrition="A source of carbohydrate and fibre.",
        recipe_uses=["Arbi fry", "Seppankizhangu roast", "Curry"],
        healthy_hue=None,
        texture_sensitive=False,
        aliases=["Taro", "Arbi", "Colocasia", "Seppankizhangu", "Chamadumpa"],
    ),
    FoodRecord(
        name="Pumpkin",
        category="vegetable",
        shelf_life=ShelfLife(pantry=(30, 60), refrigerated=(5, 7)),
        preferred_storage="pantry",
        storage_headline="Keep whole in a cool, dry place",
        storage_details=[
            "A whole pumpkin keeps for weeks.",
            "Remove the seeds from cut pumpkin, wrap it and refrigerate.",
        ],
        preservation=["Cube and freeze", "Cook into halwa"],
        nutrition="A source of vitamin A.",
        recipe_uses=["Sambar", "Kaddu sabzi", "Erissery", "Halwa"],
        healthy_hue=(5, 60),
        aliases=["Kaddu", "Parangikai", "Red pumpkin", "Mathanga", "Gummadikaya"],
    ),
]

_REGIONAL_DAIRY: list[FoodRecord] = [
    FoodRecord(
        name="Paneer",
        category="dairy",
        shelf_life=ShelfLife(refrigerated=(2, 4), frozen=(60, 90)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate submerged in water",
        storage_details=[
            "Keep fresh paneer covered in water in the fridge and change the "
            "water daily.",
            "Packaged paneer keeps to its printed date until opened.",
            "A sour smell or slimy surface means discard it.",
        ],
        preservation=["Freeze in cubes"],
        nutrition="A source of protein, calcium and fat.",
        recipe_uses=["Paneer butter masala", "Palak paneer", "Paneer bhurji"],
        healthy_hue=None,
        high_risk=True,
        aliases=["Cottage cheese", "Indian cottage cheese"],
    ),
]

# Cooked staples. Idli and the flatbreads are scored as bakery, whose weights
# make surface mould the dominant signal -- which is what spoils them.
_PREPARED: list[FoodRecord] = [
    FoodRecord(
        name="Idli",
        category="bakery",
        shelf_life=ShelfLife(refrigerated=(2, 3), frozen=(30, 60)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate within two hours of cooking",
        storage_details=[
            "Cool, then refrigerate in an airtight box.",
            "Reheat by steaming until hot right through.",
            "A sour smell or slimy surface means discard them.",
        ],
        preservation=["Freeze and re-steam", "Turn leftovers into idli upma"],
        nutrition="A source of carbohydrate, with some protein from the lentils.",
        recipe_uses=["Idli upma", "Fried idli", "Idli manchurian"],
        healthy_hue=None,
        browning_penalty=0.5,
        texture_sensitive=False,
        # Moist cooked rice: Bacillus cereus can grow at room temperature
        # without changing how it looks.
        high_risk=True,
        aliases=["Idly", "Iddli"],
    ),
    FoodRecord(
        name="Dosa",
        category="bakery",
        shelf_life=ShelfLife(refrigerated=(1, 2)),
        preferred_storage="refrigerated",
        storage_headline="Eat fresh; refrigerate leftovers",
        storage_details=["Leftover dosa turns leathery; reheat on a hot tawa."],
        preservation=["Tear into dosa upma"],
        nutrition="A source of carbohydrate, with some protein from the lentils.",
        recipe_uses=["Dosa upma", "Masala dosa"],
        healthy_hue=(10, 30),
        browning_penalty=0.5,
        texture_sensitive=False,
        aliases=["Dosai", "Thosai"],
    ),
    FoodRecord(
        name="Chapati",
        category="bakery",
        shelf_life=ShelfLife(pantry=(1, 2), refrigerated=(3, 4), frozen=(30, 60)),
        preferred_storage="pantry",
        storage_headline="Keep wrapped in a cloth or closed box",
        storage_details=[
            "Wrap in a cloth inside a closed container to keep them soft.",
            "Refrigerate if keeping past a day; freeze with paper between them.",
        ],
        preservation=["Freeze stacked with paper", "Make kothu chapati"],
        nutrition="A source of carbohydrate and fibre from whole wheat.",
        recipe_uses=["Kothu chapati", "Wrap", "Chapati noodles"],
        healthy_hue=(10, 30),
        # Char spots are part of a good chapati, not decay.
        browning_penalty=0.3,
        texture_sensitive=False,
        aliases=["Roti", "Phulka", "Chappathi", "Chapathi"],
    ),
    FoodRecord(
        name="Paratha",
        category="bakery",
        shelf_life=ShelfLife(pantry=(1, 2), refrigerated=(3, 4), frozen=(30, 60)),
        preferred_storage="refrigerated",
        storage_headline="Refrigerate wrapped after the first day",
        storage_details=[
            "Stuffed parathas spoil faster than plain; refrigerate them the "
            "same day.",
        ],
        preservation=["Freeze half-cooked and finish on the tawa"],
        nutrition="A source of carbohydrate and fat.",
        recipe_uses=["Kothu parotta", "Wrap"],
        healthy_hue=(10, 30),
        browning_penalty=0.3,
        texture_sensitive=False,
        aliases=["Parotta", "Porotta", "Parantha"],
    ),
    FoodRecord(
        name="Cooked rice",
        category="other",
        shelf_life=ShelfLife(refrigerated=(1, 1), frozen=(14, 30)),
        preferred_storage="refrigerated",
        storage_headline="Cool within an hour, refrigerate, eat within a day",
        storage_details=[
            "Spread it out to cool quickly and refrigerate within an hour.",
            "Eat within 24 hours and reheat only once, until steaming hot.",
            "Never leave cooked rice out overnight, however it looks.",
        ],
        preservation=["Freeze in portions as soon as it cools"],
        nutrition="A source of carbohydrate.",
        recipe_uses=["Lemon rice", "Curd rice", "Fried rice", "Kheer"],
        healthy_hue=None,
        texture_sensitive=False,
        # Bacillus cereus spores survive cooking and grow at room temperature,
        # and the toxin they make is invisible and not destroyed by reheating.
        high_risk=True,
        aliases=["Steamed rice", "Boiled rice", "Plain rice", "Leftover rice"],
    ),
]

ALL_FOODS: list[FoodRecord] = (
    _FRUIT
    + _VEGETABLE
    + _PROTEIN
    + _DAIRY
    + _BAKERY
    + _REGIONAL_FRUIT
    + _REGIONAL_VEGETABLE
    + _REGIONAL_DAIRY
    + _PREPARED
)

#: Lookup by lower-cased name and alias.
FOOD_INDEX: dict[str, FoodRecord] = {}
for _record in ALL_FOODS:
    FOOD_INDEX[_record.name.lower()] = _record
    for _alias in _record.aliases:
        FOOD_INDEX.setdefault(_alias.lower(), _record)


#: Foods for which the browning index is a valid measurement.
#:
#: The browning index (Palou et al.) was developed for light-coloured food
#: surfaces, where browning is the dominant visible deterioration. Applied to a
#: saturated chromatic food it is meaningless as an absolute: a flawless red
#: tomato measures ~1.0 on the normalised scale, because a large positive a*
#: drives the formula regardless of condition. Comparing it across foods would
#: need a reference BI per food, which FreshCO AI does not have.
#:
#: So browning is measured only for the foods below. For everything else the
#: metric is returned as null, the UI hides the row, and the freshness formula
#: redistributes its weight to the signals that were measured. Deterioration in
#: those foods is still caught by defect coverage, discoloration (hue drift from
#: the curated healthy hue), texture uniformity and colour consistency.
BROWNING_APPLICABLE_FOODS: frozenset[str] = frozenset(
    {
        "apple",
        "pear",
        "banana",
        "avocado",
        "guava",
        "potato",
        "onion",
        "cauliflower",
        "mushroom",
        "bread",
        "pastry",
        "cheese",
    }
)


def browning_applicable(food_name: str | None) -> bool:
    """Whether the browning index means anything for this food."""
    if not food_name:
        return False
    return food_name.strip().lower() in BROWNING_APPLICABLE_FOODS


#: Pantry staples the recipe engine may assume a household has. Keeping this
#: explicit stops a generated recipe from quietly requiring a shopping trip.
PANTRY_STAPLES: list[str] = [
    "Salt",
    "Black pepper",
    "Cooking oil",
    "Olive oil",
    "Garlic",
    "Ginger",
    "Turmeric",
    "Cumin",
    "Chilli powder",
    "Mustard seeds",
    "Rice",
    "Wheat flour",
    "Sugar",
    "Lentils",
]
