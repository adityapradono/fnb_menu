"""
OCR Error Corrections Dictionary and Processing Rules
Comprehensive list of common OCR corrections, skip patterns, and prompt templates for menu OCR processing
"""

# SKIP PATTERNS

# Phone numbers and contact info
CONTACT_PATTERNS = [
    r'^\d{4}-\d{4}-\d{4}$',           # phone: 0812-9060-2734
    r'^\+\d{2,3}-\d+',                # international phone
]

# Social media and handles
SOCIAL_PATTERNS = [
    r'^[a-z][a-z0-9]*[_.][a-z0-9_.]+$',  # handle with dot/underscore
]

# Restaurant notices
NOTICE_PATTERNS = [
    r'^NO\s',                              # NO PORK, NO MSG
    r'NO\s+(PORK|LARD|MIRIN|MSG|ALCOHOL)', # specific notices
    r'.*NO\s+LARD.*NO\s+MIRIN.*',          # combined notices
]

# Noise and artifacts
NOISE_PATTERNS = [
    r'^\*',                                # footnote markers
    r'^[.…]+$',                            # dots only
    r'^["\']+[.\s]*["\']*$',               # quotes only
    r'^[.…"\']+$',                         # mixed noise
]

# Timestamps and dates
TIMESTAMP_PATTERNS = [
    r'^\d{4}/\d{1,2}/\d{1,2}',            # date: 2025/7/16
    r'^\d{4}-\d{1,2}-\d{1,2}',            # date: 2025-07-16
    r'\d{1,2}:\d{2}$',                    # time: 12:32
]

# Combine all skip patterns
SKIP_PATTERNS = (
    CONTACT_PATTERNS +
    SOCIAL_PATTERNS +
    NOTICE_PATTERNS +
    NOISE_PATTERNS +
    TIMESTAMP_PATTERNS
)


# DESCRIPTION CONNECTORS

# Indonesian connector words — descriptions start with these
INDONESIAN_CONNECTORS = {
    'dan', 'dengan', 'atau', 'di', 'ke',
    'yang', 'lalu', 'serta', 'dari',
    'untuk', 'ini', 'itu'
}

# English connector words
ENGLISH_CONNECTORS = {
    'and', 'with', 'or', 'of', 'in',
    'a', 'an', 'the', 'topped', 'served',
    'made', 'mixed', 'garnished',
}

# Combine all description connectors
DESCRIPTION_CONNECTORS = INDONESIAN_CONNECTORS | ENGLISH_CONNECTORS


# OCR CHARACTER CORRECTIONS

# Common character swap corrections
CHARACTER_SWAP_CORRECTIONS = {
    # Numbers and characters confusion
    'lce': 'Ice',
    'lced': 'Iced',
    'Mlx': 'Mix',
    'Mllk': 'Milk',
    'C0ffee': 'Coffee',
    'C0ke': 'Coke',
    'Juice0': 'Juiceo',

    # Common doubled characters
    'IICE': 'ICE',
    'Teriyakii': 'Teriyaki',
    'Chickenn': 'Chicken',
    'Coffeee': 'Coffee',
}

# Protein name corrections
PROTEIN_CORRECTIONS = {
    'Chiken': 'Chicken',
    'Chlcken': 'Chicken',
    'Chlken': 'Chicken',
    'ChicKen': 'Chicken',
    'Chickεn': 'Chicken',
    'Beεf': 'Beef',
    'Bεef': 'Beef',
    'Flsh': 'Fish',
    'Fīsh': 'Fish',
    'Shrmp': 'Shrimp',
    'Shrlmp': 'Shrimp',
}

# Cooking method corrections
COOKING_METHOD_CORRECTIONS = {
    'Grllled': 'Grilled',
    'Frīed': 'Fried',
    'Bakеd': 'Baked',
    'Steamеd': 'Steamed',
    'Roastеd': 'Roasted',
    'Crspy': 'Crispy',
    'Crіspy': 'Crispy',
}

# Beverage corrections
BEVERAGE_CORRECTIONS = {
    'Beverge': 'Beverage',
    'Beveragε': 'Beverage',
    'Coffce': 'Coffee',
    'Coffe': 'Coffee',
    'Tεa': 'Tea',
    'Juіce': 'Juice',
    'Smoothle': 'Smoothie',
    'Mllkshake': 'Milkshake',
}

# Food term corrections
FOOD_TERM_CORRECTIONS = {
    'Salac': 'Salad',
    'Saiad': 'Salad',
    'Satad': 'Salad',
    'S0up': 'Soup',
    'Rіce': 'Rice',
    'Ricε': 'Rice',
    'Noodlε': 'Noodle',
    'Nood1e': 'Noodle',
    'Saucε': 'Sauce',
    'Saυce': 'Sauce',
}

# Restaurant specific corrections (add per restaurant)
RESTAURANT_SPECIFIC_CORRECTIONS = {
    # Restaurant poi_id: 
    'Cuny': 'Curry',
    'Gydon': 'Gyudon',
}


# COMBINE ALL CORRECTIONS

def get_all_corrections() -> dict:
    """Combine all correction dictionaries into one master dictionary."""
    all_corrections = {}

    correction_categories = [
        CHARACTER_SWAP_CORRECTIONS,
        PROTEIN_CORRECTIONS,
        COOKING_METHOD_CORRECTIONS,
        BEVERAGE_CORRECTIONS,
        FOOD_TERM_CORRECTIONS,
        RESTAURANT_SPECIFIC_CORRECTIONS,
    ]

    for category in correction_categories:
        all_corrections.update(category)

    return all_corrections

# Master corrections dictionary
OCR_CORRECTIONS = get_all_corrections()


# LLM PROMPT TEMPLATE

LLM_PROMPT_TEMPLATE = """
You are an expert Data Auditor specialized in OCR post-processing for food and beverages menus.
Review the following JSON data and apply these cleaning rules strictly:

   TASKS:
   1. LOGICAL PRICE SCALING:
      - Identify outliers where a price is missing digits (e.g., '50' instead of '50000').
      - Scale it to match the currency magnitude of the surrounding items in the same category.
      - All amounts must be integers.

   2. OCR CHARACTER CORRECTION:
      - Fix common OCR character swaps ONLY:
      * 'l' or 'I' misread as each other (e.g., 'lce' -> 'Ice')
      * '0' misread as 'O' or vice versa
      * Common doubled characters in NAMES caused by OCR
         (e.g., 'Teriyakii' -> 'Teriyaki', 'IICE' -> 'ICE', 'Chickenn' -> 'Chicken')
      * Only fix doubled characters if the result is a real recognizable word.
      - Fix clear spelling mistakes caused by OCR (e.g., 'Chiken' -> 'Chicken', 'Karaage' stays 'Karaage')
      - DO NOT translate any word to English or any other language.
      - DO NOT change local language words (e.g., 'Abokadojusu', 'Gyudon', 'Shoyu', 'Miruku', 'Ocha' must stay as-is).
      - DO NOT change brand names or restaurant-specific terms.

   3. NAME NORMALIZATION:
      - Apply Title Case consistently (e.g., 'spicy katsu curry' -> 'Spicy Katsu Curry', 'Chicken KatsuDon' -> 'Chicken Katsu Don').
      - Remove OCR noise from names: leading/trailing dots, slashes, timestamps.
      - DO NOT reorder words in a name.

   4. DEDUPLICATION:
      - If multiple items have EXACTLY the same name AND price → keep only 1.
      - If multiple items have very similar names (minor spacing/capitalization differences) 
      AND same price → keep only the best formatted one.
      - If items have the same or similar name but DIFFERENT prices → keep all.
      - Never merge items with different prices, even if names are identical.
   
   5. REMOVE ITEM
      - If name doesn't make sense (e.g., number in name 1 -> 'name': '1')

   5. STRICT RULES — NEVER DO:
      - Never translate local language words to English.
      - Never add fields that don't exist in the original.
      - Never remove items unless they are exact duplicates (same name + same price).

   Return ONLY the cleaned JSON. No explanation, no markdown, no extra text.

RAW JSON:
{raw_json}
"""


# UTILITY FUNCTIONS

def add_custom_correction(error: str, correction: str):
    """Add a custom OCR correction at runtime."""
    OCR_CORRECTIONS[error.strip()] = correction.strip()

def add_restaurant_corrections(corrections_dict: dict):
    """Add restaurant-specific corrections in bulk."""
    for error, correction in corrections_dict.items():
        add_custom_correction(error, correction)

def get_correction_stats() -> dict:
    """Get statistics about loaded corrections and patterns."""
    return {
        'total_corrections':      len(OCR_CORRECTIONS),
        'character_swaps':        len(CHARACTER_SWAP_CORRECTIONS),
        'protein_corrections':    len(PROTEIN_CORRECTIONS),
        'cooking_corrections':    len(COOKING_METHOD_CORRECTIONS),
        'beverage_corrections':   len(BEVERAGE_CORRECTIONS),
        'food_term_corrections':  len(FOOD_TERM_CORRECTIONS),
        'restaurant_specific':    len(RESTAURANT_SPECIFIC_CORRECTIONS),
        'skip_patterns':          len(SKIP_PATTERNS),
        'description_connectors': len(DESCRIPTION_CONNECTORS),
    }

def search_corrections(query: str) -> dict:
    """Search for corrections containing a specific term."""
    query = query.lower()
    return {k: v for k, v in OCR_CORRECTIONS.items() if query in k.lower() or query in v.lower()}

# EXPORT

__all__ = [
    # Skip patterns
    'SKIP_PATTERNS',

    # Description connectors
    'DESCRIPTION_CONNECTORS',

    # OCR corrections
    'OCR_CORRECTIONS',

    # LLM
    'LLM_PROMPT_TEMPLATE',

    # Utilities
    'add_custom_correction',
    'add_restaurant_corrections',
    'get_correction_stats',
    'search_corrections',
    'get_all_corrections',
]

# STATS ON IMPORT

# Print statistics when module is imported
if __name__ == '__main__':
    stats = get_correction_stats()
    print("Corrections statistics:")
    for category, count in stats.items():
        print(f"   • {category.replace('_', ' ').title()}: {count}")

    # Example usage
    print("\nExample corrections:")
    examples = ['Chiken', 'lce', 'IICE', 'Teriyakii']
    for example in examples:
        correction = OCR_CORRECTIONS.get(example, 'Not found')
        print(f"   • '{example}' → '{correction}'")