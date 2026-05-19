import cv2
import gc
import glob
import json
import csv
import re
import os
import numpy as np
import google.genai as genai
from paddleocr import PaddleOCR
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
from typing import List, Dict, Tuple

# Import config
try:
    from ocr_config import (
        OCR_CONFIG, PREPROCESSING_CONFIG, PRICE_CONFIG,
        LAYOUT_CONFIG, DEDUPE_CONFIG,
        LLM_CONFIG, FILE_CONFIG, DEBUG_CONFIG,
        PERFORMANCE_CONFIG, VALIDATION_CONFIG, FOLDER_CONFIG,
    )
    print("✅ Configuration loaded successfully")
except ImportError as e:
    print(f"⚠️ config.py not found ({e}), using default settings")
    OCR_CONFIG           = {'use_angle_cls': True, 'lang': 'en', 'use_gpu': False,
                            'show_log': False, 'det_db_thresh': 0.20,
                            'det_db_box_thresh': 0.30, 'det_db_unclip_ratio': 2.0}
    PREPROCESSING_CONFIG = {'clahe_clip_limit': 2.0, 'clahe_tile_size': (8, 8),
                            'adaptive_block': 31, 'adaptive_c': 10}
    PRICE_CONFIG         = {'min_price': 1000, 'max_price': 5000000,
                            'price_suffix_multipliers': {'K': 1000, 'k': 1000, 'RB': 1000, 'JT': 1000000},
                            'variant_labels': {'R': 'Regular', 'S': 'Small', 'M': 'Medium',
                                               'L': 'Large', 'XL': 'Extra Large', 'O': 'Original'}}
    LAYOUT_CONFIG        = {'y_threshold': 40, 'x_gap': 400, 'max_name_words': 5}
    DEDUPE_CONFIG        = {'threshold': 0.85}
    LLM_CONFIG           = {'model': 'gemini-2.5-flash-lite', 'api_key': os.getenv('GEMINI_API_KEY', '')}
    FILE_CONFIG          = {'supported_extensions': ['.jpg', '.jpeg', '.png'],
                            'batch_size': 10}
    DEBUG_CONFIG         = {'enable_debug': False, 'verbose_output': True,
                            'suppress_paddleocr_logs': True}
    PERFORMANCE_CONFIG   = {'max_workers': 4, 'timeout_seconds': 30,
                            'garbage_collect_frequency': 10}
    VALIDATION_CONFIG    = {'min_items_per_image': 0, 'max_items_per_image': 100}
    FOLDER_CONFIG        = {'input_folder':  'input_images',
                            'output_folder': 'processed_images'}
    
# Import corrections
try:
    from ocr_corrections import (
        SKIP_PATTERNS, DESCRIPTION_CONNECTORS,
        OCR_CORRECTIONS, LLM_PROMPT_TEMPLATE,
        add_custom_correction, add_restaurant_corrections,
        get_correction_stats,
    )
    print(f"✅ Loaded {len(OCR_CORRECTIONS)} OCR corrections")
except ImportError as e:
    print(f"⚠️ corrections.py not found ({e}), using basic corrections")
    SKIP_PATTERNS          = [r'^\d{4}-\d{4}-\d{4}$', r'^NO\s', r'^[.…]+$']
    DESCRIPTION_CONNECTORS = {'dan', 'dengan', 'atau', 'di', 'ke', 'yang'}
    OCR_CORRECTIONS        = {'Chiken': 'Chicken', 'lce': 'Ice'}
    LLM_PROMPT_TEMPLATE    = "Clean this menu JSON:\n{raw_json}"
 
# Initialize PaddleoCR
try:
    engine = PaddleOCR(**{k: v for k, v in OCR_CONFIG.items()
                          if k in ('use_angle_cls', 'lang', 'use_gpu', 'show_log',
                                   'det_db_thresh', 'det_db_box_thresh', 'det_db_unclip_ratio')})
    print("✅ PaddleOCR initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize PaddleOCR: {e}")
    raise

# Initialize Gemini LLM
try:
    llm_client = genai.Client(api_key=LLM_CONFIG.get('api_key', ''))
    print("✅ Gemini LLM initialized successfully")
except Exception as e:
    print(f"⚠️ Gemini LLM not initialized: {e}")
    llm_client = None

# Folder configuration
INPUT_FOLDER     = FOLDER_CONFIG.get('input_folder',  'input_images')
PROCESSED_FOLDER = FOLDER_CONFIG.get('output_folder', 'processed_images')

def setup_folders():
    """Create necessary folders if they don't exist."""
    Path(INPUT_FOLDER).mkdir(exist_ok=True)
    Path(PROCESSED_FOLDER).mkdir(exist_ok=True)
    if DEBUG_CONFIG.get('verbose_output', True):
        print(f"Folders ready:")
        print(f"   Input:     {INPUT_FOLDER}/")
        print(f"   Processed: {PROCESSED_FOLDER}/")
 
 
def get_input_images() -> List[str]:
    """Get all supported image files from input folder."""
    supported = FILE_CONFIG.get('supported_extensions', ['.jpg', '.jpeg', '.png'])
    image_files = set()
    input_path  = Path(INPUT_FOLDER)
 
    if not input_path.exists():
        return []
 
    for ext in supported:
        image_files.update(input_path.glob(f"**/*{ext}"))
        image_files.update(input_path.glob(f"**/*{ext.upper()}"))
 
    def natural_sort_key(path):
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r'(\d+)', Path(path).name)]
 
    return sorted([str(f) for f in image_files], key=natural_sort_key)

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two strings (0-1)."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
 
def correct_common_ocr_errors(text: str) -> str:
    """Apply OCR corrections from corrections.py."""
    for error, correction in OCR_CORRECTIONS.items():
        if error in text:
            text = text.replace(error, correction)
    return text

# Token classification
def normalize_price_string(raw: str) -> str:
    """Fix OCR noise in price strings before parsing."""
    s = re.sub(r'^[.\s]+', '', raw)
 
    # Fix missing space between two price blocks
    s = re.sub(r'(\d+\s*[Kk])\s*([RLSMrlsm])\s*(\d)', r'\1(\2) \3', s)
 
    # Normalize KR/KL (no parens)
    s = re.sub(r'(\d+\s*[Kk])\s*([RLSMrlsm])\b', r'\1(\2)', s)
 
    # Fix missing opening paren
    s = re.sub(r'(\d+\s*[Kk])\s*([RLSMrlsm])\)', r'\1(\2)', s)
 
    # Fix missing closing paren at end
    s = re.sub(r'(\d+\s*[Kk]\s*\([RLSMrlsm])\s*$', r'\1)', s)
 
    # Fix missing closing paren followed by digit
    s = re.sub(r'(\d+\s*[Kk]\s*\([RLSMrlsm])(\d)', r'\1) \2', s)
 
    # Fix missing closing paren followed by space+digit
    s = re.sub(r'(\d+\s*[Kk]\s*\([RLSMrlsm])\s+(\d)', r'\1) \2', s)
 
    # Ensure space between price blocks
    s = re.sub(r'(\))\s*(\d)', r'\1 \2', s)
 
    # Add missing K before (R/L/S/M)
    s = re.sub(r'(\d+)\s*(\([RLSMrlsm]\))', r'\1K\2', s)
 
    # Fix OCR noise between pairs
    s = re.sub(r'(\d+\s*[Kk])\s*[Il|/\\]\s*(\d)', r'\1 \2', s)
 
    # Remove date/time suffixes
    s = re.sub(r'\s*\d{4}[/-]\d{1,2}[/-]\d{1,2}.*$', '', s)
    s = re.sub(r'\s*\d{1,2}:\d{2}.*$', '', s)
 
    # Strip leading O/S prefix (OCR noise)
    s = re.sub(r'^[OSos]\.*\s*(\d)', r'\1', s)
 
    # Fix switched K position
    s = re.sub(r'^([SMLRslmr])[.\s]*(\d+\s*[Kk])', r'\2(\1)', s)
 
    # Strip leading digit+dots
    s = re.sub(r'^\d[.\s]+(\d)', r'\1', s)
 
    return s.strip()

def parse_variant_prices(text: str) -> list:
    """
    Extract one or more price+variant pairs from a normalized price string.

    '38K(R) 45K(L)' → [{'amount': 38000, 'tag': 'R', 'variant': 'Regular'},
                        {'amount': 45000, 'tag': 'L', 'variant': 'Large'}]
    '5K'            → [{'amount': 5000,  'tag': None, 'variant': None}]
    """
    suffixes    = PRICE_CONFIG.get('price_suffix_multipliers',
                                   {'K': 1000, 'k': 1000, 'RB': 1000, 'JT': 1000000})
    variant_map = PRICE_CONFIG.get('variant_labels',
                                   {'R': 'Regular', 'L': 'Large', 'S': 'Small', 'M': 'Medium'})
    min_price   = PRICE_CONFIG.get('min_price', 1000)
    max_price   = PRICE_CONFIG.get('max_price', 5000000)

    suffix_pat = '|'.join(re.escape(s) for s in suffixes)

    block_re = re.compile(
        rf'(\d[\d.,]*)'                     # number        → group 1
        rf'\s*(?:({suffix_pat}))?'          # suffix K/RB   → group 2
        rf'\s*(?:\(([A-Za-z]{{1,5}})\))?'   # (TAG)         → group 3
    )

    results = []
    for m in block_re.finditer(text):
        raw_num, suffix, tag = m.group(1), m.group(2), m.group(3)

        # Skip if number is followed by words (description, not price)
        after = text[m.end():].strip()
        if re.match(r'^[a-zA-Z]{2,}', after) and not suffix and not tag:
            continue

        try:
            amount = float(raw_num.replace(',', '.'))
        except ValueError:
            continue

        if suffix:
            amount *= suffixes.get(suffix.upper(), 1)

        if not (min_price <= amount <= max_price):
            continue

        tag_upper = tag.upper() if tag else None
        results.append({
            'amount':  int(amount),
            'tag':     tag,
            'variant': variant_map.get(tag_upper) if tag_upper else None,
        })

    return results

def is_skip(token: str) -> bool:
    """Return True if token should be skipped (noise, phone, date, etc.)."""
    return any(re.search(p, token) for p in SKIP_PATTERNS)

def is_menu_item_name(token: str) -> bool:
    """
    Return True if token looks like a menu item name.
    Filters out descriptions, connector phrases, and noise.
    """
    words = token.strip().split()

    if len(words) > LAYOUT_CONFIG.get('max_name_words', 5):
        return False
    if len(words) < 1:
        return False

    first_word = words[0].lower()
    if first_word in DESCRIPTION_CONNECTORS:
        return False

    # Starts with digit followed by lowercase (description: '8 potong sushi')
    if re.match(r'^\d+\s+[a-z]', token):
        return False

    return True

def classify_token(token: str) -> str:
    """Classify a token as: skip / category / price / name."""
    if is_skip(token):
        return 'skip'
    if re.search(LAYOUT_CONFIG.get('hot_ice_pattern', r'HOT\s*[/|]?\s*ICE'), token, re.IGNORECASE):
        return 'hot_ice_header'
    if parse_variant_prices(normalize_price_string(token)):
        return 'price'
    return 'name'

# Token processing
def split_fused_token(token: str) -> list:
    """
    Split tokens where OCR merged adjacent text boxes.

    '..18KNOMU'              → ['..18K', 'NOMU']
    'Sapporo Gyu Udon ..45k' → ['Sapporo Gyu Udon', '..45k']
    'Ebi Cuny Rice O39K'     → ['Ebi Cuny Rice', '39K']
    """
    # Price at start, UPPERCASE text after K (category fused): '..18KNOMU' → ['..18K', 'NOMU']
    match = re.match(r'^([.\s]*\d+\s*[Kk])\s*([A-Z]{2,}.*)$', token)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]

    # Price at start, mixed case text after K: '..32 K Ocha' → ['..32K', 'Ocha']
    match = re.match(r'^([.\s]*\d+\s*[Kk])\s+([A-Za-z].+)$', token)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]

    # Text at start, price at end: 'Sapporo Gyu Udon ..45k' → ['Sapporo Gyu Udon', '..45k']
    match = re.match(r'^([A-Za-z][A-Za-z\s]+?)\s*([.\s]*\d+\s*[Kk][RLrl]?.*)$', token)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]

    # Text at start, price after (no dots): 'NOMU18K' → ['NOMU', '18K']
    match = re.match(r'^([A-Za-z][A-Za-z\s]+?)\s*(\d+\s*[Kk].*)$', token)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]

    # Price + date fused: '..7K 2025/7/16 12:32' → ['..7K', '2025/7/16 12:32']
    match = re.match(r'^([.\s]*\d+\s*[Kk])\s+(\d{4}[/-]\d+.*)$', token)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]

    # O/0 OCR noise before price: 'Ebi Cuny Rice O39K' → ['Ebi Cuny Rice', '39K']
    match = re.match(r'^([A-Za-z][A-Za-z\s]+?)\s*[O0]\s*(\d+\s*[Kk].*)$', token)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]

    return [token]

def merge_tokens(tokens: list, y_coords: list = None) -> list:
    """
    Step 1 — split fused tokens (e.g. '..18KNOMU' → ['..18K', 'NOMU'])
    Step 2 — merge split price tokens (e.g. '38K(R)' + '45K(L)' → '38K(R) 45K(L)')
    """
    # Step 1 — split fused tokens
    split_tokens = []
    split_y      = []
    for token, y in zip(tokens, y_coords or [0] * len(tokens)):
        parts = split_fused_token(token)
        for part in parts:
            split_tokens.append(part)
            split_y.append(y)

    # Step 2 — merge consecutive price tokens
    merged   = []
    merged_y = []
    i = 0

    while i < len(split_tokens):
        token      = split_tokens[i]
        y          = split_y[i]
        normalized = normalize_price_string(token)
        prices     = parse_variant_prices(normalized)

        # Skip merge if token already has 2+ prices
        if len(prices) >= 2:
            merged.append(token)
            merged_y.append(y)
            i += 1
            continue

        next_token  = split_tokens[i + 1] if i + 1 < len(split_tokens) else None
        next_y      = split_y[i + 1]      if i + 1 < len(split_y)      else None
        next_prices = parse_variant_prices(normalize_price_string(next_token)) if next_token else []

        # Merge tagged price with its pair: '38K(R)' + '45K(L)'
        tagged_merge = (
            prices and next_prices and
            prices[0]['tag'] is not None and
            prices[0]['tag'].upper() in ['R', 'S', 'M', 'L']
        )

        # Merge two untagged prices on same row (HOT/ICE): '4K' + '5K'
        untagged_merge = (
            prices and next_prices and
            prices[0]['tag'] is None and
            next_prices[0]['tag'] is None and
            next_y is not None and
            abs(y - next_y) <= 15
        )

        if tagged_merge or untagged_merge:
            merged.append(token + ' ' + next_token)
            merged_y.append(y)
            i += 2
        else:
            merged.append(token)
            merged_y.append(y)
            i += 1

    return merged

# Layout detection
def group_by_rows(ocr_result, y_threshold: int = None) -> list:
    """
    Group OCR boxes into rows by y-coordinate, then split by x-gap.
    Filters descriptions and noise before grouping.

    Returns list of rows, each row is a list of box dicts:
    {'text', 'x_left', 'x_right', 'y', 'conf'}
    """
    if y_threshold is None:
        y_threshold = LAYOUT_CONFIG.get('y_threshold', 40)

    raw_boxes = ocr_result[0] if isinstance(ocr_result[0], list) else ocr_result
    boxes = []

    for line in raw_boxes:
        bbox, (text, conf) = line
        boxes.append({
            'text':    correct_common_ocr_errors(text),
            'x_left':  bbox[0][0],
            'x_right': bbox[1][0],
            'y':       (bbox[0][1] + bbox[2][1]) / 2,
            'conf':    conf
        })

    # Filter: keep prices and valid item names only
    boxes = [
        b for b in boxes
        if parse_variant_prices(normalize_price_string(b['text']))   # keep prices
        or (not is_skip(b['text']) and is_menu_item_name(b['text'])) # keep item names
    ]

    # Sort top-to-bottom
    boxes.sort(key=lambda b: b['y'])

    if not boxes:
        return []

    # Group into rows by y-anchor
    rows        = []
    current_row = [boxes[0]]

    for i in range(1, len(boxes)):
        if abs(boxes[i]['y'] - current_row[0]['y']) <= y_threshold:
            current_row.append(boxes[i])
        else:
            current_row.sort(key=lambda b: b['x_left'])
            rows.append(current_row)
            current_row = [boxes[i]]

    current_row.sort(key=lambda b: b['x_left'])
    rows.append(current_row)

    # Split rows with large x-gap (separate columns)
    x_gap      = LAYOUT_CONFIG.get('x_gap', 400)
    final_rows = []

    for row in rows:
        temp_row = [row[0]]
        for j in range(1, len(row)):
            gap = row[j]['x_left'] - row[j - 1]['x_right']
            if gap > x_gap:
                final_rows.append(temp_row)
                temp_row = [row[j]]
            else:
                temp_row.append(row[j])
        final_rows.append(temp_row)

    return final_rows

# Menu item extraction
def process_tokens(tokens: list) -> list:
    """
    Classify tokens and pair names with prices into structured menu items.

    Returns list of dicts: {'name', 'price'} or {'name', 'prices': [...]}
    """
    results          = []
    current_category = None
    current_name     = None
    price_tags       = [None, None]

    for token in tokens:
        kind = classify_token(token)

        if kind == 'skip':
            continue

        if kind == 'hot_ice_header':
            price_tags = ['HOT', 'ICE']
            continue

        if kind == 'category':
            current_category = token
            price_tags       = [None, None]
            continue

        if kind == 'name':
            current_name = token
            continue

        if kind == 'price':
            if current_name is None:
                # Try to append to last item (price arrived before name)
                if results:
                    normalized = normalize_price_string(token)
                    prices     = parse_variant_prices(normalized)
                    if prices:
                        last = results[-1]
                        if 'prices' in last:
                            last['prices'].extend(prices)
                        elif 'price' in last:
                            old = last.pop('price')
                            last['prices'] = [{'amount': old, 'tag': None, 'variant': None}]
                            last['prices'].extend(prices)
                continue

            normalized = normalize_price_string(token)
            prices     = parse_variant_prices(normalized)

            # Apply HOT/ICE tags if active
            if price_tags != [None, None]:
                for idx, p in enumerate(prices):
                    if idx < len(price_tags) and price_tags[idx]:
                        p['tag']     = price_tags[idx]
                        p['variant'] = price_tags[idx]

            item = {'name': current_name}
            if current_category:
                item['category'] = current_category

            if len(prices) == 1 and prices[0]['tag'] is None:
                item['price'] = prices[0]['amount']
            else:
                item['prices'] = prices

            results.append(item)
            current_name = None

    return results


# Deduplication
def dedupe_menu_items(items: list, threshold: float = None) -> list:
    """
    2-pass deduplication:
    Pass 1 — group identical names (keep first)
    Pass 2 — remove near-duplicates using Jaccard + sequence similarity
    """
    if threshold is None:
        threshold = DEDUPE_CONFIG.get('threshold', 0.85)

    # Pass 1 — exact name match
    groups = {}
    for item in items:
        key = item.get('name', '').lower().strip()
        if key not in groups:
            groups[key] = item
    deduped = list(groups.values())

    # Pass 2 — similarity-based dedup
    final = []
    for item in deduped:
        is_dupe = False
        for kept in final:
            words_a = set(item.get('name', '').lower().split())
            words_b = set(kept.get('name', '').lower().split())
            if not words_a or not words_b:
                continue

            jaccard = len(words_a & words_b) / len(words_a | words_b)
            seq_sim = calculate_similarity(item.get('name', ''), kept.get('name', ''))
            score   = (jaccard + seq_sim) / 2

            if score >= threshold:
                is_dupe = True
                break

        if not is_dupe:
            final.append(item)

    return final


def dedupe_all(all_results: list, threshold: float = None) -> list:
    """
    Group results by poi_id (extracted from filename),
    collect all items per POI, deduplicate, and sort by name.
    """
    if threshold is None:
        threshold = DEDUPE_CONFIG.get('threshold', 0.85)

    poi_groups = {}
    for result in all_results:
        poi_id = result['image'].split('_')[0]
        if poi_id not in poi_groups:
            poi_groups[poi_id] = []
        poi_groups[poi_id].append(result)

    final = []
    for poi_id, group in poi_groups.items():
        all_items = []
        for result in group:
            all_items.extend(result['items'])

        deduped = dedupe_menu_items(all_items, threshold=threshold)
        deduped.sort(key=lambda x: x.get('name') or '')

        final.append({
            'poi_id':        poi_id,
            'total_items':   len(deduped),
            'source_images': [r['image'] for r in group],
            'items':         deduped
        })

    return final

# LLM cleaning
def clean_menu_with_llm(raw_json_data: dict) -> dict:
    """
    Use Gemini LLM to fix typos, scale prices, normalize names,
    and remove duplicates from extracted menu JSON.
    """
    if llm_client is None:
        print("⚠️ LLM client not initialized — skipping cleaning")
        return raw_json_data

    prompt = LLM_PROMPT_TEMPLATE.format(
        raw_json=json.dumps(raw_json_data, ensure_ascii=False)
    )

    try:
        response     = llm_client.models.generate_content(
            model    = LLM_CONFIG.get('model', 'gemini-2.5-flash-lite'),
            contents = prompt
        )
        raw_text     = response.text
        cleaned_text = raw_text.replace('```json', '').replace('```', '').strip()
        cleaned      = json.loads(cleaned_text)

        # Handle if LLM returns list instead of dict
        if isinstance(cleaned, list):
            cleaned = {
                'poi_id':        raw_json_data.get('poi_id', ''),
                'source_images': raw_json_data.get('source_images', []),
                'items':         cleaned
            }

        cleaned['total_items'] = len(cleaned.get('items', []))
        return cleaned

    except Exception as e:
        print(f"⚠️ LLM cleaning failed: {e}")
        return raw_json_data


# Single image processing
def process_single_image(image_path: str) -> dict:
    """
    Full pipeline for one image:
    preprocess → OCR → group_by_rows → merge_tokens → process_tokens
    """
    image_name = Path(image_path).name

    try:
        # Load iamge
        img    = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        # OCR
        ocr_result = engine.ocr(img, cls=True)
        if not ocr_result or not ocr_result[0]:
            return {'image': image_name, 'status': 'failed',
                    'error': 'No text detected', 'items': []}

        # Layout grouping
        rows   = group_by_rows(ocr_result)
        tokens = []
        y_coords = []
        for row in rows:
            row.sort(key=lambda b: b['x_left'])
            for box in row:
                tokens.append(box['text'])
                y_coords.append(box['y'])

        # Token processing
        tokens     = merge_tokens(tokens, y_coords)
        menu_items = process_tokens(tokens)

        # Validation
        max_items = VALIDATION_CONFIG.get('max_items_per_image', 100)
        if len(menu_items) > max_items:
            print(f"  ⚠️ Too many items ({len(menu_items)}) — possible parsing error")

        return {
            'image':       image_name,
            'status':      'success',
            'total_items': len(menu_items),
            'items':       menu_items
        }

    except Exception as e:
        return {
            'image':  image_name,
            'status': 'error',
            'error':  str(e),
            'items':  []
        }

# Batch processing
def process_all_images() -> list:
    """
    Process all images in INPUT_FOLDER with batch memory management.
    Returns list of result dicts per image.
    """
    image_files = get_input_images()
    all_results = []
    batch_size  = FILE_CONFIG.get('batch_size', 10)
    gc_freq     = PERFORMANCE_CONFIG.get('garbage_collect_frequency', 10)

    if not image_files:
        print(f"❌ No images found in {INPUT_FOLDER}/")
        return []

    print(f"\n✅ Found {len(image_files)} images to process")

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    total_batches = -(-len(image_files) // batch_size)  # ceiling division

    for batch_idx, batch in enumerate(chunks(image_files, batch_size), 1):
        print(f"\n── Batch {batch_idx}/{total_batches} ──────────────────────")

        for image_path in batch:
            image_name = Path(image_path).name
            poi_id     = image_name.split('_')[0]
            print(f"  Processing {image_name} (poi_id={poi_id})...")

            result = process_single_image(image_path)
            result['poi_id'] = poi_id
            all_results.append(result)

            if result['status'] == 'success':
                print(f"  ✅ {result['total_items']} items extracted")
            else:
                print(f"  ❌ {result.get('error', 'Unknown error')}")

        # Memory management every batch
        gc.collect()
        print(f"  🧹 Memory cleared (batch {batch_idx})")

    return all_results


# Save results
def save_results_to_files(final_menu: dict, poi_id: str = 'unknown'):
    """Save final menu to JSON and CSV in PROCESSED_FOLDER."""
    Path(PROCESSED_FOLDER).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = Path(PROCESSED_FOLDER) / f"{poi_id}_menu_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_menu, f, indent=2, ensure_ascii=False)
    print(f"📄 JSON saved: {json_path}")

    # Save CSV
    csv_path = Path(PROCESSED_FOLDER) / f"{poi_id}_menu_{timestamp}.csv"
    csv_data = []

    for item in final_menu.get('items', []):
        if 'price' in item:
            csv_data.append({
                'poi_id':   poi_id,
                'name':     item.get('name', ''),
                'variant':  '',
                'tag':      '',
                'amount':   item.get('price', ''),
            })
        elif 'prices' in item:
            for p in item['prices']:
                csv_data.append({
                    'poi_id':   poi_id,
                    'name':     item.get('name', ''),
                    'variant':  p.get('variant', ''),
                    'tag':      p.get('tag', ''),
                    'amount':   p.get('amount', ''),
                })

    if csv_data:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"📊 CSV saved: {csv_path}")
        print(f"   Total rows: {len(csv_data)}")


def main():
    """
    Full pipeline:
    1. process_all_images   — OCR + token processing per image
    2. dedupe_all           — group by poi_id, deduplicate
    3. clean_menu_with_llm  — LLM typo/price/name correction
    4. dedupe_menu_items    — final dedupe after LLM
    5. save_results_to_files — JSON + CSV output
    """
    print("🧾 OCR Menu Reader")

    setup_folders()

    if DEBUG_CONFIG.get('verbose_output', True):
        stats = get_correction_stats() if 'get_correction_stats' in dir() else {}
        print(f"📚 OCR corrections loaded: {stats.get('total_corrections', len(OCR_CORRECTIONS))}")

    # OCR all images
    print("\n📸 Stage 1: OCR Processing")
    all_results = process_all_images()

    if not all_results:
        print("❌ No results to process")
        return

    # Deduplicate per POI
    print("\n🔄 Stage 2: Deduplication")
    final_results = dedupe_all(all_results)
    print(f"✅ {len(final_results)} POI(s) deduplicated")

    # LLM cleaning
    print("\n🤖 Stage 3: LLM Cleaning")
    final_menu_results = []
    for group in final_results:
        print(f"  Cleaning poi_id={group['poi_id']}...")
        cleaned = clean_menu_with_llm(group)
        cleaned['total_items'] = len(cleaned.get('items', []))
        final_menu_results.append(cleaned)

    # Final deduplication
    print("\n🔄 Stage 4: Final Deduplication")
    final_menus = []
    for result in final_menu_results:
        items = dedupe_menu_items(result['items'])
        final_menus.append({
            'poi_id':        result.get('poi_id', 'unknown'),
            'total_items':   len(items),
            'source_images': result.get('source_images', []),
            'items':         items
        })

    # Save
    print("\n💾 Stage 5: Saving Results")
    for poi_result in final_menus:
        save_results_to_files(poi_result, poi_id=poi_result['poi_id'])

    # Summary
    print(f"\n{'=' * 60}")
    print("PROCESSING SUMMARY")
    print(f"{'=' * 60}")
    total_images = len(all_results)
    successful   = sum(1 for r in all_results if r['status'] == 'success')
    total_items  = sum(m['total_items'] for m in final_menus)

    print(f"✅ Images processed:    {successful}/{total_images}")
    print(f"📊 Total menu items:    {total_items}")
    print(f"🏪 POIs processed:      {len(final_menus)}")
    print(f"📁 Results saved to:    {PROCESSED_FOLDER}/")


if __name__ == "__main__":
    # Example: add runtime corrections before running
    # add_restaurant_corrections({'Cuny': 'Curry', 'Gydon': 'Gyudon'})
    main()