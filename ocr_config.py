# %%
# Import library
import os
from pathlib import Path
from dotenv import load_dotenv

# %%
# Import corrections from another file
try:
    from fnb_menu.code.corrections import SKIP_PATTERNS, DESCRIPTION_CONNECTORS, OCR_CORRECTIONS, LLM_PROMPT_TEMPLATE
    print(f"Loaded corrections.")
except ImportError: 
    print("ocr_corrections.py is not found, try use basic corrections.")

# %%
# Core OCR settings
OCR_CONFIG = {
    # Engine settings
    'use_angle_cls': True,              # Enable text angle classification
    'lang': 'en',                       # Primary language is english
    'use_gpu': False,                   # Set True if have CUDA GPU
    'show_log': False,                   # Set True if want to see log information

    # Detection settings
    'det_db_thresh': 0.20,              # Lower to detect more text (but more noise)
    'det_db_box_thresh': 0.30,          # Higher to filter low-confidence boxes
    'det_db_unclip_ratio': 2.0,         # Higher to make the box larger
}

# %%
# Price settings
PRICE_CONFIG = {
    # Price range
    'min_price': 500,
    'max_price': 5000000,

    # Currency 
    'currency_keywords': ['RP', 'rp', 'Rp', 'IDR', 'idr'],

    # Price suffixes
    'price_suffix_multipliers': {'K': 1000, 'k': 1000,
                                'RB': 1000, 'rb': 1000,
                                'JT': 1000000, 'jt': 1000000},
    
    # Variant labels
    'variant_labels': {'O': 'Original', 'R': 'Regular',
                       'S': 'Small', 'M': 'Medium',
                       'L': 'Large', 'XL': 'Extra Large'},
    
    # Multi-price support
    'allow_multiple_prices': True,
    'variant_tag_pattern': r'\(([^)]{1,10})\)',
}

# %%
# Layout settings
LAYOUT_CONFIG = {
    'y_threshold': 40,                 # Max y diff to be in the same row
    'max_name_words': 5,               # Max words for item name
}

# %%
# Deduplication settings
DEDUPE_CONFIG = {
    'threshold': 0.85,                 # Similarity score threshold
}

# %%
# Load .env file
load_dotenv()

# LLM settings
LLM_CONFIG = {
    'model':   'gemini-2.5-flash-lite',
    'api_key': os.getenv('gemini_api_key')
}

# %%
# File processing settings
FILE_CONFIG = {
    # Supported file types
    'supported_extensions': ['.jpg', '.jpeg', '.png'],

    # Output settings
    'output_formats': ['json', 'csv'],
    'include_timestamp': True,

    # Batch processing 
    'batch_size': 10,                   # Images per batch
    'parallel_processing': False,       # PaddleOCR not thread-safe, keep False
}

# %%
# Debug settings
DEBUG_CONFIG = {
    'enable_debug':           False,    # Enable debug mode
    'log_level':              'INFO',   # DEBUG, INFO, WARNING, ERROR
    'verbose_output':         True,     # Show detailed processing info
    'log_processing_time':    True,     # Log time per step
    'show_progress_bar':      True,     # Show progress bar for batch
    'suppress_paddleocr_logs': True,    # Hide PaddleOCR debug messages
    'save_debug_images':      False,    # Save intermediate images
}

# %%
# Performance settings
PERFORMANCE_CONFIG = {
    'max_workers':                 4,     # Worker threads for batch processing
    'timeout_seconds':             30,    # Max processing time per image
    'max_retry_attempts':          2,     # Retry failed images N times
    'garbage_collect_frequency':   10,    # GC every N images
    'clear_memory_between_batches': True, # Clear memory between batches
}

# %%
# Validation settings
VALIDATION_CONFIG = {
    'validate_prices':         True,    # Check if prices are reasonable
    'validate_text_quality':   True,    # Filter low-quality text
    'min_items_per_image':     0,       # Minimum items for success
    'max_items_per_image':     100,     # Maximum items (safety check)
    'min_average_confidence':  0.3,     # Minimum average confidence
    'check_duplicate_items':   True,    # Flag duplicate menu items
}

# %%
# Folder paths
BASE_DIR         = Path(os.getcwd())
FOLDER_CONFIG = {
    'input_folder':    str(BASE_DIR.parent / 'data_raw' / 'input_images'),
    'output_folder':   str(BASE_DIR.parent / 'data_raw' / 'processed_images'),
    'debug_folder':    str(BASE_DIR.parent / 'data_raw' / 'processed_images' / 'debug'),
    'logs_folder':     str(BASE_DIR.parent / 'logs'),
}

# %%
# Environment overrides
if os.getenv('ENV') == 'development':
    DEBUG_CONFIG['enable_debug']              = True
    DEBUG_CONFIG['save_debug_images']         = True
    DEBUG_CONFIG['log_level']                 = 'DEBUG'
    PERFORMANCE_CONFIG['timeout_seconds']     = 60

elif os.getenv('ENV') == 'production':
    DEBUG_CONFIG['enable_debug']              = False
    DEBUG_CONFIG['suppress_paddleocr_logs']   = True
    PERFORMANCE_CONFIG['timeout_seconds']     = 15

elif os.getenv('ENV') == 'testing':
    VALIDATION_CONFIG['min_items_per_image']  = 1
    FILE_CONFIG['output_formats']             = ['json']
    DEBUG_CONFIG['log_level']                 = 'WARNING'

# %%
# Configuration validation
def validate_config():
    """Validate configuration settings for common errors."""
    errors = []

    if PRICE_CONFIG['min_price'] >= PRICE_CONFIG['max_price']:
        errors.append("PRICE min_price must be less than max_price")

    if not 0.0 <= DEDUPE_CONFIG['threshold'] <= 1.0:
        errors.append("DEDUPE threshold must be between 0.0 and 1.0")

    if PERFORMANCE_CONFIG['max_workers'] < 1:
        errors.append("PERFORMANCE max_workers must be at least 1")

    if PERFORMANCE_CONFIG['timeout_seconds'] < 5:
        errors.append("PERFORMANCE timeout_seconds should be at least 5")

    for folder_key, folder_path in FOLDER_CONFIG.items():
        if not isinstance(folder_path, str) or not folder_path.strip():
            errors.append(f"FOLDER {folder_key} must be a non-empty string")

    if not LLM_CONFIG.get('api_key'):
        errors.append("LLM api_key is not set. Check the .env file")

    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return True

# %%
# Export configuration
__all__ = [
    'OCR_CONFIG',
    'PRICE_CONFIG',
    'LAYOUT_CONFIG',
    'CATEGORY_CONFIG',
    'DEDUPE_CONFIG',
    'LLM_CONFIG',
    'FILE_CONFIG',
    'DEBUG_CONFIG',
    'PERFORMANCE_CONFIG',
    'VALIDATION_CONFIG',
    'FOLDER_CONFIG',
    'validate_config',
]

# %%
# Validate on import
if __name__ != '__main__' and os.getenv('ENV') != 'development':
    try:
        validate_config()
        if DEBUG_CONFIG.get('verbose_output', True):
            print("Configuration validated successfully.")
    except ValueError as e:
        print(f"Configuration Warning: {e}")
        print("Please check your ocr_config.py and ocr_corrections.py files")

if __name__ == '__main__':
    validate_config()
    print("Configuration ready!")