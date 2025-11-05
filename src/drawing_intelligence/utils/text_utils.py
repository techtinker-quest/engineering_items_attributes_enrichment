"""
Text utilities for the Drawing Intelligence System.

Provides functions for text processing, normalization, and unit conversion.
"""

import re
from typing import List, Optional, Dict
from difflib import SequenceMatcher


# Unit conversion constants
UNIT_CONVERSIONS = {
    # Length conversions to mm
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "inch": 25.4,
    "in": 25.4,
    '"': 25.4,
    "ft": 304.8,
    "yard": 914.4,
    # Weight conversions to kg
    "kg": 1.0,
    "g": 0.001,
    "mg": 0.000001,
    "lb": 0.453592,
    "lbs": 0.453592,
    "oz": 0.0283495,
    "ton": 1000.0,
}

# Technical symbol mappings
TECHNICAL_SYMBOLS = {
    "Ø": "diameter",
    "ø": "diameter",
    "⌀": "diameter",
    "±": "plus_minus",
    "°": "degrees",
    "′": "minutes",
    "″": "seconds",
    "□": "square",
    "∠": "angle",
    "⊥": "perpendicular",
    "∥": "parallel",
    "≈": "approximately",
    "≤": "less_than_or_equal",
    "≥": "greater_than_or_equal",
}


def normalize_whitespace(text: str) -> str:
    """
    Normalize multiple spaces, tabs, newlines to single space.

    Args:
        text: Input text

    Returns:
        Normalized text with single spaces
    """
    # Replace all whitespace sequences with single space
    normalized = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    return normalized.strip()


def remove_special_characters(text: str, keep: Optional[str] = None) -> str:
    """
    Remove special characters, optionally keeping some.

    Args:
        text: Input text
        keep: String of characters to keep (e.g., '.-_')

    Returns:
        Cleaned text
    """
    if keep:
        # Keep alphanumeric, spaces, and specified characters
        pattern = f"[^a-zA-Z0-9\\s{re.escape(keep)}]"
    else:
        # Keep only alphanumeric and spaces
        pattern = r"[^a-zA-Z0-9\s]"

    cleaned = re.sub(pattern, "", text)
    return normalize_whitespace(cleaned)


def extract_numbers(text: str) -> List[float]:
    """
    Extract all numeric values from text.

    Handles integers, decimals, and numbers with separators.

    Args:
        text: Input text

    Returns:
        List of numbers found
    """
    # Pattern matches: 123, 123.45, 1,234.56, -123.45
    pattern = r"-?\d+(?:[.,]\d+)*"
    matches = re.findall(pattern, text)

    numbers = []
    for match in matches:
        # Replace comma with dot for decimal separator
        # (handles both 1,234.56 and European 1.234,56)
        normalized = match.replace(",", ".")
        try:
            numbers.append(float(normalized))
        except ValueError:
            continue

    return numbers


def detect_measurement_unit(text: str) -> Optional[str]:
    """
    Detect measurement unit in text.

    Args:
        text: Input text (e.g., "25.4mm", "1.5 inches")

    Returns:
        Unit string or None if not found
    """
    text_lower = text.lower()

    # Check each known unit
    for unit in UNIT_CONVERSIONS.keys():
        # Match unit as whole word or at end
        pattern = rf"\b{re.escape(unit)}\b|{re.escape(unit)}$"
        if re.search(pattern, text_lower):
            return unit

    return None


def convert_unit(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert between measurement units.

    Args:
        value: Numeric value
        from_unit: Source unit (e.g., 'inch', 'mm')
        to_unit: Target unit

    Returns:
        Converted value

    Raises:
        ValueError: If units are unknown or incompatible
    """
    from_unit_lower = from_unit.lower()
    to_unit_lower = to_unit.lower()

    if from_unit_lower not in UNIT_CONVERSIONS:
        raise ValueError(f"Unknown source unit: {from_unit}")

    if to_unit_lower not in UNIT_CONVERSIONS:
        raise ValueError(f"Unknown target unit: {to_unit}")

    # Check if units are compatible (both length or both weight)
    length_units = {"mm", "cm", "m", "inch", "in", '"', "ft", "yard"}
    weight_units = {"kg", "g", "mg", "lb", "lbs", "oz", "ton"}

    from_is_length = from_unit_lower in length_units
    to_is_length = to_unit_lower in length_units
    from_is_weight = from_unit_lower in weight_units
    to_is_weight = to_unit_lower in weight_units

    if (from_is_length and not to_is_length) or (from_is_weight and not to_is_weight):
        raise ValueError(f"Incompatible units: {from_unit} to {to_unit}")

    # Convert through base unit
    base_value = value * UNIT_CONVERSIONS[from_unit_lower]
    result = base_value / UNIT_CONVERSIONS[to_unit_lower]

    return result


def fuzzy_match(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """
    Fuzzy string matching using sequence matcher.

    Args:
        text1: First string
        text2: Second string
        threshold: Similarity threshold (0.0 to 1.0)

    Returns:
        True if strings are similar above threshold
    """
    # Normalize for comparison
    text1_norm = text1.lower().strip()
    text2_norm = text2.lower().strip()

    # Calculate similarity ratio
    ratio = SequenceMatcher(None, text1_norm, text2_norm).ratio()

    return ratio >= threshold


def extract_pattern(text: str, pattern: str) -> Optional[str]:
    """
    Extract first match of regex pattern from text.

    Args:
        text: Input text
        pattern: Regular expression pattern

    Returns:
        Matched string or None
    """
    match = re.search(pattern, text)
    if match:
        return match.group(0)
    return None


def normalize_technical_symbols(text: str) -> str:
    """
    Normalize technical symbols to readable text.

    Args:
        text: Input text with symbols

    Returns:
        Text with symbols replaced by readable equivalents
    """
    result = text
    for symbol, replacement in TECHNICAL_SYMBOLS.items():
        result = result.replace(symbol, replacement)
    return result


def extract_dimension_value(text: str) -> Optional[Dict[str, any]]:
    """
    Extract dimension value with tolerance and unit from text.

    Handles formats like:
    - "25.4mm"
    - "25.4 ± 0.1 mm"
    - "Ø25.4mm"
    - "1.5 inch"

    Args:
        text: Input text

    Returns:
        Dict with 'value', 'tolerance', 'unit', 'is_diameter' or None
    """
    # Pattern for dimension: [Ø][value][±tolerance][unit]
    pattern = r'([Øø⌀])?\s*(\d+\.?\d*)\s*([±]?\s*\d+\.?\d*)?\s*([a-zA-Z"′″]+)?'
    match = re.search(pattern, text)

    if not match:
        return None

    diameter_symbol, value_str, tolerance_str, unit_str = match.groups()

    try:
        value = float(value_str)
    except (ValueError, TypeError):
        return None

    # Parse tolerance
    tolerance = None
    if tolerance_str:
        tolerance_str = tolerance_str.strip().replace("±", "").strip()
        try:
            tolerance = float(tolerance_str)
        except (ValueError, TypeError):
            pass

    # Detect unit
    unit = None
    if unit_str:
        unit = detect_measurement_unit(unit_str)

    return {
        "value": value,
        "tolerance": tolerance,
        "unit": unit,
        "is_diameter": diameter_symbol is not None,
    }


def split_camel_case(text: str) -> str:
    """
    Split camelCase or PascalCase text into words.

    Args:
        text: CamelCase text

    Returns:
        Space-separated words
    """
    # Insert space before uppercase letters
    result = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return result


def remove_duplicate_spaces(text: str) -> str:
    """
    Remove duplicate spaces while preserving single spaces.

    Args:
        text: Input text

    Returns:
        Text with single spaces only
    """
    return " ".join(text.split())


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.

    Args:
        text: Input text
        max_length: Maximum length including suffix
        suffix: Suffix to add if truncated (default: '...')

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    truncate_at = max_length - len(suffix)
    return text[:truncate_at] + suffix


def extract_part_number_candidates(text: str) -> List[str]:
    """
    Extract potential part numbers from text.

    Part numbers typically contain alphanumeric patterns with dashes/underscores.

    Args:
        text: Input text

    Returns:
        List of candidate part numbers
    """
    # Pattern: 2-10 uppercase letters, followed by 4-20 alphanumeric/dash/underscore
    pattern = r"\b[A-Z]{2,10}[-_]?[A-Z0-9]{4,20}(?:[-_][A-Z0-9]+)?\b"
    matches = re.findall(pattern, text)
    return matches


def clean_ocr_text(text: str) -> str:
    """
    Clean common OCR errors and artifacts.

    Args:
        text: OCR-extracted text

    Returns:
        Cleaned text
    """
    # Remove common OCR artifacts
    text = text.replace("|", "I")  # Pipe often misread as I
    text = text.replace("O", "0")  # Letter O to zero in numeric contexts
    text = text.replace("l", "1")  # Lowercase L to 1 in numeric contexts

    # Remove multiple consecutive punctuation
    text = re.sub(r"[.,;:!?]{2,}", ".", text)

    # Fix common spacing issues
    text = re.sub(r"\s([.,;:!?])", r"\1", text)  # Remove space before punctuation

    # Normalize whitespace
    text = normalize_whitespace(text)

    return text


def extract_words(text: str, min_length: int = 2) -> List[str]:
    """
    Extract words from text, filtering by minimum length.

    Args:
        text: Input text
        min_length: Minimum word length to include

    Returns:
        List of words
    """
    # Extract alphanumeric sequences
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text)
    # Filter by length
    return [w for w in words if len(w) >= min_length]


def is_numeric_text(text: str) -> bool:
    """
    Check if text is primarily numeric.

    Args:
        text: Input text

    Returns:
        True if text is mostly numbers
    """
    # Remove spaces and common separators
    cleaned = text.replace(" ", "").replace(",", "").replace(".", "")
    if not cleaned:
        return False

    # Count digits
    digit_count = sum(c.isdigit() for c in cleaned)
    ratio = digit_count / len(cleaned)

    return ratio > 0.7


def normalize_line_endings(text: str) -> str:
    """
    Normalize line endings to Unix style (LF).

    Args:
        text: Input text

    Returns:
        Text with normalized line endings
    """
    # Replace Windows (CRLF) and old Mac (CR) line endings with Unix (LF)
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    return text


def capitalize_first_letter(text: str) -> str:
    """
    Capitalize first letter of text, leaving rest unchanged.

    Args:
        text: Input text

    Returns:
        Text with first letter capitalized
    """
    if not text:
        return text
    return text[0].upper() + text[1:]


def count_words(text: str) -> int:
    """
    Count number of words in text.

    Args:
        text: Input text

    Returns:
        Word count
    """
    words = extract_words(text, min_length=1)
    return len(words)
