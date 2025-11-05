"""
Unit tests for text_utils module.

Tests text normalization, unit conversion, dimension parsing, and fuzzy matching.
"""

import pytest
from src.drawing_intelligence.utils.text_utils import (
    normalize_text,
    normalize_technical_symbols,
    extract_numbers,
    extract_dimension_value,
    convert_unit,
    parse_tolerance,
    extract_part_number_candidates,
    clean_ocr_text,
    fuzzy_match,
    normalize_whitespace,
)


class TestTextNormalization:
    """Test text normalization functions."""

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "Hello    World\n\n\tTest"
        result = normalize_whitespace(text)
        assert result == "Hello World Test"

    def test_normalize_text_basic(self):
        """Test basic text normalization."""
        text = "  HELLO  World  123  "
        result = normalize_text(text)
        assert result == "hello world 123"

    def test_normalize_technical_symbols_diameter(self):
        """Test diameter symbol normalization."""
        text = "Ø25.4mm"
        result = normalize_technical_symbols(text)
        assert "diameter" in result.lower()

    def test_normalize_technical_symbols_tolerance(self):
        """Test tolerance symbol normalization."""
        text = "±0.05mm"
        result = normalize_technical_symbols(text)
        assert "plus_minus" in result.lower() or "+/-" in result

    def test_normalize_technical_symbols_degree(self):
        """Test degree symbol normalization."""
        text = "45°"
        result = normalize_technical_symbols(text)
        assert "degree" in result.lower() or "deg" in result

    def test_normalize_multiple_symbols(self):
        """Test normalizing multiple symbols."""
        text = "Ø25mm ±0.1° angle"
        result = normalize_technical_symbols(text)
        assert all(x in result.lower() for x in ["diameter", "plus_minus", "degree"])


class TestNumberExtraction:
    """Test number extraction functions."""

    def test_extract_single_number(self):
        """Test extracting single number."""
        text = "The value is 42.5"
        numbers = extract_numbers(text)
        assert len(numbers) == 1
        assert numbers[0] == 42.5

    def test_extract_multiple_numbers(self):
        """Test extracting multiple numbers."""
        text = "Dimensions: 25.4 x 10.2 x 5.0"
        numbers = extract_numbers(text)
        assert len(numbers) == 3
        assert numbers == [25.4, 10.2, 5.0]

    def test_extract_negative_numbers(self):
        """Test extracting negative numbers."""
        text = "Temperature: -5.5°C"
        numbers = extract_numbers(text)
        assert -5.5 in numbers

    def test_extract_scientific_notation(self):
        """Test extracting scientific notation."""
        text = "Value: 1.5e-3"
        numbers = extract_numbers(text)
        assert len(numbers) == 1
        assert abs(numbers[0] - 0.0015) < 1e-6

    def test_extract_no_numbers(self):
        """Test text with no numbers."""
        text = "No numbers here"
        numbers = extract_numbers(text)
        assert len(numbers) == 0


class TestDimensionParsing:
    """Test dimension value extraction."""

    def test_extract_simple_dimension(self):
        """Test simple dimension extraction."""
        text = "25.4mm"
        result = extract_dimension_value(text)
        assert result["value"] == 25.4
        assert result["unit"] == "mm"
        assert result["tolerance"] is None

    def test_extract_dimension_with_tolerance(self):
        """Test dimension with tolerance."""
        text = "25.4 ± 0.1 mm"
        result = extract_dimension_value(text)
        assert result["value"] == 25.4
        assert result["tolerance"] == 0.1
        assert result["unit"] == "mm"

    def test_extract_diameter_dimension(self):
        """Test diameter dimension."""
        text = "Ø25.4mm"
        result = extract_dimension_value(text)
        assert result["value"] == 25.4
        assert result["unit"] == "mm"
        assert result["is_diameter"] is True

    def test_extract_dimension_inches(self):
        """Test inch dimension."""
        text = '1.5"'
        result = extract_dimension_value(text)
        assert result["value"] == 1.5
        assert result["unit"] in ["in", "inch", '"']

    def test_extract_dimension_with_asymmetric_tolerance(self):
        """Test dimension with +/- tolerance."""
        text = "100.0 +0.5/-0.3 mm"
        result = extract_dimension_value(text)
        assert result["value"] == 100.0
        assert "tolerance_plus" in result
        assert "tolerance_minus" in result

    def test_extract_dimension_no_space(self):
        """Test dimension without spaces."""
        text = "50mm"
        result = extract_dimension_value(text)
        assert result["value"] == 50.0
        assert result["unit"] == "mm"

    def test_extract_dimension_invalid(self):
        """Test invalid dimension returns None."""
        text = "not a dimension"
        result = extract_dimension_value(text)
        assert result is None


class TestUnitConversion:
    """Test unit conversion functions."""

    def test_convert_mm_to_inch(self):
        """Test mm to inch conversion."""
        result = convert_unit(25.4, "mm", "inch")
        assert abs(result - 1.0) < 0.001

    def test_convert_inch_to_mm(self):
        """Test inch to mm conversion."""
        result = convert_unit(1.0, "inch", "mm")
        assert abs(result - 25.4) < 0.001

    def test_convert_cm_to_mm(self):
        """Test cm to mm conversion."""
        result = convert_unit(10.0, "cm", "mm")
        assert result == 100.0

    def test_convert_m_to_mm(self):
        """Test m to mm conversion."""
        result = convert_unit(1.0, "m", "mm")
        assert result == 1000.0

    def test_convert_kg_to_lb(self):
        """Test kg to lb conversion."""
        result = convert_unit(1.0, "kg", "lb")
        assert abs(result - 2.20462) < 0.001

    def test_convert_lb_to_kg(self):
        """Test lb to kg conversion."""
        result = convert_unit(2.20462, "lb", "kg")
        assert abs(result - 1.0) < 0.001

    def test_convert_same_unit(self):
        """Test converting to same unit."""
        result = convert_unit(100.0, "mm", "mm")
        assert result == 100.0

    def test_convert_invalid_unit(self):
        """Test invalid unit conversion."""
        with pytest.raises(ValueError, match="Unknown unit"):
            convert_unit(100.0, "invalid", "mm")

    def test_convert_incompatible_units(self):
        """Test incompatible unit conversion."""
        with pytest.raises(ValueError, match="Cannot convert"):
            convert_unit(100.0, "mm", "kg")


class TestToleranceParsing:
    """Test tolerance parsing."""

    def test_parse_symmetric_tolerance(self):
        """Test symmetric tolerance parsing."""
        text = "±0.05"
        result = parse_tolerance(text)
        assert result["type"] == "symmetric"
        assert result["value"] == 0.05

    def test_parse_asymmetric_tolerance(self):
        """Test asymmetric tolerance parsing."""
        text = "+0.1/-0.05"
        result = parse_tolerance(text)
        assert result["type"] == "asymmetric"
        assert result["plus"] == 0.1
        assert result["minus"] == 0.05

    def test_parse_single_sided_tolerance(self):
        """Test single-sided tolerance."""
        text = "+0.05"
        result = parse_tolerance(text)
        assert result["type"] == "single"
        assert result["value"] == 0.05
        assert result["direction"] == "plus"

    def test_parse_no_tolerance(self):
        """Test text without tolerance."""
        text = "no tolerance here"
        result = parse_tolerance(text)
        assert result is None


class TestPartNumberExtraction:
    """Test part number extraction."""

    def test_extract_standard_part_number(self):
        """Test standard part number format."""
        text = "Part Number: ABC-12345-REV-A"
        candidates = extract_part_number_candidates(text)
        assert len(candidates) > 0
        assert "ABC-12345-REV-A" in candidates

    def test_extract_part_number_no_dashes(self):
        """Test part number without dashes."""
        text = "PN: XYZ123456"
        candidates = extract_part_number_candidates(text)
        assert len(candidates) > 0

    def test_extract_multiple_candidates(self):
        """Test extracting multiple candidates."""
        text = "Parts: ABC-123, XYZ-456, DEF-789"
        candidates = extract_part_number_candidates(text)
        assert len(candidates) >= 3

    def test_extract_part_number_with_revision(self):
        """Test part number with revision."""
        text = "P/N: PART-001-R02"
        candidates = extract_part_number_candidates(text)
        assert any("PART-001" in c for c in candidates)

    def test_extract_no_part_numbers(self):
        """Test text with no part numbers."""
        text = "Just some random text"
        candidates = extract_part_number_candidates(text)
        # May return empty or very low confidence candidates
        assert isinstance(candidates, list)


class TestOCRCleaning:
    """Test OCR text cleaning."""

    def test_clean_common_ocr_errors(self):
        """Test cleaning common OCR mistakes."""
        text = "l00 (should be 100)"
        result = clean_ocr_text(text)
        assert "100" in result

    def test_clean_o_to_0(self):
        """Test O to 0 conversion in numbers."""
        text = "Part O123"  # O should stay as letter
        result = clean_ocr_text(text)
        assert "O123" in result  # O is part of part number

        text2 = "Value: O.5"  # O should be 0
        result2 = clean_ocr_text(text2)
        assert "0.5" in result2

    def test_clean_i_to_1(self):
        """Test I to 1 conversion in numbers."""
        text = "Quantity: i2"
        result = clean_ocr_text(text)
        assert "12" in result

    def test_clean_special_chars(self):
        """Test removing unwanted special characters."""
        text = "Text with ~weird~ characters"
        result = clean_ocr_text(text)
        # Should remove or normalize weird chars
        assert len(result) > 0

    def test_clean_preserves_technical_symbols(self):
        """Test that technical symbols are preserved."""
        text = "Ø25mm ±0.1"
        result = clean_ocr_text(text)
        # Should keep diameter and tolerance symbols
        assert any(c in result for c in ["Ø", "±", "diameter", "plus"])


class TestFuzzyMatching:
    """Test fuzzy string matching."""

    def test_fuzzy_match_exact(self):
        """Test exact match returns 1.0."""
        s1 = "Steel 304"
        s2 = "Steel 304"
        similarity = fuzzy_match(s1, s2)
        assert similarity == 1.0

    def test_fuzzy_match_similar(self):
        """Test similar strings have high similarity."""
        s1 = "Steel 304"
        s2 = "Steel 305"
        similarity = fuzzy_match(s1, s2)
        assert similarity > 0.8

    def test_fuzzy_match_different(self):
        """Test different strings have low similarity."""
        s1 = "Steel 304"
        s2 = "Aluminum 6061"
        similarity = fuzzy_match(s1, s2)
        assert similarity < 0.5

    def test_fuzzy_match_case_insensitive(self):
        """Test case-insensitive matching."""
        s1 = "STEEL"
        s2 = "steel"
        similarity = fuzzy_match(s1, s2)
        assert similarity == 1.0

    def test_fuzzy_match_with_typo(self):
        """Test matching with typo."""
        s1 = "Material"
        s2 = "Materail"  # Common typo
        similarity = fuzzy_match(s1, s2)
        assert similarity > 0.7


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_parse_full_dimension_string(self):
        """Test parsing complete dimension specification."""
        text = "Length: Ø25.4 ±0.1mm (1.000 ±0.004 inch)"

        # Extract dimension
        dim_result = extract_dimension_value(text)
        assert dim_result["value"] == 25.4
        assert dim_result["tolerance"] == 0.1
        assert dim_result["is_diameter"] is True

    def test_normalize_technical_drawing_text(self):
        """Test normalizing technical drawing text."""
        text = """
        PART NO: ABC-12345-REV-A
        MATERIAL: Steel 304
        DIM: Ø25.4 ±0.1mm
        WEIGHT: 2.5 kg
        """

        normalized = normalize_text(text)
        assert "abc-12345-rev-a" in normalized
        assert "steel 304" in normalized

    def test_extract_all_dimensions_from_text(self):
        """Test extracting multiple dimensions."""
        text = "Dimensions: 100mm x 50mm x 25mm"
        numbers = extract_numbers(text)
        assert len(numbers) == 3
        assert numbers == [100.0, 50.0, 25.0]

    def test_clean_and_parse_ocr_output(self):
        """Test cleaning OCR output and parsing."""
        # Simulated poor OCR output
        text = "Part l00i Ø25.4 ±O.1mm"

        # Clean
        cleaned = clean_ocr_text(text)

        # Parse dimension
        dim = extract_dimension_value(cleaned)
        assert dim is not None
        assert dim["value"] == 25.4


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_string(self):
        """Test operations on empty string."""
        assert normalize_text("") == ""
        assert extract_numbers("") == []
        assert extract_dimension_value("") is None

    def test_none_input(self):
        """Test None input handling."""
        with pytest.raises((ValueError, TypeError)):
            normalize_text(None)

    def test_very_long_string(self):
        """Test handling very long strings."""
        text = "A" * 10000
        result = normalize_text(text)
        assert len(result) == 10000

    def test_unicode_characters(self):
        """Test handling unicode characters."""
        text = "测试 Ø25mm"
        result = normalize_text(text)
        assert len(result) > 0

    def test_special_number_formats(self):
        """Test special number formats."""
        # Fractions
        text = "1/4 inch"
        numbers = extract_numbers(text)
        # May or may not parse fractions depending on implementation

        # Thousands separator
        text2 = "1,000.5mm"
        numbers2 = extract_numbers(text2)
        assert 1000.5 in numbers2 or len(numbers2) > 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
