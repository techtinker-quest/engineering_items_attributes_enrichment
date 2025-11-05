"""
Unit tests for EntityExtractor.
"""

import pytest
from src.drawing_intelligence.processing.entity_extractor import (
    EntityExtractor,
    EntityConfig,
)
from src.drawing_intelligence.models.data_structures import (
    OCRResult,
    TextBlock,
    BoundingBox,
    EntityType,
)


@pytest.fixture
def entity_config():
    """Create test configuration."""
    return EntityConfig(
        use_regex=True,
        use_spacy=False,  # Disable for unit tests
        use_llm=False,
        confidence_threshold=0.80,
        normalize_units=True,
    )


@pytest.fixture
def entity_extractor(entity_config):
    """Create EntityExtractor instance."""
    return EntityExtractor(entity_config, llm_gateway=None)


@pytest.fixture
def sample_ocr_result():
    """Create sample OCR result."""
    text_blocks = [
        TextBlock(
            text_id="TXT-001",
            content="PART NO: ABC-12345-REV2",
            bbox=BoundingBox(x=100, y=100, width=200, height=20),
            confidence=0.95,
            ocr_engine="paddleocr",
            region_type="text",
        ),
        TextBlock(
            text_id="TXT-002",
            content="Ø25.4 ± 0.1 mm",
            bbox=BoundingBox(x=100, y=130, width=150, height=20),
            confidence=0.92,
            ocr_engine="paddleocr",
            region_type="dimension",
        ),
        TextBlock(
            text_id="TXT-003",
            content="MATERIAL: Steel 304",
            bbox=BoundingBox(x=100, y=160, width=180, height=20),
            confidence=0.93,
            ocr_engine="paddleocr",
            region_type="text",
        ),
        TextBlock(
            text_id="TXT-004",
            content="WEIGHT: 2.5 kg",
            bbox=BoundingBox(x=100, y=190, width=130, height=20),
            confidence=0.91,
            ocr_engine="paddleocr",
            region_type="text",
        ),
        TextBlock(
            text_id="TXT-005",
            content="M8 x 1.25",
            bbox=BoundingBox(x=100, y=220, width=100, height=20),
            confidence=0.89,
            ocr_engine="paddleocr",
            region_type="text",
        ),
    ]

    return OCRResult(
        text_blocks=text_blocks,
        language_detected="en",
        layout_regions=[],
        average_confidence=0.92,
    )


class TestEntityExtractor:
    """Test EntityExtractor class."""

    def test_initialization(self, entity_extractor):
        """Test extractor initialization."""
        assert entity_extractor.use_regex is True
        assert entity_extractor.use_spacy is False
        assert entity_extractor.use_llm is False

    def test_extract_part_number(self, entity_extractor, sample_ocr_result):
        """Test part number extraction."""
        result = entity_extractor.extract_entities(sample_ocr_result)

        part_numbers = [
            e for e in result.entities if e.entity_type == EntityType.PART_NUMBER
        ]

        assert len(part_numbers) > 0
        assert "ABC-12345-REV2" in part_numbers[0].value
        assert part_numbers[0].confidence > 0.8

    def test_extract_dimension(self, entity_extractor, sample_ocr_result):
        """Test dimension extraction."""
        result = entity_extractor.extract_entities(sample_ocr_result)

        dimensions = [
            e for e in result.entities if e.entity_type == EntityType.DIMENSION
        ]

        assert len(dimensions) > 0
        dim = dimensions[0]
        assert dim.normalized_value is not None
        assert "value" in dim.normalized_value
        assert dim.normalized_value["value"] == 25.4

    def test_extract_material(self, entity_extractor, sample_ocr_result):
        """Test material extraction."""
        result = entity_extractor.extract_entities(sample_ocr_result)

        materials = [e for e in result.entities if e.entity_type == EntityType.MATERIAL]

        assert len(materials) > 0
        assert "Steel 304" in materials[0].value

    def test_extract_weight(self, entity_extractor, sample_ocr_result):
        """Test weight extraction."""
        result = entity_extractor.extract_entities(sample_ocr_result)

        weights = [e for e in result.entities if e.entity_type == EntityType.WEIGHT]

        assert len(weights) > 0
        weight = weights[0]
        assert weight.normalized_value is not None
        assert weight.normalized_value["value"] == 2.5
        assert weight.normalized_value["unit"] == "kg"

    def test_extract_thread_spec(self, entity_extractor, sample_ocr_result):
        """Test thread specification extraction."""
        result = entity_extractor.extract_entities(sample_ocr_result)

        threads = [
            e for e in result.entities if e.entity_type == EntityType.THREAD_SPEC
        ]

        assert len(threads) > 0
        assert "M8" in threads[0].value

    def test_normalize_units(self, entity_extractor):
        """Test unit normalization."""
        # Test dimension normalization
        entity = type(
            "Entity", (), {"entity_type": EntityType.DIMENSION, "value": "1 inch"}
        )()

        normalized = entity_extractor._normalize_entity(EntityType.DIMENSION, "1 inch")

        assert normalized is not None
        assert "value" in normalized
        # 1 inch = 25.4 mm
        assert abs(normalized["value"] - 25.4) < 0.1

    def test_extraction_statistics(self, entity_extractor, sample_ocr_result):
        """Test extraction statistics."""
        result = entity_extractor.extract_entities(sample_ocr_result)

        stats = result.extraction_statistics

        assert stats.total_entities > 0
        assert stats.average_confidence > 0.0
        assert len(stats.entities_by_type) > 0
        assert "regex" in stats.entities_by_method

    def test_empty_ocr_result(self, entity_extractor):
        """Test with empty OCR result."""
        empty_ocr = OCRResult(
            text_blocks=[],
            language_detected="en",
            layout_regions=[],
            average_confidence=0.0,
        )

        result = entity_extractor.extract_entities(empty_ocr)

        assert len(result.entities) == 0
