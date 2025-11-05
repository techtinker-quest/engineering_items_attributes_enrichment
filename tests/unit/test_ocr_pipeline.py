"""
Unit tests for ocr_pipeline module.

Tests dual-engine OCR with fallback, layout analysis, and language detection.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.drawing_intelligence.processing.ocr_pipeline import OCRPipeline, OCRConfig
from src.drawing_intelligence.models.data_structures import (
    OCRResult,
    TextBlock,
    LayoutRegion,
    BoundingBox,
)
from src.drawing_intelligence.utils.error_handlers import OCRError


@pytest.fixture
def ocr_config():
    """Create test OCR configuration."""
    return OCRConfig(
        primary_engine="paddleocr",
        fallback_engine="easyocr",
        confidence_threshold=0.85,
        languages=["en"],
        detect_layout=True,
    )


@pytest.fixture
def test_image():
    """Create test image with text."""
    # Create white background with black text simulation
    image = np.ones((200, 400, 3), dtype=np.uint8) * 255
    # Add some "text" regions (black rectangles simulating text)
    image[50:70, 50:150] = 0  # Simulated text block 1
    image[80:100, 50:200] = 0  # Simulated text block 2
    return image


@pytest.fixture
def mock_paddleocr():
    """Create mock PaddleOCR engine."""
    with patch("src.drawing_intelligence.processing.ocr_pipeline.PaddleOCR") as mock:
        yield mock


@pytest.fixture
def mock_easyocr():
    """Create mock EasyOCR engine."""
    with patch("src.drawing_intelligence.processing.ocr_pipeline.easyocr") as mock:
        yield mock


class TestOCRPipelineInitialization:
    """Test OCR pipeline initialization."""

    def test_init_with_config(self, ocr_config, mock_paddleocr, mock_easyocr):
        """Test initialization with configuration."""
        pipeline = OCRPipeline(ocr_config)

        assert pipeline.config == ocr_config
        assert pipeline.config.primary_engine == "paddleocr"
        assert pipeline.config.confidence_threshold == 0.85

    def test_init_with_default_config(self, mock_paddleocr, mock_easyocr):
        """Test initialization with default configuration."""
        pipeline = OCRPipeline()

        assert pipeline.config is not None
        assert pipeline.config.primary_engine == "paddleocr"
        assert pipeline.config.fallback_engine == "easyocr"

    def test_init_engines_lazy_loading(self, ocr_config, mock_paddleocr, mock_easyocr):
        """Test engines are lazily loaded."""
        pipeline = OCRPipeline(ocr_config)

        # Engines should not be initialized until first use
        assert pipeline._primary_engine is None
        assert pipeline._fallback_engine is None

    def test_init_with_invalid_engine(self):
        """Test initialization with invalid engine name."""
        config = OCRConfig(primary_engine="invalid_engine")

        with pytest.raises(ValueError, match="Unsupported OCR engine"):
            pipeline = OCRPipeline(config)


class TestPrimaryOCRExtraction:
    """Test primary OCR engine (PaddleOCR) extraction."""

    def test_extract_text_success(self, ocr_config, test_image, mock_paddleocr):
        """Test successful text extraction with primary engine."""
        # Mock PaddleOCR response
        mock_engine = MagicMock()
        mock_engine.ocr.return_value = [
            [
                [[10, 20], [110, 20], [110, 50], [10, 50]],  # bbox
                ("Part Number: ABC-123", 0.95),  # text, confidence
            ],
            [
                [[120, 30], [220, 30], [220, 60], [120, 60]],
                ("Material: Steel 304", 0.92),
            ],
        ]
        mock_paddleocr.return_value = mock_engine

        pipeline = OCRPipeline(ocr_config)
        result = pipeline.extract_text(test_image)

        assert isinstance(result, OCRResult)
        assert len(result.text_blocks) == 2
        assert result.text_blocks[0].content == "Part Number: ABC-123"
        assert result.text_blocks[0].confidence == 0.95
        assert result.text_blocks[1].content == "Material: Steel 304"

    def test_extract_text_high_confidence(self, ocr_config, test_image, mock_paddleocr):
        """Test extraction with all high-confidence results."""
        mock_engine = MagicMock()
        mock_engine.ocr.return_value = [
            [[[10, 20], [110, 20], [110, 50], [10, 50]], ("High confidence text", 0.98)]
        ]
        mock_paddleocr.return_value = mock_engine

        pipeline = OCRPipeline(ocr_config)
        result = pipeline.extract_text(test_image)

        assert result.average_confidence >= 0.98
        assert all(tb.ocr_engine == "paddleocr" for tb in result.text_blocks)

    def test_extract_text_empty_result(self, ocr_config, test_image, mock_paddleocr):
        """Test extraction with no text found."""
        mock_engine = MagicMock()
        mock_engine.ocr.return_value = [[]]  # No text found
        mock_paddleocr.return_value = mock_engine

        pipeline = OCRPipeline(ocr_config)
        result = pipeline.extract_text(test_image)

        assert isinstance(result, OCRResult)
        assert len(result.text_blocks) == 0
        assert result.full_text == ""

    def test_extract_text_with_special_characters(
        self, ocr_config, test_image, mock_paddleocr
    ):
        """Test extraction of technical symbols."""
        mock_engine = MagicMock()
        mock_engine.ocr.return_value = [
            [[[10, 20], [110, 20], [110, 50], [10, 50]], ("Ø25.4 ± 0.1 mm", 0.90)]
        ]
        mock_paddleocr.return_value = mock_engine

        pipeline = OCRPipeline(ocr_config)
        result = pipeline.extract_text(test_image)

        assert "Ø" in result.text_blocks[0].content
        assert "±" in result.text_blocks[0].content


class TestFallbackOCR:
    """Test fallback OCR engine (EasyOCR) functionality."""

    def test_fallback_triggered_low_confidence(
        self, ocr_config, test_image, mock_paddleocr, mock_easyocr
    ):
        """Test fallback is triggered for low confidence blocks."""
        # Primary OCR returns low confidence
        mock_primary = MagicMock()
        mock_primary.ocr.return_value = [
            [
                [[10, 20], [110, 20], [110, 50], [10, 50]],
                ("l00 (OCR error)", 0.70),  # Low confidence, should trigger fallback
            ]
        ]
        mock_paddleocr.return_value = mock_primary

        # Fallback OCR returns better result
        mock_fallback = MagicMock()
        mock_fallback.readtext.return_value = [
            ([[10, 20], [110, 20], [110, 50], [10, 50]], "100", 0.95)  # Corrected
        ]
        mock_easyocr.Reader.return_value = mock_fallback

        pipeline = OCRPipeline(ocr_config)
        result = pipeline.extract_text(test_image)

        # Should use fallback result
        assert result.text_blocks[0].content == "100"
        assert result.text_blocks[0].confidence == 0.95
        assert result.text_blocks[0].ocr_engine == "easyocr"

    def test_fallback_improves_result(
        self, ocr_config, test_image, mock_paddleocr, mock_easyocr
    ):
        """Test fallback improves low-confidence extraction."""
        mock_primary = MagicMock()
        mock_primary.ocr.return_value = [
            [
                [[10, 20], [110, 20], [110, 50], [10, 50]],
                ("Part O123", 0.75),  # O should be 0
            ]
        ]
        mock_paddleocr.return_value = mock_primary

        mock_fallback = MagicMock()
        mock_fallback.readtext.return_value = [
            ([[10, 20], [110, 20], [110, 50], [10, 50]], "Part 0123", 0.92)
        ]
        mock_easyocr.Reader.return_value = mock_fallback

        pipeline = OCRPipeline(ocr_config)
        result = pipeline.extract_text(test_image)

        assert result.text_blocks[0].content == "Part 0123"
        assert result.text_blocks[0].ocr_engine == "easyocr"

    def test_fallback_not_triggered_high_confidence(
        self, ocr_config, test_image, mock_paddleocr, mock_easyocr
    ):
        """Test fallback is NOT triggered for high confidence."""
        mock_primary = MagicMock()
        mock_primary.ocr.return_value = [
            [
                [[10, 20], [110, 20], [110, 50], [10, 50]],
                ("Clear Text", 0.95),  # High confidence
            ]
        ]
        mock_paddleocr.return_value = mock_primary

        pipeline = OCRPipeline(ocr_config)
        result = pipeline.extract_text(test_image)

        # Should NOT use fallback
        assert result.text_blocks[0].ocr_engine == "paddleocr"
        # EasyOCR should not have been called
        mock_easyocr.Reader.assert_not_called()

    def test_fallback_multiple_blocks(
        self, ocr_config, test_image, mock_paddleocr, mock_easyocr
    ):
        """Test fallback on multiple low-confidence blocks."""
        mock_primary = MagicMock()
        mock_primary.ocr.return_value = [
            [
                [[10, 20], [110, 20], [110, 50], [10, 50]],
                ("Block 1", 0.70),  # Low confidence
            ],
            [
                [[120, 30], [220, 30], [220, 60], [120, 60]],
                ("Block 2", 0.95),  # High confidence
            ],
            [
                [[230, 40], [330, 40], [330, 70], [230, 70]],
                ("Block 3", 0.65),  # Low confidence
            ],
        ]
        mock_paddleocr.return_value = mock_primary

        # Fallback improves both low-confidence blocks
        mock_fallback = MagicMock()
        mock_fallback.readtext.side_effect = [
            [([[10, 20], [110, 20], [110, 50], [10, 50]], "Block 1 Improved", 0.92)],
            [([[230, 40], [330, 40], [330, 70], [230, 70]], "Block 3 Improved", 0.88)],
        ]
        mock_easyocr.Reader.return_value = mock_fallback

        pipeline = OCRPipeline(ocr_config)
        result = pipeline.extract_text(test_image)

        # Check fallback was applied to blocks 1 and 3
        assert result.text_blocks[0].content == "Block 1 Improved"
        assert result.text_blocks[0].ocr_engine == "easyocr"
        assert result.text_blocks[1].content == "Block 2"
        assert result.text_blocks[1].ocr_engine == "paddleocr"
        assert result.text_blocks[2].content == "Block 3 Improved"
        assert result.text_blocks[2].ocr_engine == "easyocr"


class TestLayoutAnalysis:
    """Test layout region detection."""

    def test_detect_title_block(self, ocr_config, test_image, mock_paddleocr):
        """Test title block detection."""
        mock_engine = MagicMock()
        # Title block typically in bottom-right corner
        mock_engine.ocr.return_value = [
            [
                [[300, 150], [390, 150], [390, 190], [300, 190]],  # Bottom-right
                ("PART NO: ABC-123", 0.95),
            ]
        ]
        mock_paddleocr.return_value = mock_engine

        pipeline = OCRPipeline(ocr_config)
        result = pipeline.extract_text(test_image)

        # Should classify as title block region
        assert len(result.layout_regions) > 0
        title_blocks = [
            r for r in result.layout_regions if r.region_type == "title_block"
        ]
        assert len(title_blocks) > 0

    def test_detect_table_region(self, ocr_config, test_image, mock_paddleocr):
        """Test table region detection."""
        mock_engine = MagicMock()
        # Table-like structure with aligned text
        mock_engine.ocr.return_value = [
            [[[50, 50], [100, 50], [100, 70], [50, 70]], ("Col1", 0.90)],
            [[[110, 50], [160, 50], [160, 70], [110, 70]], ("Col2", 0.90)],
            [[[50, 80], [100, 80], [100, 100], [50, 100]], ("Data1", 0.90)],
            [[[110, 80], [160, 80], [160, 100], [110, 100]], ("Data2", 0.90)],
        ]
        mock_paddleocr.return_value = mock_engine

        pipeline = OCRPipeline(ocr_config)
        result = pipeline.extract_text(test_image)

        # Should detect table region
        table_regions = [r for r in result.layout_regions if r.region_type == "table"]
        assert len(table_regions) > 0

    def test_detect_text_region(self, ocr_config, test_image, mock_paddleocr):
        """Test general text region detection."""
        mock_engine = MagicMock()
        mock_engine.ocr.return_value = [
            [[[50, 50], [200, 50], [200, 70], [50, 70]], ("General text content", 0.90)]
        ]
        mock_paddleocr.return_value = mock_engine

        pipeline = OCRPipeline(ocr_config)
        result = pipeline.extract_text(test_image)

        # Should have layout regions
        assert len(result.layout_regions) > 0

    def test_classify_text_block_types(self, ocr_config, test_image, mock_paddleocr):
        """Test classification of text block types."""
        mock_engine = MagicMock()
        mock_engine.ocr.return_value = [
            [
                [[50, 50], [150, 50], [150, 70], [50, 70]],
                ("Ø25.4mm", 0.90),  # Dimension
            ],
            [[[160, 60], [200, 60], [200, 75], [160, 75]], ("Bolt", 0.90)],  # Label
        ]
        mock_paddleocr.return_value = mock_engine

        pipeline = OCRPipeline(ocr_config)
        result = pipeline.extract_text(test_image)

        # Check blocks have region_type classification
        dimension_blocks = [
            b for b in result.text_blocks if b.region_type == "dimension"
        ]
        assert len(dimension_blocks) > 0


class TestLanguageDetection:
    """Test language detection functionality."""

    def test_detect_english(self, ocr_config, test_image, mock_paddleocr):
        """Test English language detection."""
        mock_engine = MagicMock()
        mock_engine.ocr.return_value = [
            [[[50, 50], [150, 50], [150, 70], [50, 70]], ("English text here", 0.90)]
        ]
        mock_paddleocr.return_value = mock_engine

        pipeline = OCRPipeline(ocr_config)
        result = pipeline.extract_text(test_image, language="auto")

        assert result.language == "en"

    def test_explicit_language(self, ocr_config, test_image, mock_paddleocr):
        """Test extraction with explicit language."""
        mock_engine = MagicMock()
        mock_engine.ocr.return_value = [
            [[[50, 50], [150, 50], [150, 70], [50, 70]], ("Text", 0.90)]
        ]
        mock_paddleocr.return_value = mock_engine

        pipeline = OCRPipeline(ocr_config)
        result = pipeline.extract_text(test_image, language="en")

        assert result.language == "en"

    def test_multilingual_support(self, mock_paddleocr):
        """Test multilingual OCR configuration."""
        config = OCRConfig(languages=["en", "es", "fr"])
        pipeline = OCRPipeline(config)

        assert "en" in pipeline.config.languages
        assert "es" in pipeline.config.languages


class TestConfidenceCalculation:
    """Test confidence scoring."""

    def test_average_confidence_calculation(
        self, ocr_config, test_image, mock_paddleocr
    ):
        """Test average confidence calculation."""
        mock_engine = MagicMock()
        mock_engine.ocr.return_value = [
            [[[10, 20], [110, 20], [110, 50], [10, 50]], ("Text 1", 0.90)],
            [[[120, 30], [220, 30], [220, 60], [120, 60]], ("Text 2", 0.80)],
            [[[230, 40], [330, 40], [330, 70], [230, 70]], ("Text 3", 0.85)],
        ]
        mock_paddleocr.return_value = mock_engine

        pipeline = OCRPipeline(ocr_config)
        result = pipeline.extract_text(test_image)

        expected_avg = (0.90 + 0.80 + 0.85) / 3
        assert abs(result.average_confidence - expected_avg) < 0.01

    def test_confidence_threshold_filtering(
        self, ocr_config, test_image, mock_paddleocr, mock_easyocr
    ):
        """Test confidence threshold determines fallback."""
        # Set low threshold
        config = OCRConfig(confidence_threshold=0.70)

        mock_primary = MagicMock()
        mock_primary.ocr.return_value = [
            [
                [[10, 20], [110, 20], [110, 50], [10, 50]],
                ("Text", 0.75),  # Above 0.70, should NOT trigger fallback
            ]
        ]
        mock_paddleocr.return_value = mock_primary

        pipeline = OCRPipeline(config)
        result = pipeline.extract_text(test_image)

        # Should NOT have triggered fallback
        assert result.text_blocks[0].ocr_engine == "paddleocr"


class TestErrorHandling:
    """Test error handling and recovery."""

    def test_primary_engine_failure(
        self, ocr_config, test_image, mock_paddleocr, mock_easyocr
    ):
        """Test fallback when primary engine fails."""
        mock_primary = MagicMock()
        mock_primary.ocr.side_effect = Exception("Primary OCR failed")
        mock_paddleocr.return_value = mock_primary

        # Fallback should handle it
        mock_fallback = MagicMock()
        mock_fallback.readtext.return_value = [
            ([[10, 20], [110, 20], [110, 50], [10, 50]], "Fallback text", 0.85)
        ]
        mock_easyocr.Reader.return_value = mock_fallback

        pipeline = OCRPipeline(ocr_config)
        result = pipeline.extract_text(test_image)

        # Should have results from fallback
        assert len(result.text_blocks) > 0
        assert result.text_blocks[0].ocr_engine == "easyocr"

    def test_both_engines_fail(
        self, ocr_config, test_image, mock_paddleocr, mock_easyocr
    ):
        """Test handling when both engines fail."""
        mock_primary = MagicMock()
        mock_primary.ocr.side_effect = Exception("Primary failed")
        mock_paddleocr.return_value = mock_primary

        mock_fallback = MagicMock()
        mock_fallback.readtext.side_effect = Exception("Fallback failed")
        mock_easyocr.Reader.return_value = mock_fallback

        pipeline = OCRPipeline(ocr_config)

        with pytest.raises(OCRError, match="OCR extraction failed"):
            pipeline.extract_text(test_image)

    def test_invalid_image_input(self, ocr_config, mock_paddleocr):
        """Test handling of invalid image."""
        mock_engine = MagicMock()
        mock_paddleocr.return_value = mock_engine

        pipeline = OCRPipeline(ocr_config)

        with pytest.raises((OCRError, ValueError)):
            pipeline.extract_text(None)

    def test_empty_image(self, ocr_config, mock_paddleocr):
        """Test handling of empty image."""
        mock_engine = MagicMock()
        mock_engine.ocr.return_value = [[]]
        mock_paddleocr.return_value = mock_engine

        empty_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        pipeline = OCRPipeline(ocr_config)
        result = pipeline.extract_text(empty_image)

        assert len(result.text_blocks) == 0


class TestTextBlockCreation:
    """Test text block creation and properties."""

    def test_text_block_has_bbox(self, ocr_config, test_image, mock_paddleocr):
        """Test text blocks have bounding boxes."""
        mock_engine = MagicMock()
        mock_engine.ocr.return_value = [
            [[[10, 20], [110, 20], [110, 50], [10, 50]], ("Text", 0.90)]
        ]
        mock_paddleocr.return_value = mock_engine

        pipeline = OCRPipeline(ocr_config)
        result = pipeline.extract_text(test_image)

        text_block = result.text_blocks[0]
        assert isinstance(text_block.bbox, BoundingBox)
        assert text_block.bbox.x == 10
        assert text_block.bbox.y == 20
        assert text_block.bbox.width == 100
        assert text_block.bbox.height == 30

    def test_text_block_has_id(self, ocr_config, test_image, mock_paddleocr):
        """Test text blocks have unique IDs."""
        mock_engine = MagicMock()
        mock_engine.ocr.return_value = [
            [[[10, 20], [110, 20], [110, 50], [10, 50]], ("Text 1", 0.90)],
            [[[120, 30], [220, 30], [220, 60], [120, 60]], ("Text 2", 0.90)],
        ]
        mock_paddleocr.return_value = mock_engine

        pipeline = OCRPipeline(ocr_config)
        result = pipeline.extract_text(test_image)

        ids = [block.text_id for block in result.text_blocks]
        assert len(ids) == len(set(ids))  # All unique


class TestFullTextGeneration:
    """Test full text concatenation."""

    def test_full_text_generation(self, ocr_config, test_image, mock_paddleocr):
        """Test full text is concatenated correctly."""
        mock_engine = MagicMock()
        mock_engine.ocr.return_value = [
            [[[10, 20], [110, 20], [110, 50], [10, 50]], ("Part Number:", 0.90)],
            [[[120, 25], [220, 25], [220, 55], [120, 55]], ("ABC-123", 0.90)],
        ]
        mock_paddleocr.return_value = mock_engine

        pipeline = OCRPipeline(ocr_config)
        result = pipeline.extract_text(test_image)

        assert "Part Number:" in result.full_text
        assert "ABC-123" in result.full_text

    def test_full_text_preserves_order(self, ocr_config, test_image, mock_paddleocr):
        """Test full text preserves reading order."""
        mock_engine = MagicMock()
        mock_engine.ocr.return_value = [
            [[[10, 10], [110, 10], [110, 30], [10, 30]], ("First", 0.90)],
            [[[10, 40], [110, 40], [110, 60], [10, 60]], ("Second", 0.90)],
            [[[10, 70], [110, 70], [110, 90], [10, 90]], ("Third", 0.90)],
        ]
        mock_paddleocr.return_value = mock_engine

        pipeline = OCRPipeline(ocr_config)
        result = pipeline.extract_text(test_image)

        # Should appear in order
        assert result.full_text.index("First") < result.full_text.index("Second")
        assert result.full_text.index("Second") < result.full_text.index("Third")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
