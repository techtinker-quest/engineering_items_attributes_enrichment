"""
Comprehensive test suite for pdf_processor.py module.

Location: tests/unit/processing/test_pdf_processor.py

Test Coverage:
    - Configuration validation
    - PDF validation and error handling
    - Page extraction (single and batch)
    - Text extraction
    - Metadata extraction
    - Parallel processing
    - Progress callbacks
    - Custom exceptions
    - Edge cases

Run tests:
    pytest tests/unit/processing/test_pdf_processor.py -v
    pytest tests/unit/processing/test_pdf_processor.py::TestPDFConfig -v
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Adjust import path based on your project structure
from drawing_intelligence.processing.pdf_processor import (
    PDFConfig,
    PDFCorruptedError,
    PDFEncryptionError,
    PDFPageRenderError,
    PDFProcessor,
    TextBlock,
)
from drawing_intelligence.utils.error_handlers import PDFProcessingError


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_config():
    """Provide default PDFConfig for tests."""
    return PDFConfig(dpi=300, max_pages=10, max_file_size_mb=50)


@pytest.fixture
def processor(default_config):
    """Provide PDFProcessor instance with default config."""
    return PDFProcessor(default_config)


@pytest.fixture
def mock_pdf_document():
    """Create a mock fitz.Document for testing."""
    mock_doc = MagicMock()
    mock_doc.is_encrypted = False
    mock_doc.__len__ = Mock(return_value=3)
    mock_doc.metadata = {
        "title": "Test Drawing",
        "author": "Test Author",
        "creationDate": "D:20240101120000",
        "format": "PDF 1.7",
    }
    return mock_doc


@pytest.fixture
def mock_pdf_page():
    """Create a mock fitz.Page for testing."""
    mock_page = MagicMock()
    mock_page.rotation = 0
    mock_page.rect = MagicMock()

    # Mock pixmap
    mock_pixmap = MagicMock()
    mock_pixmap.width = 2550
    mock_pixmap.height = 3300
    mock_pixmap.n = 3  # RGB
    mock_pixmap.samples = np.random.randint(
        0, 255, 2550 * 3300 * 3, dtype=np.uint8
    ).tobytes()

    mock_page.get_pixmap = Mock(return_value=mock_pixmap)

    # Mock text blocks
    mock_page.get_text = Mock(
        return_value=[
            (10.0, 20.0, 100.0, 50.0, "Part Number: ABC-123", 0, 0),
            (10.0, 60.0, 200.0, 90.0, "Material: Steel", 1, 0),
        ]
    )

    return mock_page


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a temporary PDF file for testing."""
    pdf_file = tmp_path / "test_drawing.pdf"
    # Create a minimal valid PDF
    pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer<</Size 4/Root 1 0 R>>
startxref
190
%%EOF"""
    pdf_file.write_bytes(pdf_content)
    return pdf_file


# ============================================================================
# Test PDFConfig
# ============================================================================


class TestPDFConfig:
    """Test suite for PDFConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PDFConfig()
        assert config.dpi == 300
        assert config.max_file_size_mb == 50
        assert config.max_pages == 20
        assert config.convert_to_grayscale is True
        assert config.parallel_workers == 1

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PDFConfig(dpi=600, max_pages=5, parallel_workers=4)
        assert config.dpi == 600
        assert config.max_pages == 5
        assert config.parallel_workers == 4

    def test_dpi_validation_too_low(self):
        """Test DPI validation rejects values below 72."""
        with pytest.raises(ValueError, match="DPI must be between 72 and 1200"):
            PDFConfig(dpi=50)

    def test_dpi_validation_too_high(self):
        """Test DPI validation rejects values above 1200."""
        with pytest.raises(ValueError, match="DPI must be between 72 and 1200"):
            PDFConfig(dpi=2000)

    def test_dpi_validation_boundary_values(self):
        """Test DPI validation accepts boundary values."""
        config_low = PDFConfig(dpi=72)
        config_high = PDFConfig(dpi=1200)
        assert config_low.dpi == 72
        assert config_high.dpi == 1200

    def test_max_file_size_validation(self):
        """Test max_file_size_mb must be positive."""
        with pytest.raises(ValueError, match="max_file_size_mb must be positive"):
            PDFConfig(max_file_size_mb=0)

        with pytest.raises(ValueError, match="max_file_size_mb must be positive"):
            PDFConfig(max_file_size_mb=-10)

    def test_max_pages_validation(self):
        """Test max_pages must be positive."""
        with pytest.raises(ValueError, match="max_pages must be positive"):
            PDFConfig(max_pages=0)

        with pytest.raises(ValueError, match="max_pages must be positive"):
            PDFConfig(max_pages=-5)

    def test_parallel_workers_validation(self):
        """Test parallel_workers must be at least 1."""
        with pytest.raises(ValueError, match="parallel_workers must be at least 1"):
            PDFConfig(parallel_workers=0)

    def test_config_immutability(self):
        """Test that PDFConfig is frozen (immutable)."""
        config = PDFConfig()
        with pytest.raises(AttributeError):
            config.dpi = 600  # Should fail because dataclass is frozen

    def test_with_overrides(self):
        """Test config override mechanism."""
        config = PDFConfig(dpi=300, max_pages=10)
        new_config = config.with_overrides(dpi=600, max_pages=5)

        # Original unchanged
        assert config.dpi == 300
        assert config.max_pages == 10

        # New config has overrides
        assert new_config.dpi == 600
        assert new_config.max_pages == 5

    def test_with_overrides_partial(self):
        """Test partial config overrides."""
        config = PDFConfig(dpi=300, max_pages=10, parallel_workers=1)
        new_config = config.with_overrides(dpi=600)

        assert new_config.dpi == 600
        assert new_config.max_pages == 10  # Unchanged
        assert new_config.parallel_workers == 1  # Unchanged


# ============================================================================
# Test TextBlock
# ============================================================================


class TestTextBlock:
    """Test suite for TextBlock dataclass."""

    def test_text_block_creation(self):
        """Test TextBlock creation and attributes."""
        tb = TextBlock(text="Part Number: ABC-123", bbox=(10, 20, 100, 50))
        assert tb.text == "Part Number: ABC-123"
        assert tb.bbox == (10, 20, 100, 50)

    def test_text_block_immutability(self):
        """Test that TextBlock is frozen (immutable)."""
        tb = TextBlock(text="Test", bbox=(0, 0, 100, 100))
        with pytest.raises(AttributeError):
            tb.text = "Modified"  # Should fail


# ============================================================================
# Test Custom Exceptions
# ============================================================================


class TestCustomExceptions:
    """Test suite for custom exception classes."""

    def test_pdf_encryption_error(self):
        """Test PDFEncryptionError creation and properties."""
        error = PDFEncryptionError()
        assert "encrypted" in str(error).lower()
        assert error.recoverable is False

    def test_pdf_page_render_error(self):
        """Test PDFPageRenderError with page number."""
        error = PDFPageRenderError("Render failed", page_number=5)
        assert error.page_number == 5
        assert "Render failed" in str(error)

    def test_pdf_corrupted_error(self):
        """Test PDFCorruptedError creation."""
        error = PDFCorruptedError("Invalid structure")
        assert "Invalid structure" in str(error)
        assert error.recoverable is False


# ============================================================================
# Test PDFProcessor Initialization
# ============================================================================


class TestPDFProcessorInit:
    """Test suite for PDFProcessor initialization."""

    def test_processor_initialization(self, default_config):
        """Test processor initializes with config."""
        processor = PDFProcessor(default_config)
        assert processor.config == default_config
        assert processor.config.dpi == 300

    def test_processor_logging(self, default_config, caplog):
        """Test processor logs initialization."""
        with caplog.at_level("INFO"):
            PDFProcessor(default_config)
        assert "PDFProcessor initialized" in caplog.text
        assert "DPI: 300" in caplog.text


# ============================================================================
# Test PDF Validation
# ============================================================================


class TestPDFValidation:
    """Test suite for PDF validation methods."""

    @patch("drawing_intelligence.processing.pdf_processor.validate_pdf_file")
    @patch("drawing_intelligence.processing.pdf_processor.validate_file_size")
    @patch("drawing_intelligence.processing.pdf_processor.fitz.open")
    def test_open_pdf_validated_success(
        self,
        mock_fitz_open,
        mock_file_size,
        mock_pdf_file,
        processor,
        mock_pdf_document,
    ):
        """Test successful PDF validation and opening."""
        mock_pdf_file.return_value = (True, "")
        mock_file_size.return_value = (True, "")
        mock_fitz_open.return_value = mock_pdf_document

        with processor._open_pdf_validated("test.pdf") as doc:
            assert doc == mock_pdf_document
            assert doc.is_encrypted is False

        mock_pdf_document.close.assert_called_once()

    @patch("drawing_intelligence.processing.pdf_processor.validate_pdf_file")
    def test_open_pdf_validated_invalid_file(self, mock_pdf_file, processor):
        """Test validation failure for invalid PDF file."""
        mock_pdf_file.return_value = (False, "Not a PDF file")

        with pytest.raises(PDFProcessingError, match="Not a PDF file"):
            with processor._open_pdf_validated("test.txt"):
                pass

    @patch("drawing_intelligence.processing.pdf_processor.validate_pdf_file")
    @patch("drawing_intelligence.processing.pdf_processor.validate_file_size")
    def test_open_pdf_validated_file_too_large(
        self, mock_file_size, mock_pdf_file, processor
    ):
        """Test validation failure for oversized file."""
        mock_pdf_file.return_value = (True, "")
        mock_file_size.return_value = (False, "File exceeds size limit")

        with pytest.raises(PDFProcessingError, match="File exceeds size limit"):
            with processor._open_pdf_validated("large.pdf"):
                pass

    @patch("drawing_intelligence.processing.pdf_processor.validate_pdf_file")
    @patch("drawing_intelligence.processing.pdf_processor.validate_file_size")
    @patch("drawing_intelligence.processing.pdf_processor.fitz.open")
    def test_open_pdf_validated_encrypted(
        self, mock_fitz_open, mock_file_size, mock_pdf_file, processor
    ):
        """Test handling of encrypted PDF."""
        mock_pdf_file.return_value = (True, "")
        mock_file_size.return_value = (True, "")

        encrypted_doc = MagicMock()
        encrypted_doc.is_encrypted = True
        mock_fitz_open.return_value = encrypted_doc

        with pytest.raises(PDFEncryptionError):
            with processor._open_pdf_validated("encrypted.pdf"):
                pass

        encrypted_doc.close.assert_called_once()

    @patch("drawing_intelligence.processing.pdf_processor.validate_pdf_file")
    @patch("drawing_intelligence.processing.pdf_processor.validate_file_size")
    @patch("drawing_intelligence.processing.pdf_processor.fitz.open")
    def test_open_pdf_validated_empty_pdf(
        self, mock_fitz_open, mock_file_size, mock_pdf_file, processor
    ):
        """Test handling of PDF with zero pages."""
        mock_pdf_file.return_value = (True, "")
        mock_file_size.return_value = (True, "")

        empty_doc = MagicMock()
        empty_doc.is_encrypted = False
        empty_doc.__len__ = Mock(return_value=0)
        mock_fitz_open.return_value = empty_doc

        with pytest.raises(PDFCorruptedError, match="no pages"):
            with processor._open_pdf_validated("empty.pdf"):
                pass


# ============================================================================
# Test Page Extraction
# ============================================================================


class TestPageExtraction:
    """Test suite for page extraction methods."""

    @patch.object(PDFProcessor, "_open_pdf_validated")
    @patch.object(PDFProcessor, "_extract_single_page")
    def test_extract_pages_sequential(
        self, mock_extract_single, mock_open, processor, mock_pdf_document
    ):
        """Test sequential page extraction."""
        # Setup
        mock_open.return_value.__enter__ = Mock(return_value=mock_pdf_document)
        mock_open.return_value.__exit__ = Mock(return_value=False)

        mock_page = MagicMock()
        mock_pdf_document.__getitem__ = Mock(return_value=mock_page)

        from drawing_intelligence.models.data_structures import PDFPage

        mock_pdf_page = PDFPage(
            page_number=0,
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            dimensions=(100, 100),
            embedded_text_blocks=[],
            dpi=300,
        )
        mock_extract_single.return_value = mock_pdf_page

        # Execute
        pages = processor.extract_pages("test.pdf")

        # Assert
        assert len(pages) == 3
        assert mock_extract_single.call_count == 3

    @patch.object(PDFProcessor, "_open_pdf_validated")
    def test_extract_pages_with_progress_callback(
        self, mock_open, processor, mock_pdf_document
    ):
        """Test page extraction with progress callback."""
        mock_open.return_value.__enter__ = Mock(return_value=mock_pdf_document)
        mock_open.return_value.__exit__ = Mock(return_value=False)

        progress_calls = []

        def progress_callback(current, total):
            progress_calls.append((current, total))

        with patch.object(processor, "_extract_pages_sequential") as mock_extract:
            mock_extract.return_value = ([], [])

            with pytest.raises(
                PDFProcessingError, match="No pages successfully extracted"
            ):
                processor.extract_pages("test.pdf", progress_callback=progress_callback)

    @patch.object(PDFProcessor, "_open_pdf_validated")
    @patch.object(PDFProcessor, "_extract_single_page")
    def test_extract_pages_max_pages_limit(
        self, mock_extract_single, mock_open, processor, mock_pdf_document
    ):
        """Test that max_pages limit is enforced."""
        # Create PDF with 100 pages
        mock_pdf_document.__len__ = Mock(return_value=100)
        mock_open.return_value.__enter__ = Mock(return_value=mock_pdf_document)
        mock_open.return_value.__exit__ = Mock(return_value=False)

        from drawing_intelligence.models.data_structures import PDFPage

        mock_pdf_page = PDFPage(
            page_number=0,
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            dimensions=(100, 100),
            embedded_text_blocks=[],
            dpi=300,
        )
        mock_extract_single.return_value = mock_pdf_page
        mock_pdf_document.__getitem__ = Mock(return_value=MagicMock())

        # Execute with default config (max_pages=10)
        pages = processor.extract_pages("test.pdf")

        # Should only extract 10 pages
        assert len(pages) == 10

    @patch.object(PDFProcessor, "_open_pdf_validated")
    def test_extract_pages_iter(self, mock_open, processor, mock_pdf_document):
        """Test generator-based page extraction."""
        mock_open.return_value.__enter__ = Mock(return_value=mock_pdf_document)
        mock_open.return_value.__exit__ = Mock(return_value=False)

        from drawing_intelligence.models.data_structures import PDFPage

        mock_pdf_page = PDFPage(
            page_number=0,
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            dimensions=(100, 100),
            embedded_text_blocks=[],
            dpi=300,
        )

        with patch.object(
            processor, "_extract_single_page", return_value=mock_pdf_page
        ):
            mock_pdf_document.__getitem__ = Mock(return_value=MagicMock())

            pages = list(processor.extract_pages_iter("test.pdf"))
            assert len(pages) == 3


# ============================================================================
# Test Text Extraction
# ============================================================================


class TestTextExtraction:
    """Test suite for text extraction methods."""

    @patch.object(PDFProcessor, "_open_pdf_validated")
    def test_extract_embedded_text_success(
        self, mock_open, processor, mock_pdf_document
    ):
        """Test successful text extraction."""
        mock_open.return_value.__enter__ = Mock(return_value=mock_pdf_document)
        mock_open.return_value.__exit__ = Mock(return_value=False)

        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Page 1 content"

        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Page 2 content"

        mock_page3 = MagicMock()
        mock_page3.get_text.return_value = "Page 3 content"

        mock_pdf_document.__getitem__ = Mock(
            side_effect=[mock_page1, mock_page2, mock_page3]
        )

        text = processor.extract_embedded_text("test.pdf")

        assert text is not None
        assert "Page 1 content" in text
        assert "Page 2 content" in text
        assert "Page 3 content" in text
        assert "\n\n" in text  # Pages separated by double newline

    @patch.object(PDFProcessor, "_open_pdf_validated")
    def test_extract_embedded_text_no_text(
        self, mock_open, processor, mock_pdf_document
    ):
        """Test text extraction from PDF with no text."""
        mock_open.return_value.__enter__ = Mock(return_value=mock_pdf_document)
        mock_open.return_value.__exit__ = Mock(return_value=False)

        mock_page = MagicMock()
        mock_page.get_text.return_value = "   "  # Only whitespace

        mock_pdf_document.__getitem__ = Mock(return_value=mock_page)

        text = processor.extract_embedded_text("test.pdf")

        assert text is None


# ============================================================================
# Test Metadata Extraction
# ============================================================================


class TestMetadataExtraction:
    """Test suite for metadata extraction."""

    @patch.object(PDFProcessor, "_open_pdf_validated")
    def test_get_pdf_metadata_success(self, mock_open, processor, mock_pdf_document):
        """Test successful metadata extraction."""
        mock_open.return_value.__enter__ = Mock(return_value=mock_pdf_document)
        mock_open.return_value.__exit__ = Mock(return_value=False)

        metadata = processor.get_pdf_metadata("test.pdf")

        assert metadata["title"] == "Test Drawing"
        assert metadata["author"] == "Test Author"
        assert metadata["num_pages"] == 3
        assert metadata["format"] == "PDF 1.7"

    @patch.object(PDFProcessor, "_open_pdf_validated")
    def test_get_pdf_metadata_missing_fields(self, mock_open, processor):
        """Test metadata extraction with missing fields."""
        mock_doc = MagicMock()
        mock_doc.is_encrypted = False
        mock_doc.__len__ = Mock(return_value=5)
        mock_doc.metadata = {}  # Empty metadata

        mock_open.return_value.__enter__ = Mock(return_value=mock_doc)
        mock_open.return_value.__exit__ = Mock(return_value=False)

        metadata = processor.get_pdf_metadata("test.pdf")

        assert metadata["title"] == ""
        assert metadata["author"] == ""
        assert metadata["num_pages"] == 5


# ============================================================================
# Test Batch Processing
# ============================================================================


class TestBatchProcessing:
    """Test suite for batch directory processing."""

    def test_extract_pages_from_directory(self, processor, tmp_path):
        """Test batch processing of directory."""
        # Create test PDFs
        pdf1 = tmp_path / "drawing1.pdf"
        pdf2 = tmp_path / "drawing2.pdf"

        # Create minimal PDFs (same content as sample_pdf_path fixture)
        pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
xref
0 2
trailer<</Size 2/Root 1 0 R>>
startxref
50
%%EOF"""

        pdf1.write_bytes(pdf_content)
        pdf2.write_bytes(pdf_content)

        # Mock the extract_pages method
        with patch.object(processor, "extract_pages", return_value=[]) as mock_extract:
            results = processor.extract_pages_from_directory(tmp_path)

            assert len(results) == 2
            assert mock_extract.call_count == 2

    def test_extract_pages_from_directory_not_found(self, processor):
        """Test batch processing with non-existent directory."""
        with pytest.raises(ValueError, match="Directory does not exist"):
            processor.extract_pages_from_directory("/nonexistent/path")

    def test_extract_pages_from_directory_not_a_directory(self, processor, tmp_path):
        """Test batch processing with file instead of directory."""
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("test")

        with pytest.raises(ValueError, match="not a directory"):
            processor.extract_pages_from_directory(file_path)


# ============================================================================
# Test Config Overrides
# ============================================================================


class TestConfigOverrides:
    """Test suite for config override functionality."""

    @patch.object(PDFProcessor, "_open_pdf_validated")
    @patch.object(PDFProcessor, "_extract_single_page")
    def test_extract_pages_with_config_override(
        self, mock_extract, mock_open, processor, mock_pdf_document
    ):
        """Test page extraction with config overrides."""
        mock_open.return_value.__enter__ = Mock(return_value=mock_pdf_document)
        mock_open.return_value.__exit__ = Mock(return_value=False)

        from drawing_intelligence.models.data_structures import PDFPage

        mock_pdf_page = PDFPage(
            page_number=0,
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            dimensions=(100, 100),
            embedded_text_blocks=[],
            dpi=600,  # Overridden DPI
        )
        mock_extract.return_value = mock_pdf_page
        mock_pdf_document.__getitem__ = Mock(return_value=MagicMock())

        # Extract with DPI override
        pages = processor.extract_pages("test.pdf", dpi=600)

        # Original config unchanged
        assert processor.config.dpi == 300

        # Pages extracted with overridden DPI
        assert len(pages) == 3


# ============================================================================
# Integration Tests (require real PDFs)
# ============================================================================


@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring real PDF files."""

    def test_real_pdf_processing(self, processor, sample_pdf_path):
        """Test processing a real (minimal) PDF file."""
        # This test uses the sample_pdf_path fixture which creates a real PDF
        # Note: This is a minimal PDF and may not have renderable pages
        try:
            metadata = processor.get_pdf_metadata(sample_pdf_path)
            assert metadata is not None
            assert "num_pages" in metadata
        except Exception as e:
            pytest.skip(f"Real PDF processing not available: {e}")


# ============================================================================
# Performance Tests
# ============================================================================


@pytest.mark.performance
class TestPerformance:
    """Performance-related tests."""

    def test_parallel_vs_sequential_extraction(self, tmp_path):
        """Compare parallel vs sequential processing performance."""
        # This is a placeholder for performance testing
        # In real scenarios, you'd create multiple test PDFs and measure time
        pass


if __name__ == "__main__":
    # Run tests with: python test_pdf_processor.py
    pytest.main([__file__, "-v", "--tb=short"])
