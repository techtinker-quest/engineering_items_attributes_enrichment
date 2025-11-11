"""
Unit tests for pdf_processor module.

Tests PDF extraction, text block extraction, and coordinate conversion.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.drawing_intelligence.processing.pdf_processor import PDFProcessor, PDFConfig
from src.drawing_intelligence.models.data_structures import PDFPage, TextBlock
from src.drawing_intelligence.utils.error_handlers import PDFProcessingError


@pytest.fixture
def pdf_config():
    """Create test PDF configuration."""
    return PDFConfig(
        dpi=300, max_file_size_mb=50, max_pages=20, convert_to_grayscale=True
    )


@pytest.fixture
def pdf_processor(pdf_config):
    """Create PDF processor instance."""
    return PDFProcessor(pdf_config)


class TestPDFProcessorInitialization:
    """Test PDF processor initialization."""

    def test_init_with_config(self, pdf_config):
        """Test initialization with configuration."""
        processor = PDFProcessor(pdf_config)
        assert processor.config == pdf_config
        assert processor.config.dpi == 300

    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        processor = PDFProcessor()
        assert processor.config is not None
        assert processor.config.dpi == 300  # Default value


class TestPDFValidation:
    """Test PDF file validation."""

    def test_validate_valid_pdf(self, pdf_processor, tmp_path):
        """Test validation of valid PDF."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nfake content")

        is_valid, message = pdf_processor.validate_pdf(str(pdf_path))
        # Should pass file existence check at minimum
        assert isinstance(is_valid, bool)

    def test_validate_nonexistent_pdf(self, pdf_processor):
        """Test validation of non-existent file."""
        is_valid, message = pdf_processor.validate_pdf("nonexistent.pdf")
        assert is_valid is False
        assert "not found" in message.lower() or "exist" in message.lower()

    def test_validate_too_large_pdf(self, pdf_processor, tmp_path):
        """Test validation of oversized PDF."""
        pdf_path = tmp_path / "large.pdf"
        # Create file larger than max size
        large_content = b"%PDF-1.4\n" + b"x" * (51 * 1024 * 1024)
        pdf_path.write_bytes(large_content)

        is_valid, message = pdf_processor.validate_pdf(str(pdf_path))
        assert is_valid is False
        assert "size" in message.lower()

    def test_validate_empty_pdf(self, pdf_processor, tmp_path):
        """Test validation of empty file."""
        pdf_path = tmp_path / "empty.pdf"
        pdf_path.write_bytes(b"")

        is_valid, message = pdf_processor.validate_pdf(str(pdf_path))
        assert is_valid is False
        assert "empty" in message.lower()


class TestPDFExtraction:
    """Test PDF page extraction."""

    @patch("src.drawing_intelligence.processing.pdf_processor.fitz")
    def test_extract_pages_success(self, mock_fitz, pdf_processor, tmp_path):
        """Test successful page extraction."""
        # Create mock PDF
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nfake content")

        # Mock fitz document
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_pixmap.return_value.samples = np.zeros(
            (100, 100, 3), dtype=np.uint8
        ).tobytes()
        mock_page.get_pixmap.return_value.width = 100
        mock_page.get_pixmap.return_value.height = 100
        mock_page.get_pixmap.return_value.n = 3
        mock_page.get_text.return_value = "Sample text"

        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz.open.return_value = mock_doc

        # Extract pages
        pages = pdf_processor.extract_pages(str(pdf_path))

        assert len(pages) == 1
        assert isinstance(pages[0], PDFPage)
        assert pages[0].page_number == 0
        assert isinstance(pages[0].image, np.ndarray)

    @patch("src.drawing_intelligence.processing.pdf_processor.fitz")
    def test_extract_pages_multiple_pages(self, mock_fitz, pdf_processor, tmp_path):
        """Test extraction of multiple pages."""
        pdf_path = tmp_path / "multi.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nfake content")

        # Mock 3-page document
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 3

        # Create mock pages
        pages = []
        for i in range(3):
            mock_page = MagicMock()
            mock_page.get_pixmap.return_value.samples = np.zeros(
                (100, 100, 3), dtype=np.uint8
            ).tobytes()
            mock_page.get_pixmap.return_value.width = 100
            mock_page.get_pixmap.return_value.height = 100
            mock_page.get_pixmap.return_value.n = 3
            pages.append(mock_page)

        mock_doc.__getitem__ = lambda self, i: pages[i]
        mock_fitz.open.return_value = mock_doc

        # Extract
        result = pdf_processor.extract_pages(str(pdf_path))

        assert len(result) == 3
        assert all(isinstance(p, PDFPage) for p in result)

    def test_extract_pages_too_many_pages(self, pdf_processor, tmp_path):
        """Test handling of PDF with too many pages."""
        # This would require creating actual PDF or advanced mocking
        # Simplified test
        with patch(
            "src.drawing_intelligence.processing.pdf_processor.fitz"
        ) as mock_fitz:
            mock_doc = MagicMock()
            mock_doc.__len__.return_value = 25  # Exceeds max_pages=20
            mock_fitz.open.return_value = mock_doc

            pdf_path = tmp_path / "large.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nfake")

            with pytest.raises(PDFProcessingError, match="too many pages"):
                pdf_processor.extract_pages(str(pdf_path))

    def test_extract_pages_encrypted(self, pdf_processor, tmp_path):
        """Test handling of encrypted PDF."""
        with patch(
            "src.drawing_intelligence.processing.pdf_processor.fitz"
        ) as mock_fitz:
            mock_doc = MagicMock()
            mock_doc.is_encrypted = True
            mock_fitz.open.return_value = mock_doc

            pdf_path = tmp_path / "encrypted.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nencrypted")

            with pytest.raises(PDFProcessingError, match="encrypted"):
                pdf_processor.extract_pages(str(pdf_path))


class TestTextBlockExtraction:
    """Test embedded text extraction."""

    @patch("src.drawing_intelligence.processing.pdf_processor.fitz")
    def test_extract_embedded_text_success(self, mock_fitz, pdf_processor, tmp_path):
        """Test successful embedded text extraction."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nfake")

        # Mock page with text blocks
        mock_page = MagicMock()
        mock_page.get_text.return_value = {
            "blocks": [
                {"bbox": (10, 20, 110, 50), "text": "Part Number: ABC-123", "type": 0},
                {"bbox": (120, 30, 220, 60), "text": "Material: Steel", "type": 0},
            ]
        }

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz.open.return_value = mock_doc

        # Extract
        text_blocks = pdf_processor.extract_text_blocks(str(pdf_path), page_num=0)

        assert len(text_blocks) == 2
        assert all(isinstance(tb, TextBlock) for tb in text_blocks)
        assert text_blocks[0].content == "Part Number: ABC-123"

    @patch("src.drawing_intelligence.processing.pdf_processor.fitz")
    def test_extract_embedded_text_empty_page(self, mock_fitz, pdf_processor, tmp_path):
        """Test extraction from page with no text."""
        pdf_path = tmp_path / "empty_text.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nfake")

        mock_page = MagicMock()
        mock_page.get_text.return_value = {"blocks": []}

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz.open.return_value = mock_doc

        text_blocks = pdf_processor.extract_text_blocks(str(pdf_path), page_num=0)

        assert len(text_blocks) == 0


class TestCoordinateConversion:
    """Test PDF to pixel coordinate conversion."""

    def test_pdf_bbox_to_pixel(self, pdf_processor):
        """Test PDF coordinate to pixel conversion."""
        # PDF uses points (72 DPI), convert to 300 DPI pixels
        pdf_bbox = (72, 72, 144, 144)  # 1 inch square at 1 inch offset
        page_height = 792  # 11 inches at 72 DPI

        pixel_bbox = pdf_processor.pdf_bbox_to_pixel(
            pdf_bbox, page_height=page_height, dpi=300
        )

        # 1 inch = 300 pixels at 300 DPI
        assert pixel_bbox.x == pytest.approx(300, abs=5)
        assert pixel_bbox.y == pytest.approx(300, abs=5)
        assert pixel_bbox.width == pytest.approx(300, abs=5)
        assert pixel_bbox.height == pytest.approx(300, abs=5)

    def test_coordinate_scaling(self, pdf_processor):
        """Test coordinate scaling with different DPI."""
        pdf_bbox = (0, 0, 72, 72)
        page_height = 792

        # At 150 DPI
        bbox_150 = pdf_processor.pdf_bbox_to_pixel(pdf_bbox, page_height, dpi=150)
        # At 300 DPI
        bbox_300 = pdf_processor.pdf_bbox_to_pixel(pdf_bbox, page_height, dpi=300)

        # 300 DPI should be 2x the size of 150 DPI
        assert bbox_300.width == pytest.approx(bbox_150.width * 2, abs=2)


class TestGetPDFMetadata:
    """Test PDF metadata extraction."""

    @patch("src.drawing_intelligence.processing.pdf_processor.fitz")
    def test_get_metadata_success(self, mock_fitz, pdf_processor, tmp_path):
        """Test successful metadata extraction."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nfake")

        mock_doc = MagicMock()
        mock_doc.metadata = {
            "title": "Test Drawing",
            "author": "Engineer Name",
            "creator": "CAD Software",
            "creationDate": "D:20251104120000",
            "subject": "Mechanical Assembly",
        }
        mock_doc.__len__.return_value = 5
        mock_fitz.open.return_value = mock_doc

        metadata = pdf_processor.get_pdf_metadata(str(pdf_path))

        assert metadata["title"] == "Test Drawing"
        assert metadata["author"] == "Engineer Name"
        assert metadata["num_pages"] == 5

    @patch("src.drawing_intelligence.processing.pdf_processor.fitz")
    def test_get_metadata_missing_fields(self, mock_fitz, pdf_processor, tmp_path):
        """Test metadata extraction with missing fields."""
        pdf_path = tmp_path / "minimal.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nfake")

        mock_doc = MagicMock()
        mock_doc.metadata = {}  # No metadata
        mock_doc.__len__.return_value = 1
        mock_fitz.open.return_value = mock_doc

        metadata = pdf_processor.get_pdf_metadata(str(pdf_path))

        assert metadata["num_pages"] == 1
        # Should handle missing fields gracefully
        assert "title" in metadata  # May be None or empty


class TestErrorHandling:
    """Test error handling."""

    def test_extract_invalid_file(self, pdf_processor):
        """Test extraction from invalid file."""
        with pytest.raises((PDFProcessingError, FileNotFoundError)):
            pdf_processor.extract_pages("nonexistent.pdf")

    @patch("src.drawing_intelligence.processing.pdf_processor.fitz")
    def test_extract_corrupted_pdf(self, mock_fitz, pdf_processor, tmp_path):
        """Test handling of corrupted PDF."""
        pdf_path = tmp_path / "corrupted.pdf"
        pdf_path.write_bytes(b"corrupted data")

        mock_fitz.open.side_effect = Exception("Corrupted PDF")

        with pytest.raises(PDFProcessingError, match="Failed to open PDF"):
            pdf_processor.extract_pages(str(pdf_path))

    @patch("src.drawing_intelligence.processing.pdf_processor.fitz")
    def test_extract_render_failure(self, mock_fitz, pdf_processor, tmp_path):
        """Test handling of page render failure."""
        pdf_path = tmp_path / "render_fail.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nfake")

        mock_page = MagicMock()
        mock_page.get_pixmap.side_effect = Exception("Render failed")

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz.open.return_value = mock_doc

        with pytest.raises(PDFProcessingError, match="Failed to render"):
            pdf_processor.extract_pages(str(pdf_path))


class TestConfigurationOptions:
    """Test different configuration options."""

    def test_different_dpi_settings(self):
        """Test processing with different DPI settings."""
        config_150 = PDFConfig(dpi=150)
        config_300 = PDFConfig(dpi=300)

        processor_150 = PDFProcessor(config_150)
        processor_300 = PDFProcessor(config_300)

        assert processor_150.config.dpi == 150
        assert processor_300.config.dpi == 300

    def test_grayscale_conversion(self):
        """Test grayscale conversion setting."""
        config_gray = PDFConfig(convert_to_grayscale=True)
        config_color = PDFConfig(convert_to_grayscale=False)

        processor_gray = PDFProcessor(config_gray)
        processor_color = PDFProcessor(config_color)

        assert processor_gray.config.convert_to_grayscale is True
        assert processor_color.config.convert_to_grayscale is False


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
