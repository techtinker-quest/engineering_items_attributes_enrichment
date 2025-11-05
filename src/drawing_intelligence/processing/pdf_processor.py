"""
PDF processing module for the Drawing Intelligence System.

Extracts pages as images and embedded text from PDF files.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import fitz  # PyMuPDF

from ..models.data_structures import PDFPage
from ..utils.error_handlers import PDFProcessingError
from ..utils.validation_utils import validate_pdf_file, validate_file_size

logger = logging.getLogger(__name__)


@dataclass
class PDFConfig:
    """
    Configuration for PDF processing.

    Attributes:
        dpi: Resolution for page rendering (default: 300)
        max_file_size_mb: Maximum PDF file size (default: 50)
        max_pages: Maximum pages to process (default: 20)
        convert_to_grayscale: Convert images to grayscale (default: True)
    """

    dpi: int = 300
    max_file_size_mb: int = 50
    max_pages: int = 20
    convert_to_grayscale: bool = True


class PDFProcessor:
    """
    Extract pages and embedded text from PDF files.

    Uses PyMuPDF (fitz) for high-performance PDF processing.
    """

    def __init__(self, config: PDFConfig):
        """
        Initialize PDF processor.

        Args:
            config: PDF processing configuration
        """
        self.config = config
        logger.info(f"PDFProcessor initialized (DPI: {config.dpi})")

    def extract_pages(self, pdf_path: str) -> List[PDFPage]:
        """
        Extract all pages from PDF as images with embedded text.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of PDFPage objects

        Raises:
            PDFProcessingError: If extraction fails
        """
        # Validate PDF file
        is_valid, error_msg = validate_pdf_file(pdf_path)
        if not is_valid:
            raise PDFProcessingError(error_msg)

        # Validate file size
        is_valid, error_msg = validate_file_size(pdf_path, self.config.max_file_size_mb)
        if not is_valid:
            raise PDFProcessingError(error_msg)

        try:
            doc = fitz.open(pdf_path)

            # Check page count
            if len(doc) > self.config.max_pages:
                logger.warning(
                    f"PDF has {len(doc)} pages, processing only first {self.config.max_pages}"
                )

            pages = []
            num_pages = min(len(doc), self.config.max_pages)

            for page_num in range(num_pages):
                try:
                    page = doc[page_num]

                    # Render page to image
                    image = self._render_page_to_image(page)

                    # Extract text blocks with coordinates
                    text_blocks = self._extract_text_blocks(page)

                    # Get page dimensions
                    pix = page.get_pixmap(
                        matrix=fitz.Matrix(self.config.dpi / 72, self.config.dpi / 72)
                    )
                    dimensions = (pix.width, pix.height)

                    # Create PDFPage object
                    pdf_page = PDFPage(
                        page_number=page_num,
                        image=image,
                        dimensions=dimensions,
                        embedded_text_blocks=text_blocks,
                        dpi=self.config.dpi,
                        metadata={
                            "rotation": page.rotation,
                            "mediabox": page.rect,
                        },
                    )

                    pages.append(pdf_page)
                    logger.debug(f"Extracted page {page_num + 1}/{num_pages}")

                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")
                    raise PDFProcessingError(
                        f"Failed to process page {page_num}: {e}", page_number=page_num
                    )

            doc.close()
            logger.info(f"Successfully extracted {len(pages)} pages from {pdf_path}")
            return pages

        except fitz.fitz.FileDataError as e:
            raise PDFProcessingError(f"Invalid or corrupted PDF file: {e}")
        except Exception as e:
            raise PDFProcessingError(f"Failed to process PDF: {e}")

    def _render_page_to_image(self, page: fitz.Page) -> np.ndarray:
        """
        Render PDF page to numpy image array.

        Args:
            page: PyMuPDF page object

        Returns:
            Numpy array (BGR format if color, grayscale if configured)
        """
        # Calculate zoom factor for desired DPI
        zoom = self.config.dpi / 72  # 72 DPI is default
        matrix = fitz.Matrix(zoom, zoom)

        # Render page to pixmap
        pix = page.get_pixmap(matrix=matrix)

        # Convert pixmap to numpy array
        # PyMuPDF returns RGB format
        img_data = pix.samples
        width = pix.width
        height = pix.height
        channels = pix.n  # Number of color channels

        # Reshape to image array
        if channels == 1:
            # Grayscale
            image = np.frombuffer(img_data, dtype=np.uint8).reshape(height, width)
        elif channels == 3:
            # RGB
            image = np.frombuffer(img_data, dtype=np.uint8).reshape(height, width, 3)
            # Convert RGB to BGR for OpenCV compatibility
            import cv2

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif channels == 4:
            # RGBA
            image = np.frombuffer(img_data, dtype=np.uint8).reshape(height, width, 4)
            import cv2

            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            raise PDFProcessingError(f"Unsupported color format: {channels} channels")

        # Convert to grayscale if configured
        if self.config.convert_to_grayscale and len(image.shape) == 3:
            import cv2

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image

    def _extract_text_blocks(self, page: fitz.Page) -> List[dict]:
        """
        Extract text blocks with bounding box coordinates from page.

        Args:
            page: PyMuPDF page object

        Returns:
            List of text blocks with 'text' and 'bbox' keys
        """
        text_blocks = []

        # Get text blocks with coordinates
        # blocks format: (x0, y0, x1, y1, "text", block_no, block_type)
        blocks = page.get_text("blocks")

        for block in blocks:
            x0, y0, x1, y1, text = block[:5]

            # Skip empty text
            text = text.strip()
            if not text:
                continue

            # Convert PDF coordinates to pixel coordinates
            bbox_pixels = self.pdf_bbox_to_pixel((x0, y0, x1, y1), self.config.dpi)

            text_blocks.append(
                {"text": text, "bbox": bbox_pixels}  # (x0, y0, x1, y1) in pixels
            )

        return text_blocks

    def pdf_bbox_to_pixel(
        self, bbox_pdf: Tuple[float, float, float, float], page_dpi: float
    ) -> Tuple[int, int, int, int]:
        """
        Convert bounding box from PDF points to pixel coordinates.

        PDF uses points (1/72 inch), we need pixels at specified DPI.

        Args:
            bbox_pdf: Bounding box in PDF coordinates (x0, y0, x1, y1)
            page_dpi: Page DPI

        Returns:
            Bounding box in pixel coordinates (x0, y0, x1, y1)
        """
        scale = page_dpi / 72.0
        return (
            int(bbox_pdf[0] * scale),
            int(bbox_pdf[1] * scale),
            int(bbox_pdf[2] * scale),
            int(bbox_pdf[3] * scale),
        )

    def extract_embedded_text(self, pdf_path: str) -> Optional[str]:
        """
        Extract all embedded text from PDF without coordinates.

        Useful for quick text extraction without page rendering.

        Args:
            pdf_path: Path to PDF file

        Returns:
            All text concatenated, or None if no text found
        """
        # Validate PDF file
        is_valid, error_msg = validate_pdf_file(pdf_path)
        if not is_valid:
            raise PDFProcessingError(error_msg)

        try:
            doc = fitz.open(pdf_path)

            full_text = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    full_text.append(text)

            doc.close()

            if full_text:
                return "\n\n".join(full_text)
            return None

        except Exception as e:
            raise PDFProcessingError(f"Failed to extract text from PDF: {e}")

    def get_pdf_metadata(self, pdf_path: str) -> dict:
        """
        Extract metadata from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with metadata (title, author, creator, etc.)
        """
        is_valid, error_msg = validate_pdf_file(pdf_path)
        if not is_valid:
            raise PDFProcessingError(error_msg)

        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata

            result = {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
                "num_pages": len(doc),
                "format": metadata.get("format", ""),
            }

            doc.close()
            return result

        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")
            return {}

    def validate_pdf(self, pdf_path: str) -> Tuple[bool, str]:
        """
        Comprehensive PDF validation.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Basic file validation
        is_valid, error_msg = validate_pdf_file(pdf_path)
        if not is_valid:
            return False, error_msg

        # File size validation
        is_valid, error_msg = validate_file_size(pdf_path, self.config.max_file_size_mb)
        if not is_valid:
            return False, error_msg

        # Try to open and verify
        try:
            doc = fitz.open(pdf_path)

            # Check if encrypted
            if doc.is_encrypted:
                doc.close()
                return False, "PDF is encrypted/password protected"

            # Check page count
            if len(doc) == 0:
                doc.close()
                return False, "PDF has no pages"

            if len(doc) > self.config.max_pages:
                logger.warning(
                    f"PDF has {len(doc)} pages, exceeds max of {self.config.max_pages}"
                )

            # Try to render first page as test
            page = doc[0]
            pix = page.get_pixmap()
            if pix.width == 0 or pix.height == 0:
                doc.close()
                return False, "PDF page has invalid dimensions"

            doc.close()
            return True, ""

        except fitz.fitz.FileDataError:
            return False, "Corrupted or invalid PDF file"
        except Exception as e:
            return False, f"PDF validation failed: {e}"
