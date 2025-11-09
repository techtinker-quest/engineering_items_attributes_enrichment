"""
PDF processing module for the Drawing Intelligence System.

This module provides functionality to extract pages as images and embedded text
from PDF files using PyMuPDF (fitz). It supports configurable DPI rendering,
page limits, grayscale conversion, and comprehensive validation.

Classes:
    PDFConfig: Configuration dataclass for PDF processing parameters.
    PDFProcessor: Main processor class for PDF extraction operations.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np

from ..models.data_structures import PDFPage
from ..utils.error_handlers import PDFProcessingError
from ..utils.validation_utils import validate_file_size, validate_pdf_file

logger = logging.getLogger(__name__)


@dataclass
class PDFConfig:
    """Configuration for PDF processing operations.

    Controls rendering quality, file size limits, and output format preferences
    for PDF page extraction and text processing.

    Attributes:
        dpi: Resolution for page rendering in dots per inch. Higher values
            produce better quality but increase memory usage. Default is 300 DPI
            (standard for document processing). 600 DPI may be needed for fine
            detail extraction.
        max_file_size_mb: Maximum allowed PDF file size in megabytes. Files
            exceeding this limit are rejected to prevent resource exhaustion.
            Default is 50 MB.
        max_pages: Maximum number of pages to process from a PDF. Prevents
            processing of extremely large documents. Default is 20 pages.
        convert_to_grayscale: If True, convert rendered images to grayscale.
            Reduces memory usage and may improve OCR performance. Default is True.

    Note:
        DPI affects memory usage quadratically (300 DPI uses 4x memory vs 150 DPI).
        Adjust based on available system resources and required detail level.
    """

    dpi: int = 300
    max_file_size_mb: int = 50
    max_pages: int = 20
    convert_to_grayscale: bool = True


class PDFProcessor:
    """Extracts pages and embedded text from PDF files.

    Uses PyMuPDF (fitz) for high-performance PDF processing with configurable
    rendering parameters. Supports page-by-page extraction with embedded text
    block coordinates, metadata extraction, and comprehensive validation.

    The processor handles PDF coordinate system conversion (points to pixels),
    multiple color formats (grayscale, RGB, RGBA), and includes safeguards
    against corrupted or encrypted files.

    Attributes:
        config: PDF processing configuration controlling DPI, limits, and format.

    Example:
        >>> config = PDFConfig(dpi=300, max_pages=10)
        >>> processor = PDFProcessor(config)
        >>> pages = processor.extract_pages("drawing.pdf")
        >>> print(f"Extracted {len(pages)} pages")
    """

    def __init__(self, config: PDFConfig) -> None:
        """Initialize PDF processor with specified configuration.

        Args:
            config: PDF processing configuration object controlling rendering
                parameters, file size limits, and format preferences.
        """
        self.config = config
        logger.info(f"PDFProcessor initialized (DPI: {config.dpi})")

    def extract_pages(self, pdf_path: str) -> List[PDFPage]:
        """Extract all pages from PDF as images with embedded text blocks.

        Renders each page to a numpy image array at the configured DPI and
        extracts embedded text blocks with bounding box coordinates. Validates
        file before processing and enforces page count limits.

        Args:
            pdf_path: Absolute or relative path to the PDF file to process.
                Must be a valid, non-encrypted PDF file within size limits.

        Returns:
            List of PDFPage objects, each containing:
                - Rendered page image as numpy array
                - Page dimensions (width, height) in pixels
                - Embedded text blocks with bounding boxes
                - Page metadata (rotation, media box)

        Raises:
            PDFProcessingError: If file validation fails, PDF is corrupted or
                encrypted, file exceeds size limits, or page rendering fails.
                Error includes page_number attribute if failure occurs during
                page-specific processing.

        Note:
            Processing stops at max_pages limit (configured in PDFConfig).
            Pages beyond this limit are logged but not processed.
        """
        # Validate PDF file
        is_valid, error_msg = validate_pdf_file(pdf_path)
        if not is_valid:
            raise PDFProcessingError(error_msg)

        # Validate file size
        is_valid, error_msg = validate_file_size(pdf_path, self.config.max_file_size_mb)
        if not is_valid:
            raise PDFProcessingError(error_msg)

        doc = None
        try:
            doc = fitz.open(pdf_path)

            # Check page count
            if len(doc) > self.config.max_pages:
                logger.warning(
                    f"PDF has {len(doc)} pages, processing only first "
                    f"{self.config.max_pages}"
                )

            pages = []
            num_pages = min(len(doc), self.config.max_pages)

            for page_num in range(num_pages):
                try:
                    page = doc[page_num]

                    # Render page to image and get dimensions in one call
                    image, dimensions = self._render_page_to_image(page)

                    # Extract text blocks with coordinates
                    text_blocks = self._extract_text_blocks(page)

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
                    logger.error(f"Error processing page {page_num + 1}: {e}")
                    raise PDFProcessingError(
                        f"Failed to process page {page_num + 1}: {e}",
                        page_number=page_num,
                    )

            logger.info(f"Successfully extracted {len(pages)} pages from {pdf_path}")
            return pages

        except fitz.FileDataError as e:
            raise PDFProcessingError(f"Invalid or corrupted PDF file: {e}")
        except Exception as e:
            raise PDFProcessingError(f"Failed to process PDF: {e}")
        finally:
            if doc is not None:
                doc.close()

    def _render_page_to_image(
        self, page: fitz.Page
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Render PDF page to numpy image array at configured DPI.

        Converts a PyMuPDF page object to a numpy array suitable for image
        processing. Handles multiple color formats (grayscale, RGB, RGBA) and
        converts to BGR format for OpenCV compatibility when needed. Optionally
        converts to grayscale based on configuration.

        Args:
            page: PyMuPDF page object to render. Must be a valid page from an
                open fitz.Document.

        Returns:
            Tuple containing:
                - Numpy array with rendered page image. Shape depends on
                  configuration:
                    * Grayscale: (height, width) with dtype uint8
                    * Color: (height, width, 3) BGR format with dtype uint8
                - Tuple of (width, height) in pixels at configured DPI

        Raises:
            PDFProcessingError: If page has unsupported color format (not 1, 3,
                or 4 channels) or rendering fails.

        Note:
            PDF default resolution is 72 DPI. This method scales rendering
            according to config.dpi using zoom matrix transformation.
        """
        # Calculate zoom factor for desired DPI
        zoom = self.config.dpi / 72  # 72 DPI is default
        matrix = fitz.Matrix(zoom, zoom)

        # Render page to pixmap
        pix = page.get_pixmap(matrix=matrix)

        # Store dimensions before converting to array
        dimensions = (pix.width, pix.height)

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
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif channels == 4:
            # RGBA
            image = np.frombuffer(img_data, dtype=np.uint8).reshape(height, width, 4)
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            raise PDFProcessingError(f"Unsupported color format: {channels} channels")

        # Convert to grayscale if configured
        if self.config.convert_to_grayscale and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image, dimensions

    def _extract_text_blocks(self, page: fitz.Page) -> List[dict]:
        """Extract text blocks with bounding box coordinates from page.

        Retrieves all text blocks from the page along with their spatial
        positions. Empty text blocks are filtered out. Coordinates are
        converted from PDF points (72 DPI) to pixels at the configured DPI.

        Args:
            page: PyMuPDF page object from which to extract text. Must be a
                valid page from an open fitz.Document.

        Returns:
            List of dictionaries, each containing:
                - 'text': Extracted text content (whitespace stripped)
                - 'bbox': Bounding box tuple (x0, y0, x1, y1) in pixel
                  coordinates at configured DPI

        Note:
            Only blocks with non-empty text are returned. Bounding boxes
            represent the minimum rectangle enclosing the text block.
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
            bbox_pixels = self._pdf_bbox_to_pixel((x0, y0, x1, y1))

            text_blocks.append(
                {
                    "text": text,
                    "bbox": bbox_pixels,  # (x0, y0, x1, y1) in pixels
                }
            )

        return text_blocks

    def _pdf_bbox_to_pixel(
        self, bbox_pdf: Tuple[float, float, float, float]
    ) -> Tuple[int, int, int, int]:
        """Convert bounding box from PDF points to pixel coordinates.

        PDF uses points as the base unit (1 point = 1/72 inch). This method
        converts bounding box coordinates to pixels at the configured DPI for
        image processing operations.

        Args:
            bbox_pdf: Bounding box in PDF point coordinates as tuple
                (x0, y0, x1, y1) where (x0, y0) is top-left and
                (x1, y1) is bottom-right.

        Returns:
            Bounding box in pixel coordinates as tuple (x0, y0, x1, y1) with
            integer values suitable for image array indexing.

        Note:
            Conversion formula: pixels = points * (dpi / 72.0)
        """
        scale = self.config.dpi / 72.0
        return (
            int(bbox_pdf[0] * scale),
            int(bbox_pdf[1] * scale),
            int(bbox_pdf[2] * scale),
            int(bbox_pdf[3] * scale),
        )

    def extract_embedded_text(self, pdf_path: str) -> Optional[str]:
        """Extract all embedded text from PDF without spatial coordinates.

        Provides fast text extraction when bounding box information is not
        needed. Useful for quick text searches, content preview, or text-only
        analysis. Does not render pages to images.

        Args:
            pdf_path: Path to PDF file to extract text from. Must be valid,
                non-encrypted PDF within size limits.

        Returns:
            All text from all pages concatenated with double newline separators
            between pages. Returns None if no text is found in the PDF.

        Raises:
            PDFProcessingError: If file validation fails, PDF is corrupted or
                encrypted, or text extraction fails.

        Note:
            Much faster than extract_pages() since it doesn't render images.
            Page limit (max_pages) is not enforced for this method.
        """
        # Validate PDF file
        is_valid, error_msg = validate_pdf_file(pdf_path)
        if not is_valid:
            raise PDFProcessingError(error_msg)

        doc = None
        try:
            doc = fitz.open(pdf_path)

            full_text = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    full_text.append(text)

            if full_text:
                return "\n\n".join(full_text)
            return None

        except Exception as e:
            raise PDFProcessingError(f"Failed to extract text from PDF: {e}")
        finally:
            if doc is not None:
                doc.close()

    def get_pdf_metadata(self, pdf_path: str) -> dict:
        """Extract metadata from PDF document.

        Retrieves document properties including title, author, creation date,
        and other metadata fields. Useful for audit trails, document
        classification, and provenance tracking.

        Args:
            pdf_path: Path to PDF file. Must be valid, non-encrypted PDF.

        Returns:
            Dictionary containing metadata fields (all snake_case):
                - title: Document title
                - author: Document author
                - subject: Document subject
                - creator: Application that created the document
                - producer: PDF producer software
                - creation_date: Creation timestamp (mapped from 'creationDate')
                - modification_date: Last modification timestamp (from 'modDate')
                - num_pages: Total number of pages
                - format: PDF format version
            All string fields return empty strings if not present in metadata.

        Note:
            Returns empty dict if metadata extraction fails. Logs warning but
            does not raise exception to allow processing to continue.
            PyMuPDF returns camelCase keys; this method standardizes to
            snake_case for Python conventions.
        """
        is_valid, error_msg = validate_pdf_file(pdf_path)
        if not is_valid:
            raise PDFProcessingError(error_msg)

        doc = None
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata

            # Standardize PyMuPDF camelCase keys to Python snake_case
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

            return result

        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")
            return {}
        finally:
            if doc is not None:
                doc.close()

    def validate_pdf(self, pdf_path: str) -> Tuple[bool, str]:
        """Perform comprehensive PDF validation checks.

        Validates file existence, size, format, encryption status, page count,
        and renderability. This is a thorough validation that attempts to
        open and render the first page to ensure the PDF is processable.

        Args:
            pdf_path: Path to PDF file to validate.

        Returns:
            Tuple containing:
                - bool: True if PDF passes all validation checks, False otherwise
                - str: Empty string if valid, error message describing failure
                  reason if invalid

        Note:
            This method is more comprehensive than basic file validation.
            It opens the PDF and attempts rendering, which may take longer
            but provides stronger guarantees of processability.
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
        doc = None
        try:
            doc = fitz.open(pdf_path)

            # Check if encrypted
            if doc.is_encrypted:
                return False, "PDF is encrypted/password protected"

            # Check page count
            if len(doc) == 0:
                return False, "PDF has no pages"

            if len(doc) > self.config.max_pages:
                logger.warning(
                    f"PDF has {len(doc)} pages, exceeds max of "
                    f"{self.config.max_pages}"
                )

            # Try to render first page as test
            page = doc[0]
            pix = page.get_pixmap()
            if pix.width == 0 or pix.height == 0:
                return False, "PDF page has invalid dimensions"

            return True, ""

        except fitz.FileDataError:
            return False, "Corrupted or invalid PDF file"
        except Exception as e:
            return False, f"PDF validation failed: {e}"
        finally:
            if doc is not None:
                doc.close()
