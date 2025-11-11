"""
PDF processing module for the Drawing Intelligence System.

This module provides functionality to extract pages as images and embedded text
from PDF files using PyMuPDF (fitz). It supports configurable DPI rendering,
page limits, grayscale conversion, parallel processing, and comprehensive validation.

The module implements resource management best practices using context managers
and provides both high-level batch processing and low-level page extraction APIs.

Classes:
    PDFConfig: Immutable configuration dataclass for PDF processing parameters.
    PDFProcessor: Main processor class for PDF extraction operations.

Exceptions:
    PDFEncryptionError: Raised when PDF is password-protected or encrypted.
    PDFPageRenderError: Raised when page rendering fails.
    PDFCorruptedError: Raised when PDF file is corrupted or invalid.

Typical usage example:
    config = PDFConfig(dpi=300, max_pages=10)
    processor = PDFProcessor(config)

    # Extract pages with progress tracking
    def on_progress(current, total):
        print(f"Processing page {current}/{total}")

    pages = processor.extract_pages("drawing.pdf", progress_callback=on_progress)

    # Or use generator for memory efficiency
    for page in processor.extract_pages_iter("drawing.pdf"):
        process_page(page)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import cv2
import fitz  # PyMuPDF
import numpy as np
from numpy.typing import NDArray

from ..models.data_structures import PDFPage
from ..utils.error_handlers import PDFProcessingError
from ..utils.pdf_utils import convert_pdf_points_to_pixels, convert_rgb_to_bgr
from ..utils.validation_utils import validate_file_size, validate_pdf_file

logger = logging.getLogger(__name__)

# Type aliases
PathType = Union[str, PathLike]
ProgressCallback = Callable[[int, int], None]


# ============================================================================
# Custom Exceptions for Granular Error Handling
# ============================================================================


class PDFEncryptionError(PDFProcessingError):
    """Raised when attempting to process encrypted or password-protected PDFs."""

    def __init__(self, message: str = "PDF is encrypted or password-protected"):
        super().__init__(message, recoverable=False)


class PDFPageRenderError(PDFProcessingError):
    """Raised when PDF page rendering fails."""

    def __init__(
        self, message: str, page_number: Optional[int] = None, recoverable: bool = False
    ):
        super().__init__(message, page_number=page_number, recoverable=recoverable)


class PDFCorruptedError(PDFProcessingError):
    """Raised when PDF file is corrupted or has invalid structure."""

    def __init__(self, message: str):
        super().__init__(f"Corrupted or invalid PDF: {message}", recoverable=False)


# ============================================================================
# TextBlock Dataclass for Type-Safe Text Extraction
# ============================================================================


@dataclass(frozen=True)
class TextBlock:
    """Represents a text block extracted from PDF with spatial coordinates.

    Attributes:
        text: The extracted text content (whitespace-stripped).
        bbox: Bounding box in pixels (x0, y0, x1, y1) where (x0, y0) is
            top-left corner and (x1, y1) is bottom-right corner.
    """

    text: str
    bbox: Tuple[int, int, int, int]


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class PDFConfig:
    """Immutable configuration for PDF processing operations.

    Controls rendering quality, file size limits, and output format preferences
    for PDF page extraction and text processing. Once created, configuration
    cannot be modified to ensure thread safety.

    Attributes:
        dpi: Resolution for page rendering in dots per inch. Higher values
            produce better quality but increase memory usage quadratically.
            Default is 300 DPI (standard for document processing).
            Must be between 72-1200.
        max_file_size_mb: Maximum allowed PDF file size in megabytes. Files
            exceeding this limit are rejected to prevent resource exhaustion.
            Default is 50 MB. Must be positive.
        max_pages: Maximum number of pages to process from a PDF. Prevents
            processing of extremely large documents. Default is 20 pages.
            Must be positive.
        convert_to_grayscale: If True, convert rendered images to grayscale.
            Reduces memory usage (~66% reduction) and may improve OCR
            performance. Default is True.
        parallel_workers: Number of worker threads for parallel page processing.
            Default is 1 (sequential). Set to None for auto-detection based on
            CPU count. Use with caution on memory-constrained systems.

    Raises:
        ValueError: If DPI is outside acceptable range or other parameters
            are invalid.

    Note:
        Memory usage scales with DPI²: 300 DPI uses 4x memory vs 150 DPI.
        Parallel processing multiplies memory usage by worker count.
    """

    dpi: int = 300
    max_file_size_mb: int = 50
    max_pages: int = 20
    convert_to_grayscale: bool = True
    parallel_workers: int = 1

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization.

        Raises:
            ValueError: If any parameter is outside acceptable range.
        """
        if not 72 <= self.dpi <= 1200:
            raise ValueError(
                f"DPI must be between 72 and 1200, got {self.dpi}. "
                "Extreme values can cause memory issues or processing failures."
            )
        if self.max_file_size_mb <= 0:
            raise ValueError(
                f"max_file_size_mb must be positive, got {self.max_file_size_mb}"
            )
        if self.max_pages <= 0:
            raise ValueError(f"max_pages must be positive, got {self.max_pages}")
        if self.parallel_workers < 1:
            raise ValueError(
                f"parallel_workers must be at least 1, got {self.parallel_workers}"
            )

    def with_overrides(self, **kwargs) -> PDFConfig:
        """Create new config with specified parameter overrides.

        Args:
            **kwargs: Parameters to override (dpi, max_pages, etc.).

        Returns:
            New PDFConfig instance with overridden parameters.

        Example:
            >>> config = PDFConfig(dpi=300)
            >>> high_res_config = config.with_overrides(dpi=600)
        """
        current_values = {
            "dpi": self.dpi,
            "max_file_size_mb": self.max_file_size_mb,
            "max_pages": self.max_pages,
            "convert_to_grayscale": self.convert_to_grayscale,
            "parallel_workers": self.parallel_workers,
        }
        current_values.update(kwargs)
        return PDFConfig(**current_values)


# ============================================================================
# Main Processor Class
# ============================================================================


class PDFProcessor:
    """Extracts pages and embedded text from PDF files.

    Uses PyMuPDF (fitz) for high-performance PDF processing with configurable
    rendering parameters. Supports page-by-page extraction with embedded text
    block coordinates, metadata extraction, parallel processing, and
    comprehensive validation.

    The processor handles PDF coordinate system conversion (points to pixels),
    multiple color formats (grayscale, RGB, RGBA), and includes safeguards
    against corrupted or encrypted files.

    Attributes:
        config: Immutable PDF processing configuration controlling DPI, limits,
            and format preferences.

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
        logger.info(
            f"PDFProcessor initialized (DPI: {config.dpi}, "
            f"Workers: {config.parallel_workers})"
        )

    @contextmanager
    def _open_pdf_validated(
        self, pdf_path: PathType
    ) -> Generator[fitz.Document, None, None]:
        """Context manager for validated PDF document handling.

        Performs comprehensive validation checks and safely opens/closes
        the PDF document. Eliminates code duplication across methods.

        Args:
            pdf_path: Path to PDF file to open.

        Yields:
            Opened fitz.Document object ready for processing.

        Raises:
            PDFProcessingError: If validation fails.
            PDFEncryptionError: If PDF is encrypted or password-protected.
            PDFCorruptedError: If PDF structure is invalid.
            IOError: If file cannot be read.

        Example:
            >>> with self._open_pdf_validated(pdf_path) as doc:
            ...     pages = len(doc)
        """
        path_str = str(pdf_path)

        # Step 1: Validate file format
        is_valid, error_msg = validate_pdf_file(path_str)
        if not is_valid:
            raise PDFProcessingError(error_msg)

        # Step 2: Validate file size
        is_valid, error_msg = validate_file_size(path_str, self.config.max_file_size_mb)
        if not is_valid:
            raise PDFProcessingError(error_msg)

        # Step 3: Open and validate document structure
        doc = None
        try:
            doc = fitz.open(path_str)

            # Check encryption
            if doc.is_encrypted:
                raise PDFEncryptionError()

            # Check page count
            if len(doc) == 0:
                raise PDFCorruptedError("PDF contains no pages")

            yield doc

        except PDFEncryptionError:
            raise
        except PDFCorruptedError:
            raise
        except fitz.FileDataError as e:
            raise PDFCorruptedError(str(e)) from e
        except IOError as e:
            raise PDFProcessingError(f"Failed to open PDF file: {e}") from e
        except Exception as e:
            raise PDFProcessingError(f"Unexpected error opening PDF: {e}") from e
        finally:
            if doc is not None:
                doc.close()

    def extract_pages(
        self,
        pdf_path: PathType,
        progress_callback: Optional[ProgressCallback] = None,
        **config_overrides,
    ) -> List[PDFPage]:
        """Extract all pages from PDF as images with embedded text blocks.

        Renders each page to a numpy image array at the configured DPI and
        extracts embedded text blocks with bounding box coordinates. Validates
        file before processing and enforces page count limits.

        Args:
            pdf_path: Path to the PDF file to process. Must be a valid,
                non-encrypted PDF file within size limits.
            progress_callback: Optional callback function(current, total) called
                after each page is processed. Useful for progress bars.
            **config_overrides: Temporary config overrides (e.g., dpi=600).

        Returns:
            List of PDFPage objects, each containing:
                - Rendered page image as numpy array
                - Page dimensions (width, height) in pixels
                - Embedded text blocks with bounding boxes
                - Page metadata (rotation, media box)

        Raises:
            PDFProcessingError: If file validation fails or processing fails.
            PDFEncryptionError: If PDF is encrypted.
            PDFCorruptedError: If PDF structure is invalid.
            PDFPageRenderError: If page rendering fails.

        Note:
            Processing stops at max_pages limit (configured in PDFConfig).
            Pages beyond this limit are logged but not processed. If all pages
            fail, raises PDFProcessingError with details.

        Example:
            >>> def progress(current, total):
            ...     print(f"{current}/{total}")
            >>> pages = processor.extract_pages("doc.pdf", progress_callback=progress)
        """
        # Apply config overrides if provided
        config = (
            self.config.with_overrides(**config_overrides)
            if config_overrides
            else self.config
        )

        with self._open_pdf_validated(pdf_path) as doc:
            num_pages = min(len(doc), config.max_pages)

            if len(doc) > config.max_pages:
                logger.warning(
                    f"PDF has {len(doc)} pages, processing only first {config.max_pages}"
                )

            pages: List[PDFPage] = []
            errors: List[Tuple[int, str]] = []

            # Sequential or parallel processing
            if config.parallel_workers == 1:
                pages, errors = self._extract_pages_sequential(
                    doc, num_pages, config, progress_callback
                )
            else:
                pages, errors = self._extract_pages_parallel(
                    doc, num_pages, config, progress_callback
                )

            # Report any errors
            if errors:
                error_summary = "; ".join(
                    [f"page {pg}: {err}" for pg, err in errors[:3]]
                )
                logger.warning(
                    f"Failed to extract {len(errors)}/{num_pages} pages: {error_summary}"
                )

            if not pages:
                raise PDFProcessingError(
                    f"No pages successfully extracted from {pdf_path}. "
                    f"All {num_pages} pages failed processing."
                )

            logger.info(
                f"Successfully extracted {len(pages)}/{num_pages} pages from {pdf_path}"
            )
            return pages

    def _extract_pages_sequential(
        self,
        doc: fitz.Document,
        num_pages: int,
        config: PDFConfig,
        progress_callback: Optional[ProgressCallback],
    ) -> Tuple[List[PDFPage], List[Tuple[int, str]]]:
        """Extract pages sequentially (single-threaded).

        Args:
            doc: Opened PDF document.
            num_pages: Number of pages to extract.
            config: Configuration to use.
            progress_callback: Optional progress callback.

        Returns:
            Tuple of (successfully extracted pages, list of (page_num, error) tuples).
        """
        pages: List[PDFPage] = []
        errors: List[Tuple[int, str]] = []

        for page_num in range(num_pages):
            try:
                page = doc[page_num]
                pdf_page = self._extract_single_page(page, page_num, config)
                pages.append(pdf_page)
                logger.debug(f"Extracted page {page_num + 1}/{num_pages}")

                if progress_callback:
                    progress_callback(page_num + 1, num_pages)

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error processing page {page_num + 1}: {error_msg}")
                errors.append((page_num + 1, error_msg))

        return pages, errors

    def _extract_pages_parallel(
        self,
        doc: fitz.Document,
        num_pages: int,
        config: PDFConfig,
        progress_callback: Optional[ProgressCallback],
    ) -> Tuple[List[PDFPage], List[Tuple[int, str]]]:
        """Extract pages in parallel using thread pool.

        Args:
            doc: Opened PDF document.
            num_pages: Number of pages to extract.
            config: Configuration to use.
            progress_callback: Optional progress callback.

        Returns:
            Tuple of (successfully extracted pages, list of (page_num, error) tuples).

        Note:
            Pages are returned in order despite parallel processing.
        """
        pages_dict: Dict[int, PDFPage] = {}
        errors: List[Tuple[int, str]] = []
        completed = 0

        with ThreadPoolExecutor(max_workers=config.parallel_workers) as executor:
            # Submit all page extraction tasks
            future_to_page = {
                executor.submit(
                    self._extract_single_page, doc[page_num], page_num, config
                ): page_num
                for page_num in range(num_pages)
            }

            # Collect results as they complete
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                completed += 1

                try:
                    pdf_page = future.result()
                    pages_dict[page_num] = pdf_page
                    logger.debug(f"Extracted page {page_num + 1}/{num_pages}")

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error processing page {page_num + 1}: {error_msg}")
                    errors.append((page_num + 1, error_msg))

                if progress_callback:
                    progress_callback(completed, num_pages)

        # Return pages in original order
        pages = [pages_dict[i] for i in sorted(pages_dict.keys())]
        return pages, errors

    def _extract_single_page(
        self, page: fitz.Page, page_num: int, config: PDFConfig
    ) -> PDFPage:
        """Extract a single page with all its data.

        Args:
            page: PyMuPDF page object.
            page_num: Zero-based page number.
            config: Configuration to use.

        Returns:
            Extracted PDFPage object.

        Raises:
            PDFPageRenderError: If page rendering or extraction fails.
        """
        try:
            # Render page to image
            image, dimensions = self._render_page_to_image(page, config)

            # Extract text blocks
            text_blocks = self._extract_text_blocks(page, config)

            # Create PDFPage object
            pdf_page = PDFPage(
                page_number=page_num,
                image=image,
                dimensions=dimensions,
                embedded_text_blocks=[
                    {"text": tb.text, "bbox": tb.bbox} for tb in text_blocks
                ],
                dpi=config.dpi,
                metadata={
                    "rotation": page.rotation,
                    "mediabox": page.rect,
                },
            )

            return pdf_page

        except Exception as e:
            raise PDFPageRenderError(
                f"Failed to process page {page_num + 1}: {e}", page_number=page_num
            ) from e

    def extract_pages_iter(
        self,
        pdf_path: PathType,
        progress_callback: Optional[ProgressCallback] = None,
        **config_overrides,
    ) -> Iterator[PDFPage]:
        """Extract pages as generator for memory-efficient streaming.

        Yields pages one at a time without loading entire document into memory.
        Useful for processing very large PDFs where memory is constrained.

        Args:
            pdf_path: Path to the PDF file to process.
            progress_callback: Optional callback function(current, total) called
                after each page is yielded.
            **config_overrides: Temporary config overrides (e.g., dpi=600).

        Yields:
            PDFPage objects one at a time.

        Raises:
            PDFProcessingError: If file validation fails.
            PDFEncryptionError: If PDF is encrypted.
            PDFCorruptedError: If PDF structure is invalid.

        Note:
            Failed pages are logged but skipped. Generator continues to next page.

        Example:
            >>> for page in processor.extract_pages_iter("large.pdf"):
            ...     process_page(page)
            ...     del page  # Free memory immediately
        """
        config = (
            self.config.with_overrides(**config_overrides)
            if config_overrides
            else self.config
        )

        with self._open_pdf_validated(pdf_path) as doc:
            num_pages = min(len(doc), config.max_pages)

            if len(doc) > config.max_pages:
                logger.warning(
                    f"PDF has {len(doc)} pages, processing only first {config.max_pages}"
                )

            for page_num in range(num_pages):
                try:
                    page = doc[page_num]
                    pdf_page = self._extract_single_page(page, page_num, config)

                    if progress_callback:
                        progress_callback(page_num + 1, num_pages)

                    yield pdf_page

                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {e}")
                    # Continue to next page instead of stopping

    def _render_page_to_image(
        self, page: fitz.Page, config: PDFConfig
    ) -> Tuple[NDArray[np.uint8], Tuple[int, int]]:
        """Render PDF page to numpy image array at configured DPI.

        Converts a PyMuPDF page object to a numpy array suitable for image
        processing. Handles multiple color formats (grayscale, RGB, RGBA) and
        converts to BGR format for OpenCV compatibility when needed.

        Args:
            page: PyMuPDF page object to render.
            config: Configuration specifying DPI and color preferences.

        Returns:
            Tuple containing:
                - Numpy array with rendered page image:
                    * Grayscale: (height, width) uint8
                    * Color: (height, width, 3) BGR uint8
                - Tuple of (width, height) in pixels at configured DPI

        Raises:
            PDFPageRenderError: If page has unsupported color format or
                rendering fails.

        Note:
            PDF default resolution is 72 DPI. This method scales rendering
            according to config.dpi using zoom matrix transformation.
        """
        try:
            # Calculate zoom factor for desired DPI
            zoom = config.dpi / 72.0  # 72 DPI is PDF default
            matrix = fitz.Matrix(zoom, zoom)

            # Render with appropriate color space
            if config.convert_to_grayscale:
                pix = page.get_pixmap(
                    matrix=matrix, colorspace=fitz.csGRAY, alpha=False
                )
            else:
                pix = page.get_pixmap(matrix=matrix)

            # Store dimensions
            dimensions = (pix.width, pix.height)

            # Convert pixmap to numpy array
            img_data = pix.samples
            channels = pix.n

            if channels == 1:
                # Grayscale
                image = np.frombuffer(img_data, dtype=np.uint8).reshape(
                    pix.height, pix.width
                )
            elif channels == 3:
                # RGB -> BGR for OpenCV (using utility function)
                image = np.frombuffer(img_data, dtype=np.uint8).reshape(
                    pix.height, pix.width, 3
                )
                image = convert_rgb_to_bgr(image)
            elif channels == 4:
                # RGBA -> BGR for OpenCV
                image = np.frombuffer(img_data, dtype=np.uint8).reshape(
                    pix.height, pix.width, 4
                )
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            else:
                raise PDFPageRenderError(
                    f"Unsupported color format: {channels} channels"
                )

            return image, dimensions

        except PDFPageRenderError:
            raise
        except Exception as e:
            raise PDFPageRenderError(f"Page rendering failed: {e}") from e

    def _extract_text_blocks(
        self, page: fitz.Page, config: PDFConfig
    ) -> List[TextBlock]:
        """Extract text blocks with bounding box coordinates from page.

        Retrieves all text blocks from the page along with their spatial
        positions. Empty text blocks are filtered out. Coordinates are
        converted from PDF points (72 DPI) to pixels at the configured DPI.

        Args:
            page: PyMuPDF page object from which to extract text.
            config: Configuration specifying DPI for coordinate conversion.

        Returns:
            List of TextBlock objects with text content and pixel coordinates.

        Note:
            Only blocks with non-empty text are returned. Bounding boxes
            represent the minimum rectangle enclosing the text block.
        """
        text_blocks: List[TextBlock] = []

        # Get text blocks with coordinates
        # blocks format: (x0, y0, x1, y1, "text", block_no, block_type)
        blocks = page.get_text("blocks")

        for block in blocks:
            x0, y0, x1, y1, text = block[:5]

            # Skip empty text
            text_stripped = text.strip()
            if not text_stripped:
                continue

            # Convert PDF coordinates to pixel coordinates
            bbox_pixels = convert_pdf_points_to_pixels((x0, y0, x1, y1), config.dpi)

            text_blocks.append(TextBlock(text=text_stripped, bbox=bbox_pixels))

        return text_blocks

    def extract_embedded_text(
        self, pdf_path: PathType, **config_overrides
    ) -> Optional[str]:
        """Extract all embedded text from PDF without spatial coordinates.

        Provides fast text extraction when bounding box information is not
        needed. Useful for quick text searches, content preview, or text-only
        analysis. Does not render pages to images.

        Args:
            pdf_path: Path to PDF file to extract text from. Must be valid,
                non-encrypted PDF within size limits.
            **config_overrides: Temporary config overrides (e.g., max_pages=100).

        Returns:
            All text from all pages concatenated with double newline separators
            between pages. Returns None if no text is found in any page.

        Raises:
            PDFProcessingError: If file validation fails.
            PDFEncryptionError: If PDF is encrypted.
            PDFCorruptedError: If PDF structure is invalid.

        Note:
            Much faster than extract_pages() since it doesn't render images.
            Respects max_pages limit for consistency with other methods.
            If a page fails text extraction, it's logged and skipped.

        Example:
            >>> text = processor.extract_embedded_text("document.pdf")
            >>> if text:
            ...     print(f"Found {len(text)} characters")
        """
        config = (
            self.config.with_overrides(**config_overrides)
            if config_overrides
            else self.config
        )

        with self._open_pdf_validated(pdf_path) as doc:
            full_text: List[str] = []
            num_pages = min(len(doc), config.max_pages)
            failed_pages = 0

            for page_num in range(num_pages):
                try:
                    page = doc[page_num]
                    text = page.get_text()
                    if text.strip():
                        full_text.append(text)
                except Exception as e:
                    logger.warning(
                        f"Failed to extract text from page {page_num + 1}: {e}"
                    )
                    failed_pages += 1

            if failed_pages > 0:
                logger.info(
                    f"Text extraction completed with {failed_pages} failed pages"
                )

            if full_text:
                return "\n\n".join(full_text)
            return None

    def get_pdf_metadata(self, pdf_path: PathType) -> Dict[str, Union[str, int]]:
        """Extract metadata from PDF document.

        Retrieves document properties including title, author, creation date,
        and other metadata fields. Useful for audit trails, document
        classification, and provenance tracking.

        Args:
            pdf_path: Path to PDF file. Must be valid, non-encrypted PDF.

        Returns:
            Dictionary containing metadata fields (all snake_case):
                - title (str): Document title
                - author (str): Document author
                - subject (str): Document subject
                - creator (str): Application that created the document
                - producer (str): PDF producer software
                - creation_date (str): Creation timestamp (ISO format if parseable)
                - modification_date (str): Last modification timestamp
                - num_pages (int): Total number of pages
                - format (str): PDF format version
            All string fields return empty strings if not present in metadata.

        Raises:
            PDFProcessingError: If file validation fails or PDF cannot be opened.
            PDFEncryptionError: If PDF is encrypted.
            PDFCorruptedError: If PDF structure is invalid.

        Note:
            PyMuPDF returns camelCase keys; this method standardizes to
            snake_case for Python conventions.

        Example:
            >>> metadata = processor.get_pdf_metadata("drawing.pdf")
            >>> print(f"Title: {metadata['title']}, Pages: {metadata['num_pages']}")
        """
        with self._open_pdf_validated(pdf_path) as doc:
            metadata = doc.metadata

            # Standardize PyMuPDF camelCase keys to Python snake_case
            result: Dict[str, Union[str, int]] = {
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

    def extract_pages_from_directory(
        self,
        directory_path: PathType,
        pattern: str = "*.pdf",
        recursive: bool = False,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        **config_overrides,
    ) -> Dict[str, List[PDFPage]]:
        """Batch process all PDFs in a directory.

        Extracts pages from all PDF files matching the pattern in the specified
        directory. Useful for bulk processing of drawing archives.

        Args:
            directory_path: Path to directory containing PDF files.
            pattern: Glob pattern for matching PDF files (default: "*.pdf").
            recursive: If True, search subdirectories recursively.
            progress_callback: Optional callback(filename, current_file, total_files)
                called before processing each file.
            **config_overrides: Temporary config overrides applied to all files.

        Returns:
            Dictionary mapping file paths (as strings) to lists of extracted
            PDFPage objects. Files that fail processing are omitted from results
            but logged as errors.

        Raises:
            ValueError: If directory_path doesn't exist or is not a directory.

        Example:
            >>> def progress(filename, current, total):
            ...     print(f"Processing {filename} ({current}/{total})")
            >>> results = processor.extract_pages_from_directory(
            ...     "drawings/",
            ...     progress_callback=progress
            ... )
            >>> print(f"Processed {len(results)} files")
        """
        dir_path = Path(directory_path)
        if not dir_path.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        # Find all matching PDFs
        if recursive:
            pdf_files = list(dir_path.rglob(pattern))
        else:
            pdf_files = list(dir_path.glob(pattern))

        if not pdf_files:
            logger.warning(
                f"No PDF files matching '{pattern}' found in {directory_path}"
            )
            return {}

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        results: Dict[str, List[PDFPage]] = {}
        failed_files: List[str] = []

        for idx, pdf_file in enumerate(pdf_files, start=1):
            file_str = str(pdf_file)

            if progress_callback:
                progress_callback(pdf_file.name, idx, len(pdf_files))

            try:
                pages = self.extract_pages(pdf_file, **config_overrides)
                results[file_str] = pages
                logger.info(f"Successfully processed {pdf_file.name}")

            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                failed_files.append(file_str)

        logger.info(
            f"Batch processing complete: {len(results)} succeeded, "
            f"{len(failed_files)} failed"
        )

        return results


# ============================================================================
# Manual Testing Entry Point
# ============================================================================


def _demo_usage():
    """Demonstrate basic usage of PDFProcessor for manual testing."""
    import sys
    from pathlib import Path

    print("=" * 70)
    print("PDF Processor Demo")
    print("=" * 70)

    # Check for command-line argument
    if len(sys.argv) < 2:
        print("\nUsage: python pdf_processor.py <path_to_pdf>")
        print("\nExample:")
        print("  python pdf_processor.py sample.pdf")
        print("  python pdf_processor.py /path/to/drawings/")
        return

    input_path = Path(sys.argv[1])

    # Create processor with default config
    config = PDFConfig(dpi=300, max_pages=5, parallel_workers=2)
    processor = PDFProcessor(config)

    try:
        if input_path.is_file():
            # Single file processing
            print(f"\n📄 Processing file: {input_path.name}")
            print("-" * 70)

            # Extract metadata first
            print("\n1. Extracting metadata...")
            metadata = processor.get_pdf_metadata(input_path)
            print(f"   Title: {metadata.get('title', 'N/A')}")
            print(f"   Author: {metadata.get('author', 'N/A')}")
            print(f"   Pages: {metadata.get('num_pages', 'N/A')}")
            print(f"   Format: {metadata.get('format', 'N/A')}")

            # Extract embedded text
            print("\n2. Extracting embedded text...")
            text = processor.extract_embedded_text(input_path)
            if text:
                print(f"   Extracted {len(text)} characters")
                print(f"   Preview: {text[:200]}..." if len(text) > 200 else text)
            else:
                print("   No text found")

            # Extract pages with progress
            print("\n3. Extracting pages...")

            def progress(current, total):
                print(f"   Progress: {current}/{total} pages")

            pages = processor.extract_pages(input_path, progress_callback=progress)

            print(f"\n✅ Successfully extracted {len(pages)} pages")
            for idx, page in enumerate(pages):
                print(f"   Page {idx + 1}:")
                print(f"     - Dimensions: {page.dimensions}")
                print(f"     - Image shape: {page.image.shape}")
                print(f"     - Text blocks: {len(page.embedded_text_blocks)}")
                print(f"     - DPI: {page.dpi}")

        elif input_path.is_dir():
            # Directory batch processing
            print(f"\n📁 Processing directory: {input_path}")
            print("-" * 70)

            def batch_progress(filename, current, total):
                print(f"   [{current}/{total}] Processing: {filename}")

            results = processor.extract_pages_from_directory(
                input_path, progress_callback=batch_progress, dpi=300
            )

            print(f"\n✅ Batch processing complete")
            print(f"   Total files processed: {len(results)}")
            for pdf_path, pages in results.items():
                print(f"   - {Path(pdf_path).name}: {len(pages)} pages")

        else:
            print(f"\n❌ Error: {input_path} is not a valid file or directory")

    except PDFEncryptionError as e:
        print(f"\n🔒 Error: {e}")
    except PDFCorruptedError as e:
        print(f"\n💥 Error: {e}")
    except PDFProcessingError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n⚠️  Unexpected error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s"
    )
    _demo_usage()
