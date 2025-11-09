"""
OCR pipeline module for the Drawing Intelligence System.

This module provides a dual-engine OCR pipeline optimized for technical drawings.
It uses PaddleOCR as the primary engine for speed and accuracy, with EasyOCR
as a fallback for low-confidence regions. Supports layout analysis to identify
title blocks, tables, and text regions.

Typical usage example:

    config = OCRConfig(
        primary_engine="paddleocr",
        confidence_threshold=0.85,
        languages=["en"]
    )
    pipeline = OCRPipeline(config)
    result = pipeline.extract_text(image)
    print(f"Found {len(result.text_blocks)} text blocks")
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Any, Dict
import numpy as np

from ..models.data_structures import OCRResult, TextBlock, LayoutRegion
from ..utils.geometry_utils import BoundingBox
from ..utils.text_utils import normalize_whitespace
from ..utils.file_utils import generate_unique_id
from ..utils.error_handlers import OCRError

logger = logging.getLogger(__name__)

# Valid OCR engine names
VALID_ENGINES = {"paddleocr", "easyocr"}


@dataclass
class OCRConfig:
    """Configuration for OCR pipeline behavior and thresholds.

    Controls the OCR engine selection, confidence thresholds for fallback
    processing, and performance options like GPU acceleration.

    Attributes:
        primary_engine: Primary OCR engine name ('paddleocr' or 'easyocr').
            Default: 'paddleocr'.
        fallback_engine: Fallback OCR engine for low-confidence regions.
            Default: 'easyocr'.
        confidence_threshold: Minimum confidence (0.0-1.0) to avoid fallback
            processing. Blocks below this threshold are reprocessed. Default: 0.85.
        languages: List of ISO 639-1 language codes for OCR (e.g., ['en', 'de']).
            If None, defaults to ['en'] in __post_init__.
        layout_analysis: Whether to perform layout region classification
            (title blocks, tables, etc.). Default: True.
        use_gpu: Whether to use GPU acceleration if available. Will fall back
            to CPU if GPU initialization fails. Default: True.

    Raises:
        ValueError: If engine names are invalid or configuration is inconsistent.

    Note:
        Different OCR engines have different language support. PaddleOCR supports
        80+ languages, EasyOCR supports 80+ languages. Validate language codes
        against engine capabilities before use.
    """

    primary_engine: str = "paddleocr"
    fallback_engine: str = "easyocr"
    confidence_threshold: float = 0.85
    languages: Optional[List[str]] = None
    layout_analysis: bool = True
    use_gpu: bool = True

    def __post_init__(self) -> None:
        """Initialize and validate configuration.

        Raises:
            ValueError: If engine names are invalid or threshold out of range.
        """
        if self.languages is None:
            self.languages = ["en"]

        # Validate engine names
        if self.primary_engine not in VALID_ENGINES:
            raise ValueError(
                f"Invalid primary_engine '{self.primary_engine}'. "
                f"Must be one of: {VALID_ENGINES}"
            )
        if self.fallback_engine not in VALID_ENGINES:
            raise ValueError(
                f"Invalid fallback_engine '{self.fallback_engine}'. "
                f"Must be one of: {VALID_ENGINES}"
            )

        # Validate confidence threshold
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, "
                f"got {self.confidence_threshold}"
            )

        # Validate languages list is not empty
        if not self.languages:
            raise ValueError("languages list cannot be empty")


class OCRPipeline:
    """Dual-engine OCR pipeline with intelligent fallback processing.

    This class implements a two-stage OCR strategy:
    1. Primary OCR (PaddleOCR) processes the entire image for speed
    2. Low-confidence regions are cropped and reprocessed with fallback (EasyOCR)
    3. Results are merged, keeping the highest confidence text for each region

    The pipeline also performs optional layout analysis to classify regions
    as title blocks, tables, or general text areas.

    Attributes:
        config: OCR configuration object controlling engine behavior.
        _primary_ocr: Lazily-initialized PaddleOCR instance (None until first use).
        _fallback_ocr: Lazily-initialized EasyOCR Reader instance (None until first use).

    Example:
        >>> config = OCRConfig(confidence_threshold=0.80)
        >>> pipeline = OCRPipeline(config)
        >>> result = pipeline.extract_text(image)
        >>> print(f"Extracted {len(result.text_blocks)} text blocks")
        >>> print(f"Average confidence: {result.average_confidence:.2f}")
    """

    def __init__(self, config: OCRConfig) -> None:
        """Initialize OCR pipeline with configuration.

        OCR engines are not loaded immediately (lazy initialization) to save
        memory and startup time. They are loaded on first use.

        Args:
            config: OCR configuration specifying engines, thresholds, and options.

        Note:
            This method does not validate that OCR engines are installed.
            Installation errors will be raised on first use during lazy loading.
        """
        self.config = config

        # Lazy loading of OCR engines (type hints for clarity)
        self._primary_ocr: Optional[Any] = None
        self._fallback_ocr: Optional[Any] = None

        logger.info(
            f"OCRPipeline initialized (Primary: {config.primary_engine}, "
            f"Fallback: {config.fallback_engine})"
        )

    @property
    def _primary_language(self) -> str:
        """Get primary language for OCR engines.

        Returns:
            First language from config.languages, or 'en' if list is empty.

        Note:
            PaddleOCR only supports one language at initialization. This property
            provides safe access to the primary language with a fallback default.
        """
        return self.config.languages[0] if self.config.languages else "en"

    def extract_text(self, image: np.ndarray) -> OCRResult:
        """Extract text from image with bounding boxes and confidence scores.

        Processes the image with the primary OCR engine, identifies low-confidence
        regions, and reprocesses them with the fallback engine. Performs layout
        analysis if enabled in configuration.

        Args:
            image: Input image as numpy array. Can be grayscale (H, W) or
                color (H, W, 3) in BGR format (OpenCV convention).

        Returns:
            OCRResult containing:
                - text_blocks: List of TextBlock objects with content and positions
                - language_detected: Language code (from config.languages[0])
                - layout_regions: List of classified layout regions (if enabled)
                - average_confidence: Mean confidence across all text blocks

        Raises:
            OCRError: If OCR extraction fails due to engine errors, invalid input,
                or missing dependencies.

        Example:
            >>> result = pipeline.extract_text(image)
            >>> for block in result.text_blocks:
            ...     if block.confidence < 0.9:
            ...         print(f"Low confidence: {block.content}")

        Note:
            Language is determined from config.languages, not auto-detected.
            Auto-detection is not currently implemented.
        """
        try:
            # Use primary language from configuration
            language = self._primary_language

            # Primary OCR
            logger.debug(f"Running primary OCR ({self.config.primary_engine})")
            text_blocks = self._run_primary_ocr(image)

            # Layout analysis
            layout_regions: List[LayoutRegion] = []
            if self.config.layout_analysis:
                layout_regions = self._classify_layout_regions(image, text_blocks)

            # Calculate average confidence
            avg_confidence = self._calculate_average_confidence(text_blocks)

            # Identify low-confidence regions for fallback
            low_conf_blocks = [
                b
                for b in text_blocks
                if b.confidence < self.config.confidence_threshold
            ]

            if low_conf_blocks:
                logger.debug(
                    f"Found {len(low_conf_blocks)} low-confidence blocks, "
                    f"applying fallback OCR"
                )
                improved_blocks = self._apply_fallback_ocr(image, low_conf_blocks)

                # Replace low-confidence blocks with improved ones using dict lookup
                text_blocks = self._merge_improved_blocks(text_blocks, improved_blocks)

                # Recalculate average confidence after improvements
                avg_confidence = self._calculate_average_confidence(text_blocks)

            result = OCRResult(
                text_blocks=text_blocks,
                language_detected=language,
                layout_regions=layout_regions,
                average_confidence=avg_confidence,
            )

            logger.info(
                f"OCR complete: {len(text_blocks)} blocks, "
                f"avg confidence: {avg_confidence:.2f}"
            )

            return result

        except OCRError:
            # Re-raise OCR-specific errors without wrapping
            raise
        except Exception as e:
            raise OCRError(
                f"OCR extraction failed: {e}",
                ocr_engine=self.config.primary_engine,
                recoverable=True,
            )

    def _run_primary_ocr(self, image: np.ndarray) -> List[TextBlock]:
        """Run primary OCR engine (PaddleOCR) on full image.

        Lazily initializes PaddleOCR if not already loaded. Converts PaddleOCR's
        output format (bbox points + text + confidence) to standardized TextBlock
        objects.

        Args:
            image: Input image (grayscale or BGR).

        Returns:
            List of TextBlock objects extracted from the image. Empty list if no
            text detected.

        Raises:
            OCRError: If PaddleOCR is not installed, initialization fails, or
                execution encounters an error.

        Note:
            PaddleOCR returns bounding boxes as 4-point polygons which are
            converted to axis-aligned rectangles (may lose rotation info).
            Language is determined by initialization, not per-call.
        """
        if self._primary_ocr is None:
            self._initialize_primary_ocr()

        try:
            # PaddleOCR returns list of lines with [bbox, (text, confidence)]
            result = self._primary_ocr.ocr(image, cls=True)

            text_blocks: List[TextBlock] = []

            # Handle PaddleOCR output format
            if result and result[0]:
                for line in result[0]:
                    if len(line) >= 2:
                        bbox_points, (text, confidence) = line[0], line[1]

                        # Convert bbox points to BoundingBox
                        bbox = self._points_to_bbox(bbox_points)

                        # Normalize text
                        text = self._normalize_text(text)

                        if text:  # Skip empty text
                            text_block = TextBlock(
                                text_id=generate_unique_id("TXT"),
                                content=text,
                                bbox=bbox,
                                confidence=float(confidence),
                                ocr_engine="paddleocr",
                                region_type="text",  # Refined by layout analysis
                            )
                            text_blocks.append(text_block)

            return text_blocks

        except Exception as e:
            logger.error(f"PaddleOCR execution failed: {e}")
            raise OCRError(
                f"PaddleOCR execution failed: {e}",
                ocr_engine="paddleocr",
                recoverable=True,
            )

    def _apply_fallback_ocr(
        self, image: np.ndarray, low_confidence_blocks: List[TextBlock]
    ) -> List[TextBlock]:
        """Reprocess low-confidence regions with fallback OCR engine (EasyOCR).

        For each low-confidence block, crops the region (with 5px padding),
        runs EasyOCR, and keeps the result only if confidence improved.

        Args:
            image: Full source image for cropping regions.
            low_confidence_blocks: List of TextBlock objects below the confidence
                threshold that need reprocessing.

        Returns:
            List of TextBlock objects with either improved results (from EasyOCR)
            or original results (if fallback didn't improve confidence). Length
            always matches input list.

        Note:
            - Each block is processed independently (no batching)
            - If fallback OCR fails or finds no text, original block is kept
            - Bounding box coordinates are preserved from original detection
        """
        if self._fallback_ocr is None:
            self._initialize_fallback_ocr()

        improved_blocks: List[TextBlock] = []
        improvement_count = 0

        for block in low_confidence_blocks:
            try:
                # Crop region from image with padding
                cropped = self._crop_region(image, block.bbox, padding=5)

                # Run EasyOCR on cropped region
                result = self._fallback_ocr.readtext(cropped)

                if result:
                    # Take first result (most confident)
                    bbox_points, text, confidence = result[0]

                    # Compare with original
                    if confidence > block.confidence:
                        # Keep improved result
                        text = self._normalize_text(text)

                        improved_block = TextBlock(
                            text_id=block.text_id,
                            content=text,
                            bbox=block.bbox,
                            confidence=float(confidence),
                            ocr_engine="easyocr",
                            region_type=block.region_type,
                        )
                        improved_blocks.append(improved_block)
                        improvement_count += 1
                        logger.debug(
                            f"Improved block {block.text_id}: "
                            f"{block.confidence:.2f} → {confidence:.2f}"
                        )
                    else:
                        # Keep original
                        improved_blocks.append(block)
                else:
                    # No result from fallback, keep original
                    improved_blocks.append(block)

            except Exception as e:
                logger.warning(f"Fallback OCR failed for block {block.text_id}: {e}")
                # Keep original on error
                improved_blocks.append(block)

        logger.info(
            f"Fallback OCR improved {improvement_count}/{len(low_confidence_blocks)} blocks"
        )

        return improved_blocks

    def _initialize_primary_ocr(self) -> None:
        """Lazy initialization of PaddleOCR engine.

        Creates PaddleOCR instance with configuration from self.config. Uses
        angle classification for better text orientation handling. Attempts
        GPU initialization first, falls back to CPU if GPU unavailable.

        Raises:
            OCRError: If PaddleOCR package is not installed or initialization
                fails on both GPU and CPU.

        Note:
            PaddleOCR models are downloaded automatically on first use (~100MB).
            Language is set from config.languages[0].
        """
        try:
            from paddleocr import PaddleOCR

            lang = self._primary_language

            # Try GPU first if requested, fall back to CPU if it fails
            if self.config.use_gpu:
                try:
                    self._primary_ocr = PaddleOCR(
                        use_angle_cls=True,
                        lang=lang,
                        use_gpu=True,
                        show_log=False,
                    )
                    logger.info(f"PaddleOCR initialized with GPU (language: {lang})")
                except Exception as gpu_error:
                    logger.warning(
                        f"GPU initialization failed: {gpu_error}. "
                        f"Falling back to CPU"
                    )
                    self._primary_ocr = PaddleOCR(
                        use_angle_cls=True,
                        lang=lang,
                        use_gpu=False,
                        show_log=False,
                    )
                    logger.info(f"PaddleOCR initialized with CPU (language: {lang})")
            else:
                # CPU-only initialization
                self._primary_ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=lang,
                    use_gpu=False,
                    show_log=False,
                )
                logger.info(f"PaddleOCR initialized with CPU (language: {lang})")

        except ImportError:
            raise OCRError(
                "PaddleOCR not installed. Install with: pip install paddleocr",
                ocr_engine="paddleocr",
                recoverable=False,
            )
        except Exception as e:
            raise OCRError(
                f"Failed to initialize PaddleOCR: {e}",
                ocr_engine="paddleocr",
                recoverable=False,
            )

    def _initialize_fallback_ocr(self) -> None:
        """Lazy initialization of EasyOCR engine.

        Creates EasyOCR Reader instance with languages from self.config. Models
        are downloaded automatically if not cached. Attempts GPU initialization
        first, falls back to CPU if GPU unavailable.

        Raises:
            OCRError: If EasyOCR package is not installed or initialization fails
                on both GPU and CPU.

        Note:
            EasyOCR downloads language models on first use (~100MB per language).
        """
        try:
            import easyocr

            # Try GPU first if requested, fall back to CPU if it fails
            if self.config.use_gpu:
                try:
                    self._fallback_ocr = easyocr.Reader(self.config.languages, gpu=True)
                    logger.info(
                        f"EasyOCR initialized with GPU "
                        f"(languages: {self.config.languages})"
                    )
                except Exception as gpu_error:
                    logger.warning(
                        f"GPU initialization failed: {gpu_error}. "
                        f"Falling back to CPU"
                    )
                    self._fallback_ocr = easyocr.Reader(
                        self.config.languages, gpu=False
                    )
                    logger.info(
                        f"EasyOCR initialized with CPU "
                        f"(languages: {self.config.languages})"
                    )
            else:
                # CPU-only initialization
                self._fallback_ocr = easyocr.Reader(self.config.languages, gpu=False)
                logger.info(
                    f"EasyOCR initialized with CPU "
                    f"(languages: {self.config.languages})"
                )

        except ImportError:
            raise OCRError(
                "EasyOCR not installed. Install with: pip install easyocr",
                ocr_engine="easyocr",
                recoverable=False,
            )
        except Exception as e:
            raise OCRError(
                f"Failed to initialize EasyOCR: {e}",
                ocr_engine="easyocr",
                recoverable=False,
            )

    def _points_to_bbox(self, points: List[List[float]]) -> BoundingBox:
        """Convert OCR polygon points to axis-aligned bounding box.

        Takes a list of 2D points (typically 4 corners of a quadrilateral) and
        computes the minimal axis-aligned rectangle containing all points.
        Coordinates are rounded to nearest integer pixels.

        Args:
            points: List of [x, y] coordinate pairs, typically 4 corners from
                OCR engine output.

        Returns:
            BoundingBox with top-left corner (x, y) and dimensions (width, height)
            in integer pixel coordinates.

        Note:
            This conversion loses rotation information. A rotated text region
            will have a larger bounding box than its tight oriented bounding box.
        """
        # Extract x and y coordinates
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        # Get bounding rectangle (rounded to integer pixels)
        x_min = int(min(xs))
        y_min = int(min(ys))
        x_max = int(max(xs))
        y_max = int(max(ys))

        return BoundingBox(x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min)

    def _normalize_text(self, text: str) -> str:
        """Normalize OCR output text by fixing common errors.

        Applies whitespace normalization and corrects common OCR misrecognitions
        (e.g., pipe character '|' misread as letter 'I').

        Args:
            text: Raw text string from OCR engine.

        Returns:
            Normalized text with whitespace cleaned and artifacts corrected.

        Note:
            Uses text_utils.normalize_whitespace() for consistent whitespace handling.
        """
        # Normalize whitespace (uses globally imported function)
        text = normalize_whitespace(text)

        # Remove common OCR artifacts
        text = text.replace("|", "I")  # Pipe often misread as I

        return text

    def _classify_layout_regions(
        self, image: np.ndarray, text_blocks: List[TextBlock]
    ) -> List[LayoutRegion]:
        """Classify layout regions using position-based heuristics.

        Identifies title blocks (bottom-right corner heuristic) and updates
        text_blocks region_type in place. Table detection is mentioned but
        not yet implemented.

        Args:
            image: Input image for dimension calculations.
            text_blocks: List of TextBlock objects to classify. region_type
                attributes are modified in place.

        Returns:
            List of LayoutRegion objects for identified regions (currently
            only title blocks). Empty list if no regions identified.

        Note:
            - Title block heuristic: x > 60% width AND y > 70% height
            - This is simplified; production should use ISO/ANSI drawing standards
            - Table detection is not implemented (TODO)
        """
        layout_regions: List[LayoutRegion] = []

        if not text_blocks:
            return layout_regions

        height, width = image.shape[:2]

        # Heuristic: Title block is usually in bottom-right corner
        title_block_candidates = [
            b for b in text_blocks if b.bbox.x > width * 0.6 and b.bbox.y > height * 0.7
        ]

        if title_block_candidates:
            # Merge bboxes of candidates into single region
            title_bbox = self._merge_bboxes_simple(
                [b.bbox for b in title_block_candidates]
            )

            layout_regions.append(
                LayoutRegion(
                    region_id=generate_unique_id("REG"),
                    region_type="title_block",
                    bbox=title_bbox,
                    confidence=0.8,
                )
            )

            # Update text block types
            for block in title_block_candidates:
                block.region_type = "title_block"

        # TODO: Implement table detection using alignment analysis
        # Heuristic: Tables have high text density in grid pattern

        return layout_regions

    def _merge_bboxes_simple(self, bboxes: List[BoundingBox]) -> BoundingBox:
        """Merge multiple bounding boxes into a single encompassing box.

        Computes the minimal axis-aligned bounding box that contains all
        input bounding boxes.

        Args:
            bboxes: List of BoundingBox objects to merge.

        Returns:
            BoundingBox that encompasses all input boxes.

        Raises:
            ValueError: If bboxes list is empty.

        Note:
            This is a simple implementation that replaces the missing
            merge_bboxes utility function from geometry_utils.
        """
        if not bboxes:
            raise ValueError("Cannot merge empty list of bounding boxes")

        # Find min/max coordinates
        x_min = min(b.x for b in bboxes)
        y_min = min(b.y for b in bboxes)
        x_max = max(b.x + b.width for b in bboxes)
        y_max = max(b.y + b.height for b in bboxes)

        return BoundingBox(x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min)

    def _crop_region(
        self, image: np.ndarray, bbox: BoundingBox, padding: int = 5
    ) -> np.ndarray:
        """Crop a region from image with padding, handling boundary conditions.

        Args:
            image: Source image to crop from.
            bbox: Bounding box defining the region to crop.
            padding: Pixels to add around the bbox (default: 5).

        Returns:
            Cropped image region as numpy array.

        Note:
            Automatically clips coordinates to image boundaries.
        """
        x1 = max(0, bbox.x - padding)
        y1 = max(0, bbox.y - padding)
        x2 = min(image.shape[1], bbox.x + bbox.width + padding)
        y2 = min(image.shape[0], bbox.y + bbox.height + padding)

        return image[y1:y2, x1:x2]

    def _calculate_average_confidence(self, text_blocks: List[TextBlock]) -> float:
        """Calculate average confidence across text blocks.

        Args:
            text_blocks: List of TextBlock objects.

        Returns:
            Average confidence (0.0-1.0), or 0.0 if list is empty.
        """
        if not text_blocks:
            return 0.0
        return sum(b.confidence for b in text_blocks) / len(text_blocks)

    def _merge_improved_blocks(
        self, original_blocks: List[TextBlock], improved_blocks: List[TextBlock]
    ) -> List[TextBlock]:
        """Merge original and improved blocks, replacing by text_id.

        Uses dictionary lookup for O(1) replacement efficiency.

        Args:
            original_blocks: All text blocks from primary OCR.
            improved_blocks: Blocks that were reprocessed with fallback OCR.

        Returns:
            Merged list with improved blocks replacing their originals.
        """
        # Create dict of improved blocks by text_id for O(1) lookup
        improved_dict: Dict[str, TextBlock] = {
            block.text_id: block for block in improved_blocks
        }

        # Replace or keep original blocks
        merged_blocks: List[TextBlock] = []
        for block in original_blocks:
            if block.text_id in improved_dict:
                merged_blocks.append(improved_dict[block.text_id])
            elif block.confidence >= self.config.confidence_threshold:
                merged_blocks.append(block)
            # Skip blocks that were low-confidence but not improved

        return merged_blocks

    def detect_language(self, text_sample: str) -> str:
        """Detect language from text sample using langdetect library.

        Uses statistical language detection on text content. Requires langdetect
        package to be installed.

        Args:
            text_sample: Sample text string for language detection. Longer samples
                (100+ characters) provide more accurate results.

        Returns:
            ISO 639-1 language code (e.g., 'en', 'de', 'fr'). Returns 'en' if
            detection fails or text_sample is too short.

        Note:
            Detection accuracy improves with longer text samples. Very short
            samples may produce incorrect results.
        """
        try:
            from langdetect import detect

            lang = detect(text_sample)
            return lang
        except ImportError:
            logger.warning(
                "langdetect not installed. Install with: pip install langdetect"
            )
            return "en"  # Default to English
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "en"  # Default to English

    def cleanup(self) -> None:
        """Clean up OCR engine resources.

        Explicitly releases OCR engine instances to free memory. Useful for
        long-running processes or when processing is complete.

        Note:
            After calling cleanup(), the engines will be reinitialized on next use.
        """
        if self._primary_ocr is not None:
            del self._primary_ocr
            self._primary_ocr = None
            logger.debug("Primary OCR engine cleaned up")

        if self._fallback_ocr is not None:
            del self._fallback_ocr
            self._fallback_ocr = None
            logger.debug("Fallback OCR engine cleaned up")
