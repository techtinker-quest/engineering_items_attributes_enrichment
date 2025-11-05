"""
OCR pipeline module for the Drawing Intelligence System.

Dual-engine OCR with fallback: PaddleOCR (primary) + EasyOCR (fallback).
"""

import logging
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from ..models.data_structures import OCRResult, TextBlock, LayoutRegion
from ..utils.geometry_utils import BoundingBox
from ..utils.file_utils import generate_unique_id
from ..utils.error_handlers import OCRError

logger = logging.getLogger(__name__)


@dataclass
class OCRConfig:
    """
    Configuration for OCR pipeline.

    Attributes:
        primary_engine: Primary OCR engine ('paddleocr' or 'easyocr')
        fallback_engine: Fallback OCR engine
        confidence_threshold: Threshold for triggering fallback (default: 0.85)
        languages: List of languages to detect (default: ['en'])
        layout_analysis: Enable layout detection (default: True)
        use_gpu: Use GPU acceleration if available (default: True)
    """

    primary_engine: str = "paddleocr"
    fallback_engine: str = "easyocr"
    confidence_threshold: float = 0.85
    languages: List[str] = None
    layout_analysis: bool = True
    use_gpu: bool = True

    def __post_init__(self):
        if self.languages is None:
            self.languages = ["en"]


class OCRPipeline:
    """
    Extract text from images using dual OCR engines.

    Strategy:
    1. Primary OCR (PaddleOCR) processes entire image
    2. Low-confidence regions are re-processed with fallback (EasyOCR)
    3. Best results are kept
    """

    def __init__(self, config: OCRConfig):
        """
        Initialize OCR pipeline.

        Args:
            config: OCR configuration
        """
        self.config = config

        # Lazy loading of OCR engines
        self._primary_ocr = None
        self._fallback_ocr = None

        logger.info(
            f"OCRPipeline initialized (Primary: {config.primary_engine}, "
            f"Fallback: {config.fallback_engine})"
        )

    def extract_text(self, image: np.ndarray, language: str = "auto") -> OCRResult:
        """
        Extract text with bounding boxes using primary OCR.

        Args:
            image: Input image (grayscale or BGR)
            language: Language code or 'auto' for detection

        Returns:
            OCRResult with text blocks and layout info

        Raises:
            OCRError: If OCR extraction fails
        """
        try:
            # Detect language if auto
            if language == "auto":
                language = self.detect_language_from_image(image)

            # Primary OCR
            logger.debug(f"Running primary OCR ({self.config.primary_engine})")
            text_blocks = self._run_primary_ocr(image, language)

            # Layout analysis
            layout_regions = []
            if self.config.layout_analysis:
                layout_regions = self._classify_layout_regions(image, text_blocks)

            # Calculate average confidence
            if text_blocks:
                avg_confidence = sum(b.confidence for b in text_blocks) / len(
                    text_blocks
                )
            else:
                avg_confidence = 0.0

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
                improved_blocks = self.apply_fallback_ocr(image, low_conf_blocks)

                # Replace low-confidence blocks with improved ones
                text_blocks = [
                    b
                    for b in text_blocks
                    if b.confidence >= self.config.confidence_threshold
                ]
                text_blocks.extend(improved_blocks)

                # Recalculate average confidence
                if text_blocks:
                    avg_confidence = sum(b.confidence for b in text_blocks) / len(
                        text_blocks
                    )

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

        except Exception as e:
            raise OCRError(
                f"OCR extraction failed: {e}", ocr_engine=self.config.primary_engine
            )

    def _run_primary_ocr(self, image: np.ndarray, language: str) -> List[TextBlock]:
        """
        Run primary OCR engine (PaddleOCR).

        Args:
            image: Input image
            language: Language code

        Returns:
            List of TextBlock objects
        """
        if self._primary_ocr is None:
            self._initialize_primary_ocr()

        try:
            # PaddleOCR returns list of lines with [bbox, (text, confidence)]
            result = self._primary_ocr.ocr(image, cls=True)

            text_blocks = []

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
                                region_type="text",  # Will be refined by layout analysis
                            )
                            text_blocks.append(text_block)

            return text_blocks

        except Exception as e:
            logger.error(f"PaddleOCR failed: {e}")
            raise OCRError(f"PaddleOCR execution failed: {e}", ocr_engine="paddleocr")

    def apply_fallback_ocr(
        self, image: np.ndarray, low_confidence_blocks: List[TextBlock]
    ) -> List[TextBlock]:
        """
        Re-process low-confidence regions with fallback OCR.

        Args:
            image: Full image
            low_confidence_blocks: Blocks to re-process

        Returns:
            List of improved TextBlocks
        """
        if self._fallback_ocr is None:
            self._initialize_fallback_ocr()

        improved_blocks = []

        for block in low_confidence_blocks:
            try:
                # Crop region from image
                x, y, w, h = (
                    block.bbox.x,
                    block.bbox.y,
                    block.bbox.width,
                    block.bbox.height,
                )
                # Add padding
                padding = 5
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image.shape[1], x + w + padding)
                y2 = min(image.shape[0], y + h + padding)

                cropped = image[y1:y2, x1:x2]

                # Run EasyOCR on cropped region
                result = self._fallback_ocr.readtext(cropped)

                if result:
                    # Take first result (most confident)
                    bbox_points, text, confidence = result[0]

                    # Use original bbox coordinates
                    # (EasyOCR bbox is relative to crop)

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

        return improved_blocks

    def _initialize_primary_ocr(self):
        """Lazy initialization of primary OCR engine."""
        try:
            from paddleocr import PaddleOCR

            self._primary_ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",  # Will be overridden per call
                use_gpu=self.config.use_gpu,
                show_log=False,
            )
            logger.info("PaddleOCR initialized")

        except ImportError:
            raise OCRError(
                "PaddleOCR not installed. Install with: pip install paddleocr",
                ocr_engine="paddleocr",
            )
        except Exception as e:
            raise OCRError(
                f"Failed to initialize PaddleOCR: {e}", ocr_engine="paddleocr"
            )

    def _initialize_fallback_ocr(self):
        """Lazy initialization of fallback OCR engine."""
        try:
            import easyocr

            self._fallback_ocr = easyocr.Reader(
                self.config.languages, gpu=self.config.use_gpu
            )
            logger.info("EasyOCR initialized")

        except ImportError:
            raise OCRError(
                "EasyOCR not installed. Install with: pip install easyocr",
                ocr_engine="easyocr",
            )
        except Exception as e:
            raise OCRError(f"Failed to initialize EasyOCR: {e}", ocr_engine="easyocr")

    def _points_to_bbox(self, points: List[List[float]]) -> BoundingBox:
        """
        Convert OCR bbox points to BoundingBox.

        Args:
            points: List of [x, y] coordinates (usually 4 corners)

        Returns:
            BoundingBox object
        """
        # Extract x and y coordinates
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        # Get bounding rectangle
        x_min = int(min(xs))
        y_min = int(min(ys))
        x_max = int(max(xs))
        y_max = int(max(ys))

        return BoundingBox(x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min)

    def _normalize_text(self, text: str) -> str:
        """
        Normalize OCR text.

        Args:
            text: Raw OCR text

        Returns:
            Normalized text
        """
        from ..utils.text_utils import normalize_whitespace

        # Normalize whitespace
        text = normalize_whitespace(text)

        # Remove common OCR artifacts
        text = text.replace("|", "I")  # Pipe often misread

        return text

    def _classify_layout_regions(
        self, image: np.ndarray, text_blocks: List[TextBlock]
    ) -> List[LayoutRegion]:
        """
        Classify layout regions (title_block, table, text, figure).

        Uses heuristics based on position and density.

        Args:
            image: Input image
            text_blocks: Extracted text blocks

        Returns:
            List of LayoutRegion objects
        """
        layout_regions = []

        if not text_blocks:
            return layout_regions

        height, width = image.shape[:2]

        # Heuristic: Title block is usually in bottom-right corner
        title_block_candidates = [
            b for b in text_blocks if b.bbox.x > width * 0.6 and b.bbox.y > height * 0.7
        ]

        if title_block_candidates:
            # Merge bboxes of candidates
            from ..utils.geometry_utils import merge_bboxes

            title_bbox = merge_bboxes([b.bbox for b in title_block_candidates])

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

        # Heuristic: Tables have high text density in grid pattern
        # (Simplified - full implementation would use more sophisticated detection)

        # Remaining blocks are general text
        for block in text_blocks:
            if block.region_type == "text":  # Not yet classified
                block.region_type = "text"

        return layout_regions

    def detect_language(self, text_sample: str) -> str:
        """
        Detect language from text sample.

        Args:
            text_sample: Sample text for detection

        Returns:
            Language code (e.g., 'en', 'de', 'fr')
        """
        try:
            from langdetect import detect

            lang = detect(text_sample)
            return lang
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "en"  # Default to English

    def detect_language_from_image(self, image: np.ndarray) -> str:
        """
        Detect language by doing quick OCR and analyzing text.

        Args:
            image: Input image

        Returns:
            Language code
        """
        # For now, default to English
        # Full implementation would run quick OCR and detect from results
        return "en"
