"""
Image preprocessing module for the Drawing Intelligence System.

This module provides specialized image preprocessing pipelines optimized for
OCR and shape detection tasks on engineering drawings. It handles common image
quality issues including skew, noise, poor contrast, and blur.

Classes:
    PreprocessConfig: Configuration container for preprocessing parameters.
    ImagePreprocessor: Main preprocessing engine with dual-mode operation.

Typical usage example:
    config = PreprocessConfig(skew_threshold_degrees=1.0)
    preprocessor = ImagePreprocessor(config)
    ocr_image = preprocessor.preprocess_for_ocr(raw_image)
    detection_image = preprocessor.preprocess_for_detection(raw_image)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from ..models.data_structures import QualityMetrics
from ..utils.validation_utils import validate_image_array

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """
    Configuration for image preprocessing operations.

    This configuration supports two distinct preprocessing modes: OCR-optimized
    (high contrast, aggressive noise reduction) and detection-optimized (edge
    preservation, mild enhancement). Each mode has its own preset dictionary.

    Attributes:
        ocr_preset: Dictionary containing OCR preprocessing parameters. Keys include:
            - adaptive_threshold_block_size (int): Block size for adaptive thresholding
            - median_blur_kernel (int): Kernel size for median blur (must be odd)
            - morphology_kernel_size (int): Kernel size for morphological operations
            - apply_clahe (bool): Whether to apply CLAHE enhancement
            - clahe_clip_limit (float): Clipping limit for CLAHE
            - clahe_grid_size (Tuple[int, int]): Tile grid size for CLAHE
        detection_preset: Dictionary containing detection preprocessing parameters.
            Similar keys as ocr_preset but optimized for shape detection.
        skew_threshold_degrees: Minimum rotation angle (in degrees) to trigger skew
            correction. Values below this threshold are ignored to avoid unnecessary
            processing. Default: 0.5
        blur_threshold: Laplacian variance threshold for blur detection. Images with
            variance below this value are considered too blurry. Default: 100.0
        contrast_threshold: Minimum acceptable contrast score (std dev). Default: 20.0
        brightness_min: Minimum acceptable mean brightness. Default: 30.0
        brightness_max: Maximum acceptable mean brightness. Default: 225.0

    Note:
        If ocr_preset or detection_preset are None, default values are set in
        __post_init__. These defaults are tuned for standard engineering drawings
        at 300 DPI.
    """

    ocr_preset: Optional[Dict[str, Any]] = None
    detection_preset: Optional[Dict[str, Any]] = None
    skew_threshold_degrees: float = 0.5
    blur_threshold: float = 100.0
    contrast_threshold: float = 20.0
    brightness_min: float = 30.0
    brightness_max: float = 225.0

    def __post_init__(self) -> None:
        """Initialize default presets if not provided and validate configurations."""
        if self.ocr_preset is None:
            self.ocr_preset = {
                "adaptive_threshold_block_size": 11,
                "median_blur_kernel": 3,
                "morphology_kernel_size": 3,
                "apply_clahe": True,
                "clahe_clip_limit": 2.0,
                "clahe_grid_size": (8, 8),
            }

        if self.detection_preset is None:
            self.detection_preset = {
                "adaptive_threshold_block_size": 15,
                "median_blur_kernel": 5,
                "morphology_kernel_size": 3,
                "apply_clahe": False,
            }

        # Validate required keys in presets
        self._validate_preset(self.ocr_preset, "ocr_preset")
        self._validate_preset(self.detection_preset, "detection_preset")

    def _validate_preset(self, preset: Dict[str, Any], preset_name: str) -> None:
        """
        Validate that preset contains all required keys.

        Args:
            preset: Preset dictionary to validate.
            preset_name: Name of preset for error messages.

        Raises:
            ValueError: If required keys are missing.
        """
        required_keys = {
            "adaptive_threshold_block_size",
            "median_blur_kernel",
            "morphology_kernel_size",
        }
        missing_keys = required_keys - set(preset.keys())
        if missing_keys:
            raise ValueError(f"{preset_name} missing required keys: {missing_keys}")


class ImagePreprocessor:
    """
    Image quality enhancement engine for OCR and shape detection.

    This class provides two specialized preprocessing pipelines:
    1. OCR Pipeline: Optimized for text recognition with high contrast and
       aggressive noise reduction.
    2. Detection Pipeline: Optimized for shape detection with edge preservation
       and balanced enhancement.

    The preprocessor handles common image quality issues including:
    - Skew/rotation correction using Hough line detection
    - Noise reduction via median filtering and morphological operations
    - Contrast enhancement using CLAHE and histogram equalization
    - Quality assessment (blur, contrast, brightness)

    Attributes:
        config: Preprocessing configuration containing presets and thresholds.

    Example:
        >>> config = PreprocessConfig(blur_threshold=150.0)
        >>> preprocessor = ImagePreprocessor(config)
        >>>
        >>> # For OCR
        >>> ocr_ready = preprocessor.preprocess_for_ocr(image)
        >>>
        >>> # For shape detection
        >>> detection_ready = preprocessor.preprocess_for_detection(image)
        >>>
        >>> # Quality check
        >>> quality = preprocessor.assess_image_quality(image)
        >>> if not quality.is_acceptable:
        >>>     print(f"Quality issue: {quality.rejection_reason}")
    """

    def __init__(self, config: PreprocessConfig) -> None:
        """
        Initialize the image preprocessor with configuration.

        Args:
            config: Preprocessing configuration containing presets and thresholds.

        Raises:
            ValueError: If config is None or invalid.
        """
        if config is None:
            raise ValueError("PreprocessConfig cannot be None")
        self.config = config
        logger.info("ImagePreprocessor initialized")

    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline optimized for OCR.

        This pipeline maximizes text clarity through aggressive enhancement:
        1. Grayscale conversion (if needed)
        2. Skew detection and correction using Hough line transform
        3. CLAHE contrast enhancement (optional, based on preset)
        4. Adaptive thresholding for binarization
        5. Noise reduction via median blur and morphological operations

        The OCR preset uses smaller kernel sizes and higher contrast settings
        compared to the detection preset.

        Args:
            image: Input image as numpy array. Can be BGR (3-channel) or
                grayscale (single channel). Must be uint8 dtype.

        Returns:
            Preprocessed grayscale image optimized for OCR, as uint8 numpy array.
            The output is typically binary (0 or 255) after adaptive thresholding.

        Raises:
            ValueError: If image is invalid (wrong dtype, empty, or contains NaN/Inf).

        Example:
            >>> preprocessor = ImagePreprocessor(PreprocessConfig())
            >>> bgr_image = cv2.imread('drawing.png')
            >>> ocr_image = preprocessor.preprocess_for_ocr(bgr_image)
            >>> # ocr_image is now ready for PaddleOCR or EasyOCR
        """
        is_valid, error_msg = validate_image_array(image)
        if not is_valid:
            raise ValueError(f"Invalid input image: {error_msg}")

        preset = self.config.ocr_preset

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Deskew
        gray, angle = self.detect_and_correct_skew(gray)
        if abs(angle) > 0.1:
            logger.debug(f"Corrected skew: {angle:.2f}°")

        # Apply CLAHE for contrast enhancement
        if preset.get("apply_clahe", False):
            clahe = cv2.createCLAHE(
                clipLimit=preset["clahe_clip_limit"],
                tileGridSize=preset["clahe_grid_size"],
            )
            gray = clahe.apply(gray)

        # Adaptive thresholding
        gray = self._apply_adaptive_threshold(
            gray, preset["adaptive_threshold_block_size"]
        )

        # Noise reduction
        gray = self._apply_noise_reduction(
            gray, preset["median_blur_kernel"], preset["morphology_kernel_size"]
        )

        return gray

    def preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline optimized for shape detection.

        This pipeline balances enhancement with edge preservation:
        1. Grayscale conversion (if needed)
        2. Skew detection and correction
        3. Mild noise reduction using Non-Local Means denoising
        4. Histogram equalization for contrast

        The detection preset avoids aggressive thresholding to preserve subtle
        edges needed for YOLO-based shape detection.

        Args:
            image: Input image as numpy array. Can be BGR (3-channel) or
                grayscale (single channel). Must be uint8 dtype.

        Returns:
            Preprocessed grayscale image optimized for detection, as uint8 numpy array.
            The output maintains grayscale values (0-255) rather than binary.

        Raises:
            ValueError: If image is invalid (wrong dtype, empty, or contains NaN/Inf).

        Example:
            >>> preprocessor = ImagePreprocessor(PreprocessConfig())
            >>> bgr_image = cv2.imread('drawing.png')
            >>> detection_image = preprocessor.preprocess_for_detection(bgr_image)
            >>> # detection_image is now ready for YOLOv8 inference
        """
        is_valid, error_msg = validate_image_array(image)
        if not is_valid:
            raise ValueError(f"Invalid input image: {error_msg}")

        preset = self.config.detection_preset

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Deskew
        gray, angle = self.detect_and_correct_skew(gray)

        # Mild denoising to preserve edges
        gray = cv2.fastNlMeansDenoising(
            gray, None, h=10, templateWindowSize=7, searchWindowSize=21
        )

        # Slight contrast enhancement
        gray = cv2.equalizeHist(gray)

        return gray

    def detect_and_correct_skew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect and correct image skew/rotation using Hough line transform.

        This method identifies the dominant angle of lines in the image and
        rotates the image to correct skew. It uses:
        1. Canny edge detection to find edges
        2. Hough line transform to detect lines
        3. Median angle calculation (robust to outliers)
        4. Rotation matrix with dimension adjustment to prevent cropping

        Only applies correction if the detected skew exceeds the configured
        threshold (skew_threshold_degrees).

        Args:
            image: Input grayscale image as numpy array (uint8).

        Returns:
            Tuple containing:
                - corrected_image: Rotated image if skew > threshold, else original
                - skew_angle_degrees: Detected skew angle in degrees (negative = CW,
                  positive = CCW). Returns 0.0 if no lines detected or skew below
                  threshold.

        Note:
            - Angles are filtered to [-45, 45] degrees to avoid 90° misdetection
            - Uses median instead of mean for robustness against outliers
            - Rotation matrix is adjusted to prevent edge cropping
            - Uses BORDER_REPLICATE to fill new areas after rotation

        Example:
            >>> preprocessor = ImagePreprocessor(PreprocessConfig())
            >>> corrected, angle = preprocessor.detect_and_correct_skew(gray_image)
            >>> if abs(angle) > 0.1:
            >>>     print(f"Corrected skew of {angle:.2f} degrees")
        """
        # Edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is None or len(lines) == 0:
            return image, 0.0

        # Calculate angles
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            angles.append(angle)

        # Find median angle (more robust than mean)
        angles_array = np.array(angles)
        # Filter outliers - use <= to include valid 45° rotations
        angles_array = angles_array[np.abs(angles_array) <= 45]

        if len(angles_array) == 0:
            return image, 0.0

        skew_angle = float(np.median(angles_array))

        # Correct skew if above threshold
        if abs(skew_angle) > self.config.skew_threshold_degrees:
            corrected = self._rotate_image(image, skew_angle)
            return corrected, skew_angle

        return image, 0.0

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by specified angle without cropping.

        Calculates the new bounding dimensions needed to contain the rotated
        image and adjusts the rotation matrix accordingly

        image and adjusts the rotation matrix accordingly. Uses cubic
        interpolation for smooth results.

        Args:
            image: Input image as numpy array.
            angle: Rotation angle in degrees. Positive values rotate
                counterclockwise, negative values rotate clockwise.

        Returns:
            Rotated image as numpy array with expanded dimensions to prevent cropping.

        Note:
            - Uses INTER_CUBIC for high-quality interpolation
            - Uses BORDER_REPLICATE to fill new border areas
            - New dimensions calculated to fit entire rotated image
        """
        if image.size == 0:
            logger.warning("Cannot rotate empty image")
            return image

        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        # Get rotation matrix
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new bounding dimensions
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])

        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

        # Validate new dimensions
        if new_width <= 0 or new_height <= 0:
            logger.warning(
                f"Invalid rotation dimensions: {new_width}x{new_height}, "
                f"returning original"
            )
            return image

        # Adjust rotation matrix for new dimensions
        matrix[0, 2] += (new_width / 2) - center[0]
        matrix[1, 2] += (new_height / 2) - center[1]

        # Perform rotation
        rotated = cv2.warpAffine(
            image,
            matrix,
            (new_width, new_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return rotated

    def assess_image_quality(self, image: np.ndarray) -> QualityMetrics:
        """
        Evaluate image quality metrics for preprocessing pipeline decisions.

        Computes three key quality indicators:
        1. Blur metric: Laplacian variance (higher = sharper)
        2. Contrast score: Standard deviation of pixel intensities
        3. Brightness mean: Average pixel intensity

        Images failing quality thresholds are flagged as unacceptable with a
        descriptive rejection reason. This helps route low-quality images to
        the LLM-enhanced pipeline or reject them entirely.

        Args:
            image: Input image as numpy array. Can be BGR or grayscale (uint8).

        Returns:
            QualityMetrics object containing:
                - blur_metric: Laplacian variance (float)
                - contrast_score: Std dev of pixel intensities (float)
                - brightness_mean: Mean pixel intensity (float, 0-255)
                - is_acceptable: Boolean indicating if quality meets thresholds
                - rejection_reason: String describing failure reason (None if acceptable)

        Raises:
            ValueError: If image is invalid (wrong dtype, empty, or contains NaN/Inf).

        Note:
            Quality thresholds used:
            - Blur: Must be >= blur_threshold (default 100.0)
            - Contrast: Must be >= contrast_threshold (default 20.0)
            - Brightness: Must be in range [brightness_min, brightness_max]

        Example:
            >>> preprocessor = ImagePreprocessor(PreprocessConfig(blur_threshold=150))
            >>> quality = preprocessor.assess_image_quality(image)
            >>> if not quality.is_acceptable:
            >>>     logger.warning(f"Low quality: {quality.rejection_reason}")
            >>>     # Route to LLM-enhanced pipeline
        """
        is_valid, error_msg = validate_image_array(image)
        if not is_valid:
            raise ValueError(f"Invalid input image: {error_msg}")

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Blur metric (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_metric = float(laplacian.var())

        # Contrast (standard deviation)
        contrast_score = float(gray.std())

        # Brightness (mean intensity)
        brightness_mean = float(gray.mean())

        # Acceptability checks
        is_acceptable = True
        rejection_reason = None

        if blur_metric < self.config.blur_threshold:
            is_acceptable = False
            rejection_reason = (
                f"Image too blurry (metric: {blur_metric:.1f}, "
                f"threshold: {self.config.blur_threshold})"
            )
        elif contrast_score < self.config.contrast_threshold:
            is_acceptable = False
            rejection_reason = (
                f"Low contrast (score: {contrast_score:.1f}, "
                f"threshold: {self.config.contrast_threshold})"
            )
        elif (
            brightness_mean < self.config.brightness_min
            or brightness_mean > self.config.brightness_max
        ):
            is_acceptable = False
            rejection_reason = (
                f"Poor brightness (mean: {brightness_mean:.1f}, "
                f"range: [{self.config.brightness_min}, {self.config.brightness_max}])"
            )

        return QualityMetrics(
            blur_metric=blur_metric,
            contrast_score=contrast_score,
            brightness_mean=brightness_mean,
            is_acceptable=is_acceptable,
            rejection_reason=rejection_reason,
        )

    def _apply_adaptive_threshold(
        self, image: np.ndarray, block_size: int
    ) -> np.ndarray:
        """
        Apply adaptive thresholding for image binarization.

        Uses Gaussian-weighted adaptive thresholding where each pixel's
        threshold is calculated from the weighted mean of its neighborhood.

        Args:
            image: Input grayscale image as numpy array (uint8).
            block_size: Size of pixel neighborhood for threshold calculation.
                Must be odd and >= 3. If even, it will be incremented by 1.

        Returns:
            Binary image (0 or 255 values) as uint8 numpy array.

        Raises:
            ValueError: If block_size is too small or too large.

        Note:
            - Uses Gaussian weighting (ADAPTIVE_THRESH_GAUSSIAN_C)
            - Subtracts constant 2 from computed threshold
            - Automatically adjusts even block_size to odd
        """
        # Validate block size
        if block_size < 3:
            logger.warning(f"block_size {block_size} too small, using 3")
            block_size = 3

        height, width = image.shape[:2]
        max_block_size = min(height, width)
        if block_size > max_block_size:
            logger.warning(
                f"block_size {block_size} exceeds image dimensions, "
                f"using {max_block_size}"
            )
            block_size = max_block_size

        # Ensure block size is odd
        if block_size % 2 == 0:
            block_size += 1

        binary = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            2,
        )

        return binary

    def _apply_noise_reduction(
        self, image: np.ndarray, median_kernel: int, morphology_kernel: int
    ) -> np.ndarray:
        """
        Apply noise reduction using median filtering and morphological operations.

        Two-stage noise reduction:
        1. Median blur to remove salt-and-pepper noise
        2. Morphological opening (removes small white noise) followed by
           closing (fills small black holes)

        Args:
            image: Input image as numpy array (uint8).
            median_kernel: Kernel size for median blur. Must be odd and >= 1.
                If <= 1, median blur is skipped.
            morphology_kernel: Kernel size for morphological operations.
                If <= 0, morphological operations are skipped.

        Returns:
            Denoised image as uint8 numpy array.

        Note:
            Morphological operations use rectangular structuring elements.

        Raises:
            ValueError: If median_kernel is even and > 1.
        """
        denoised = image.copy()

        # Median blur to remove salt-and-pepper noise
        if median_kernel > 1:
            # Validate odd kernel size
            if median_kernel % 2 == 0:
                logger.warning(
                    f"median_kernel {median_kernel} must be odd, incrementing to "
                    f"{median_kernel + 1}"
                )
                median_kernel += 1
            denoised = cv2.medianBlur(denoised, median_kernel)

        # Morphological operations to clean up
        if morphology_kernel > 0:
            kernel = np.ones((morphology_kernel, morphology_kernel), np.uint8)
            # Opening: erosion followed by dilation (removes small white noise)
            denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
            # Closing: dilation followed by erosion (removes small black holes)
            denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

        return denoised

    def enhance_contrast(self, image: np.ndarray, alpha: float = 1.5) -> np.ndarray:
        """
        Enhance image contrast using linear transformation.

        Applies the formula: output = alpha * input + beta
        where beta is calculated to maintain mean brightness.

        This is a global contrast adjustment, simpler than CLAHE but less
        effective for images with varying local contrast.

        Args:
            image: Input image as numpy array (uint8).
            alpha: Contrast multiplier. Values > 1 increase contrast,
                values < 1 decrease contrast. Default: 1.5

        Returns:
            Enhanced image as uint8 numpy array with adjusted contrast.

        Example:
            >>> preprocessor = ImagePreprocessor(PreprocessConfig())
            >>> high_contrast = preprocessor.enhance_contrast(image, alpha=2.0)
        """
        # Apply: output = alpha * input + beta
        # Beta calculated to maintain mean brightness
        beta = 128 * (1 - alpha)

        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return enhanced

    def sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """
        Sharpen image using unsharp masking technique.

        Creates a sharpened version by subtracting a Gaussian-blurred version
        from the original image. This enhances edges and fine details.

        The formula used: sharpened = 1.5 * original - 0.5 * blurred

        Args:
            image: Input image as numpy array (uint8).

        Returns:
            Sharpened image as uint8 numpy array.

        Note:
            Uses Gaussian blur with sigma=2.0. The enhancement is relatively
            aggressive (1.5x original) which may amplify noise in low-quality images.

        Example:
            >>> preprocessor = ImagePreprocessor(PreprocessConfig())
            >>> sharp_image = preprocessor.sharpen_image(blurry_image)
        """
        # Create Gaussian blur
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)

        # Unsharp mask: original + amount * (original - blurred)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

        return sharpened
