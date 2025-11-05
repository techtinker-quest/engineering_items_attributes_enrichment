"""
Image preprocessing module for the Drawing Intelligence System.

Enhances image quality for optimal OCR and shape detection.
"""

import logging
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import cv2

from ..models.data_structures import QualityMetrics
from ..utils.validation_utils import validate_image_array

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """
    Configuration for image preprocessing.

    Attributes:
        ocr_preset: Settings optimized for OCR
        detection_preset: Settings optimized for shape detection
        skew_threshold_degrees: Minimum angle for skew correction (default: 0.5)
        blur_threshold: Threshold for blur detection (default: 100.0)
    """

    # OCR preset: high contrast for text clarity
    ocr_preset: dict = None
    # Detection preset: balanced for edge preservation
    detection_preset: dict = None
    skew_threshold_degrees: float = 0.5
    blur_threshold: float = 100.0

    def __post_init__(self):
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


class ImagePreprocessor:
    """
    Enhance image quality for OCR and shape detection.

    Provides two preprocessing variants:
    - OCR: High contrast for text clarity
    - Detection: Balanced for edge preservation
    """

    def __init__(self, config: PreprocessConfig):
        """
        Initialize image preprocessor.

        Args:
            config: Preprocessing configuration
        """
        self.config = config
        logger.info("ImagePreprocessor initialized")

    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing optimized for OCR.

        Pipeline:
        1. Convert to grayscale
        2. Deskew if needed
        3. Apply CLAHE (optional)
        4. Adaptive thresholding
        5. Noise reduction

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            Preprocessed grayscale image optimized for OCR
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
        if preset["apply_clahe"]:
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
        Apply preprocessing optimized for shape detection.

        Pipeline:
        1. Convert to grayscale (if needed)
        2. Deskew if needed
        3. Mild noise reduction
        4. Edge preservation

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            Preprocessed image optimized for detection
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
        Detect and correct image skew/rotation.

        Uses Hough line transform to detect dominant angles.

        Args:
            image: Input grayscale image

        Returns:
            Tuple of (corrected_image, skew_angle_degrees)
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
        angles = np.array(angles)
        # Filter outliers
        angles = angles[np.abs(angles) < 45]

        if len(angles) == 0:
            return image, 0.0

        skew_angle = np.median(angles)

        # Correct skew if above threshold
        if abs(skew_angle) > self.config.skew_threshold_degrees:
            corrected = self._rotate_image(image, skew_angle)
            return corrected, skew_angle

        return image, 0.0

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by specified angle.

        Args:
            image: Input image
            angle: Rotation angle in degrees (positive = counterclockwise)

        Returns:
            Rotated image
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        # Get rotation matrix
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new bounding dimensions
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])

        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

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
        Evaluate image quality metrics.

        Assesses:
        - Blur (Laplacian variance)
        - Contrast (standard deviation)
        - Brightness (mean intensity)

        Args:
            image: Input image

        Returns:
            QualityMetrics object
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
        blur_metric = laplacian.var()

        # Contrast (standard deviation)
        contrast_score = gray.std()

        # Brightness (mean intensity)
        brightness_mean = gray.mean()

        # Acceptability checks
        is_acceptable = True
        rejection_reason = None

        if blur_metric < self.config.blur_threshold:
            is_acceptable = False
            rejection_reason = f"Image too blurry (metric: {blur_metric:.1f})"
        elif contrast_score < 20:
            is_acceptable = False
            rejection_reason = f"Low contrast (score: {contrast_score:.1f})"
        elif brightness_mean < 30 or brightness_mean > 225:
            is_acceptable = False
            rejection_reason = f"Poor brightness (mean: {brightness_mean:.1f})"

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
        Apply adaptive thresholding for binarization.

        Args:
            image: Input grayscale image
            block_size: Size of pixel neighborhood (must be odd)

        Returns:
            Binary image
        """
        # Ensure block size is odd
        if block_size % 2 == 0:
            block_size += 1

        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2
        )

        return binary

    def _apply_noise_reduction(
        self, image: np.ndarray, median_kernel: int, morphology_kernel: int
    ) -> np.ndarray:
        """
        Apply noise reduction using median blur and morphological operations.

        Args:
            image: Input image
            median_kernel: Kernel size for median blur
            morphology_kernel: Kernel size for morphology

        Returns:
            Denoised image
        """
        # Median blur to remove salt-and-pepper noise
        if median_kernel > 1:
            denoised = cv2.medianBlur(image, median_kernel)
        else:
            denoised = image.copy()

        # Morphological operations to clean up
        if morphology_kernel > 0:
            kernel = np.ones((morphology_kernel, morphology_kernel), np.uint8)
            # Opening: erosion followed by dilation (removes small white noise)
            denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
            # Closing: dilation followed by erosion (removes small black holes)
            denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

        return denoised

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization.

        Args:
            image: Input grayscale image

        Returns:
            Enhanced image
        """
        preset = self.config.ocr_preset
        clahe = cv2.createCLAHE(
            clipLimit=preset["clahe_clip_limit"], tileGridSize=preset["clahe_grid_size"]
        )
        return clahe.apply(image)

    def enhance_contrast(self, image: np.ndarray, alpha: float = 1.5) -> np.ndarray:
        """
        Enhance image contrast using simple linear transformation.

        Args:
            image: Input image
            alpha: Contrast multiplier (>1 increases contrast)

        Returns:
            Enhanced image
        """
        # Apply: output = alpha * input + beta
        # Beta calculated to maintain mean brightness
        beta = 128 * (1 - alpha)

        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return enhanced

    def sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """
        Sharpen image using unsharp masking.

        Args:
            image: Input image

        Returns:
            Sharpened image
        """
        # Create Gaussian blur
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)

        # Unsharp mask: original + amount * (original - blurred)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

        return sharpened
