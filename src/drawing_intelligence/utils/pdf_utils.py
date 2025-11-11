"""
Utility functions for PDF processing operations.

This module contains reusable helper functions for coordinate conversion,
color space transformation, and spatial analysis operations used across
PDF processing modules.

Location: src/drawing_intelligence/utils/pdf_utils.py

Functions:
    convert_rgb_to_bgr: Convert RGB image to BGR format for OpenCV.
    convert_pdf_points_to_pixels: Convert PDF point coordinates to pixels.
    normalize_bbox: Normalize bounding box coordinates to 0-1 range.
    calculate_bbox_area: Calculate area of a bounding box.
    bbox_iou: Calculate Intersection over Union between two bounding boxes.
    validate_bbox: Validate bounding box is well-formed and within bounds.
    clip_bbox_to_image: Clip bounding box coordinates to image boundaries.
"""

from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


def convert_rgb_to_bgr(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert RGB image to BGR format for OpenCV compatibility.

    OpenCV uses BGR color order by default, while most image sources
    (including PyMuPDF) use RGB. This function performs the conversion.

    Args:
        image: RGB image array with shape (height, width, 3) and dtype uint8.

    Returns:
        BGR image array with same shape and dtype.

    Example:
        >>> rgb_image = load_rgb_image()
        >>> bgr_image = convert_rgb_to_bgr(rgb_image)
        >>> cv2.imshow("Image", bgr_image)  # OpenCV expects BGR
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def convert_pdf_points_to_pixels(
    points: Tuple[float, float, float, float], dpi: int
) -> Tuple[int, int, int, int]:
    """Convert PDF point coordinates to pixel coordinates at specified DPI.

    PDF uses points as the base unit where 1 point = 1/72 inch. This function
    converts point coordinates to pixels based on the target DPI.

    Args:
        points: Coordinates in PDF points as tuple (x0, y0, x1, y1).
        dpi: Target dots per inch for conversion. Standard values are
            72 (screen), 150 (draft), 300 (standard print), 600 (high quality).

    Returns:
        Coordinates in pixels as tuple (x0, y0, x1, y1) with integer values.

    Note:
        Conversion formula: pixels = points * (dpi / 72.0)

    Example:
        >>> pdf_bbox = (0.0, 0.0, 612.0, 792.0)  # Letter size in points
        >>> pixel_bbox = convert_pdf_points_to_pixels(pdf_bbox, 300)
        >>> print(pixel_bbox)  # (0, 0, 2550, 3300) at 300 DPI
    """
    scale = dpi / 72.0
    return (
        int(points[0] * scale),
        int(points[1] * scale),
        int(points[2] * scale),
        int(points[3] * scale),
    )


def normalize_bbox(
    bbox: Tuple[int, int, int, int], image_width: int, image_height: int
) -> Tuple[float, float, float, float]:
    """Normalize bounding box coordinates to 0-1 range.

    Converts absolute pixel coordinates to normalized coordinates relative
    to image dimensions. Useful for scale-invariant representations.

    Args:
        bbox: Bounding box in pixels as (x0, y0, x1, y1).
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.

    Returns:
        Normalized bounding box as (x0_norm, y0_norm, x1_norm, y1_norm)
        where all values are in range [0.0, 1.0].

    Raises:
        ValueError: If image dimensions are zero or negative.

    Example:
        >>> bbox = (100, 200, 300, 400)
        >>> normalized = normalize_bbox(bbox, 800, 600)
        >>> print(normalized)  # (0.125, 0.333, 0.375, 0.667)
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"Invalid image dimensions: {image_width}x{image_height}. "
            "Dimensions must be positive."
        )

    return (
        bbox[0] / image_width,
        bbox[1] / image_height,
        bbox[2] / image_width,
        bbox[3] / image_height,
    )


def calculate_bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    """Calculate area of a bounding box in square pixels.

    Args:
        bbox: Bounding box as (x0, y0, x1, y1) where (x0, y0) is top-left
            and (x1, y1) is bottom-right corner.

    Returns:
        Area in square pixels. Returns 0 for invalid bounding boxes where
        x1 <= x0 or y1 <= y0.

    Example:
        >>> bbox = (10, 20, 50, 60)
        >>> area = calculate_bbox_area(bbox)
        >>> print(area)  # 1600 (40 * 40)
    """
    width = max(0, bbox[2] - bbox[0])
    height = max(0, bbox[3] - bbox[1])
    return width * height


def bbox_iou(
    bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.

    IoU is a metric for measuring overlap between two bounding boxes,
    commonly used in object detection and spatial analysis.

    Args:
        bbox1: First bounding box as (x0, y0, x1, y1).
        bbox2: Second bounding box as (x0, y0, x1, y1).

    Returns:
        IoU value in range [0.0, 1.0] where:
            - 0.0 = no overlap
            - 1.0 = perfect overlap (identical boxes)

    Note:
        Returns 0.0 if either bounding box has zero area.

    Example:
        >>> box1 = (0, 0, 100, 100)
        >>> box2 = (50, 50, 150, 150)
        >>> iou = bbox_iou(box1, box2)
        >>> print(f"IoU: {iou:.2f}")  # IoU: 0.14 (2500 / 17500)
    """
    # Calculate intersection coordinates
    x0_inter = max(bbox1[0], bbox2[0])
    y0_inter = max(bbox1[1], bbox2[1])
    x1_inter = min(bbox1[2], bbox2[2])
    y1_inter = min(bbox1[3], bbox2[3])

    # Calculate intersection area
    inter_width = max(0, x1_inter - x0_inter)
    inter_height = max(0, y1_inter - y0_inter)
    intersection_area = inter_width * inter_height

    # Calculate union area
    area1 = calculate_bbox_area(bbox1)
    area2 = calculate_bbox_area(bbox2)
    union_area = area1 + area2 - intersection_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def validate_bbox(
    bbox: Tuple[int, int, int, int], max_width: int, max_height: int
) -> bool:
    """Validate that bounding box is within image bounds and well-formed.

    Args:
        bbox: Bounding box as (x0, y0, x1, y1).
        max_width: Maximum allowed width (image width).
        max_height: Maximum allowed height (image height).

    Returns:
        True if bounding box is valid, False otherwise.

    Note:
        A valid bounding box must satisfy:
            - x0 < x1 and y0 < y1 (positive dimensions)
            - All coordinates >= 0
            - x1 <= max_width and y1 <= max_height (within bounds)

    Example:
        >>> bbox = (10, 20, 100, 80)
        >>> is_valid = validate_bbox(bbox, 200, 150)
        >>> print(is_valid)  # True
    """
    x0, y0, x1, y1 = bbox

    # Check positive dimensions
    if x1 <= x0 or y1 <= y0:
        return False

    # Check non-negative coordinates
    if x0 < 0 or y0 < 0:
        return False

    # Check within bounds
    if x1 > max_width or y1 > max_height:
        return False

    return True


def clip_bbox_to_image(
    bbox: Tuple[int, int, int, int], image_width: int, image_height: int
) -> Tuple[int, int, int, int]:
    """Clip bounding box coordinates to image boundaries.

    Ensures bounding box stays within image dimensions by clamping
    coordinates to valid ranges.

    Args:
        bbox: Bounding box as (x0, y0, x1, y1).
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.

    Returns:
        Clipped bounding box with all coordinates within [0, width/height].

    Example:
        >>> bbox = (-10, 50, 1000, 200)
        >>> clipped = clip_bbox_to_image(bbox, 800, 600)
        >>> print(clipped)  # (0, 50, 800, 200)
    """
    return (
        max(0, min(bbox[0], image_width)),
        max(0, min(bbox[1], image_height)),
        max(0, min(bbox[2], image_width)),
        max(0, min(bbox[3], image_height)),
    )
