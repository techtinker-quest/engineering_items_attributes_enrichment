"""
Geometry utilities for the Drawing Intelligence System.

Provides functions for bounding box operations and geometric calculations.
"""

import math
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class BoundingBox:
    """
    Bounding box with integer pixel coordinates.

    Attributes:
        x: Left coordinate
        y: Top coordinate
        width: Width in pixels
        height: Height in pixels
    """

    x: int
    y: int
    width: int
    height: int

    def center(self) -> Tuple[int, int]:
        """
        Calculate center point of bounding box.

        Returns:
            Tuple of (center_x, center_y)
        """
        center_x = self.x + self.width // 2
        center_y = self.y + self.height // 2
        return (center_x, center_y)

    def area(self) -> int:
        """
        Calculate area of bounding box.

        Returns:
            Area in square pixels
        """
        return self.width * self.height

    def iou(self, other: "BoundingBox") -> float:
        """
        Calculate Intersection over Union with another bounding box.

        Args:
            other: Another BoundingBox

        Returns:
            IoU score (0.0 to 1.0)
        """
        return calculate_iou(self, other)

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)

    def to_corners(self) -> Tuple[int, int, int, int]:
        """Convert to (x1, y1, x2, y2) corner coordinates."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class NormalizedBBox:
    """
    Normalized bounding box with coordinates in [0, 1] range.

    Attributes:
        x_center: Normalized center x (0.0 to 1.0)
        y_center: Normalized center y (0.0 to 1.0)
        width: Normalized width (0.0 to 1.0)
        height: Normalized height (0.0 to 1.0)
    """

    x_center: float
    y_center: float
    width: float
    height: float

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Convert to (x_center, y_center, width, height) tuple."""
        return (self.x_center, self.y_center, self.width, self.height)


def calculate_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.

    Args:
        bbox1: First bounding box
        bbox2: Second bounding box

    Returns:
        IoU score (0.0 to 1.0)
    """
    # Get corner coordinates
    x1_1, y1_1, x2_1, y2_1 = bbox1.to_corners()
    x1_2, y1_2, x2_2, y2_2 = bbox2.to_corners()

    # Calculate intersection rectangle
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    # Check if there is intersection
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    # Calculate intersection area
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # Calculate union area
    area1 = bbox1.area()
    area2 = bbox2.area()
    union_area = area1 + area2 - intersection_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    iou = intersection_area / union_area
    return iou


def calculate_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        point1: First point (x, y)
        point2: Second point (x, y)

    Returns:
        Distance in pixels
    """
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    distance = math.sqrt(dx * dx + dy * dy)
    return distance


def bbox_contains(outer: BoundingBox, inner: BoundingBox) -> bool:
    """
    Check if inner bounding box is completely inside outer bounding box.

    Args:
        outer: Outer bounding box
        inner: Inner bounding box

    Returns:
        True if inner is completely inside outer
    """
    x1_outer, y1_outer, x2_outer, y2_outer = outer.to_corners()
    x1_inner, y1_inner, x2_inner, y2_inner = inner.to_corners()

    return (
        x1_inner >= x1_outer
        and y1_inner >= y1_outer
        and x2_inner <= x2_outer
        and y2_inner <= y2_outer
    )


def bbox_overlaps(bbox1: BoundingBox, bbox2: BoundingBox) -> bool:
    """
    Check if two bounding boxes overlap.

    Args:
        bbox1: First bounding box
        bbox2: Second bounding box

    Returns:
        True if boxes overlap
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1.to_corners()
    x1_2, y1_2, x2_2, y2_2 = bbox2.to_corners()

    # Check if one box is to the left of the other
    if x2_1 < x1_2 or x2_2 < x1_1:
        return False

    # Check if one box is above the other
    if y2_1 < y1_2 or y2_2 < y1_1:
        return False

    return True


def expand_bbox(bbox: BoundingBox, pixels: int) -> BoundingBox:
    """
    Expand bounding box by N pixels in all directions.

    Args:
        bbox: Original bounding box
        pixels: Number of pixels to expand

    Returns:
        Expanded bounding box
    """
    return BoundingBox(
        x=max(0, bbox.x - pixels),
        y=max(0, bbox.y - pixels),
        width=bbox.width + 2 * pixels,
        height=bbox.height + 2 * pixels,
    )


def merge_bboxes(bboxes: List[BoundingBox]) -> BoundingBox:
    """
    Merge multiple bounding boxes into single bounding box.

    Args:
        bboxes: List of bounding boxes

    Returns:
        Merged bounding box that contains all input boxes

    Raises:
        ValueError: If bboxes list is empty
    """
    if not bboxes:
        raise ValueError("Cannot merge empty list of bounding boxes")

    # Get all corners
    corners = [bbox.to_corners() for bbox in bboxes]

    # Find min/max coordinates
    x1_min = min(c[0] for c in corners)
    y1_min = min(c[1] for c in corners)
    x2_max = max(c[2] for c in corners)
    y2_max = max(c[3] for c in corners)

    return BoundingBox(
        x=x1_min, y=y1_min, width=x2_max - x1_min, height=y2_max - y1_min
    )


def calculate_bbox_center(bbox: BoundingBox) -> Tuple[int, int]:
    """
    Calculate center point of bounding box.

    Args:
        bbox: Bounding box

    Returns:
        Tuple of (center_x, center_y)
    """
    return bbox.center()


def normalize_bbox(
    bbox: BoundingBox, image_width: int, image_height: int
) -> NormalizedBBox:
    """
    Normalize bounding box to [0, 1] coordinates relative to image size.

    Args:
        bbox: Bounding box in pixel coordinates
        image_width: Image width
        image_height: Image height

    Returns:
        Normalized bounding box

    Raises:
        ValueError: If image dimensions are zero
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")

    # Calculate center in pixels
    center_x_px = bbox.x + bbox.width / 2.0
    center_y_px = bbox.y + bbox.height / 2.0

    # Normalize
    x_center = center_x_px / image_width
    y_center = center_y_px / image_height
    width = bbox.width / image_width
    height = bbox.height / image_height

    return NormalizedBBox(
        x_center=x_center, y_center=y_center, width=width, height=height
    )


def denormalize_bbox(
    norm_bbox: NormalizedBBox, image_width: int, image_height: int
) -> BoundingBox:
    """
    Convert normalized bounding box back to pixel coordinates.

    Args:
        norm_bbox: Normalized bounding box
        image_width: Image width
        image_height: Image height

    Returns:
        Bounding box in pixel coordinates

    Raises:
        ValueError: If image dimensions are zero
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")

    # Convert center to pixels
    center_x_px = norm_bbox.x_center * image_width
    center_y_px = norm_bbox.y_center * image_height

    # Convert size to pixels
    width_px = norm_bbox.width * image_width
    height_px = norm_bbox.height * image_height

    # Calculate top-left corner
    x = int(center_x_px - width_px / 2.0)
    y = int(center_y_px - height_px / 2.0)

    return BoundingBox(
        x=max(0, x), y=max(0, y), width=int(width_px), height=int(height_px)
    )


def clip_bbox_to_image(
    bbox: BoundingBox, image_width: int, image_height: int
) -> BoundingBox:
    """
    Clip bounding box to image boundaries.

    Args:
        bbox: Original bounding box
        image_width: Image width
        image_height: Image height

    Returns:
        Clipped bounding box within image bounds
    """
    x1, y1, x2, y2 = bbox.to_corners()

    # Clip to boundaries
    x1 = max(0, min(x1, image_width))
    y1 = max(0, min(y1, image_height))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))

    return BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


def bbox_intersection(
    bbox1: BoundingBox, bbox2: BoundingBox
) -> Tuple[int, int, int, int]:
    """
    Calculate intersection rectangle of two bounding boxes.

    Args:
        bbox1: First bounding box
        bbox2: Second bounding box

    Returns:
        Tuple of (x, y, width, height) for intersection, or (0, 0, 0, 0) if no intersection
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1.to_corners()
    x1_2, y1_2, x2_2, y2_2 = bbox2.to_corners()

    # Calculate intersection
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    # Check if there is intersection
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return (0, 0, 0, 0)

    return (x1_inter, y1_inter, x2_inter - x1_inter, y2_inter - y1_inter)


def calculate_aspect_ratio(bbox: BoundingBox) -> float:
    """
    Calculate aspect ratio of bounding box.

    Args:
        bbox: Bounding box

    Returns:
        Aspect ratio (width / height)

    Raises:
        ValueError: If height is zero
    """
    if bbox.height == 0:
        raise ValueError("Cannot calculate aspect ratio: height is zero")

    return bbox.width / bbox.height


def point_in_bbox(point: Tuple[int, int], bbox: BoundingBox) -> bool:
    """
    Check if a point is inside a bounding box.

    Args:
        point: Point coordinates (x, y)
        bbox: Bounding box

    Returns:
        True if point is inside bbox
    """
    x, y = point
    x1, y1, x2, y2 = bbox.to_corners()

    return x1 <= x <= x2 and y1 <= y <= y2


def bbox_distance(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """
    Calculate minimum distance between two bounding boxes.

    If boxes overlap, returns 0.

    Args:
        bbox1: First bounding box
        bbox2: Second bounding box

    Returns:
        Minimum distance in pixels
    """
    if bbox_overlaps(bbox1, bbox2):
        return 0.0

    # Get centers
    center1 = bbox1.center()
    center2 = bbox2.center()

    # Calculate distance between centers
    return calculate_distance(center1, center2)


def scale_bbox(bbox: BoundingBox, scale_factor: float) -> BoundingBox:
    """
    Scale bounding box by a factor around its center.

    Args:
        bbox: Original bounding box
        scale_factor: Scale factor (e.g., 1.5 for 150%)

    Returns:
        Scaled bounding box
    """
    center_x, center_y = bbox.center()

    new_width = int(bbox.width * scale_factor)
    new_height = int(bbox.height * scale_factor)

    new_x = center_x - new_width // 2
    new_y = center_y - new_height // 2

    return BoundingBox(
        x=max(0, new_x), y=max(0, new_y), width=new_width, height=new_height
    )
