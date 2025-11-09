"""
Geometry utilities for the Drawing Intelligence System.

This module provides dataclasses and utility functions for working with bounding boxes
in both pixel and normalized coordinate systems. Key functionality includes:
- Bounding box operations (IoU, overlap detection, containment checks)
- Coordinate transformations (normalization, denormalization)
- Geometric calculations (distance, area, aspect ratio)
- Spatial manipulations (scaling, expanding, merging, clipping)

The module supports two coordinate systems:
- BoundingBox: Integer pixel coordinates (x, y, width, height)
- NormalizedBBox: Normalized [0,1] coordinates (x_center, y_center, width, height)

Typical usage:
    from drawing_intelligence.utils.geometry_utils import BoundingBox, calculate_iou

    bbox1 = BoundingBox(x=10, y=20, width=100, height=50)
    bbox2 = BoundingBox(x=50, y=30, width=100, height=50)
    overlap = calculate_iou(bbox1, bbox2)
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class BoundingBox:
    """
    Bounding box with integer pixel coordinates.

    Uses top-left corner (x, y) and dimensions (width, height) representation.
    All coordinates must be non-negative integers.

    Attributes:
        x: Left coordinate (pixels from left edge)
        y: Top coordinate (pixels from top edge)
        width: Width in pixels (must be >= 0)
        height: Height in pixels (must be >= 0)

    Raises:
        ValueError: If width or height is negative.

    Example:
        >>> bbox = BoundingBox(x=10, y=20, width=100, height=50)
        >>> bbox.center()
        (60.0, 45.0)
        >>> bbox.area()
        5000
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """Validate bounding box dimensions."""
        if self.width < 0:
            raise ValueError(f"Width must be non-negative, got {self.width}")
        if self.height < 0:
            raise ValueError(f"Height must be non-negative, got {self.height}")

    def center(self) -> Tuple[float, float]:
        """
        Calculate center point of bounding box with sub-pixel precision.

        Returns:
            Tuple of (center_x, center_y) as floats

        Example:
            >>> bbox = BoundingBox(x=10, y=20, width=11, height=11)
            >>> bbox.center()
            (15.5, 25.5)
        """
        center_x = self.x + self.width / 2.0
        center_y = self.y + self.height / 2.0
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
        """
        Convert to (x, y, width, height) tuple.

        Returns:
            Tuple of (x, y, width, height)
        """
        return (self.x, self.y, self.width, self.height)

    def to_corners(self) -> Tuple[int, int, int, int]:
        """
        Convert to (x1, y1, x2, y2) corner coordinates.

        Returns:
            Tuple of (x1, y1, x2, y2) representing top-left and bottom-right corners
        """
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass(frozen=True)
class NormalizedBBox:
    """
    Normalized bounding box with coordinates in [0, 1] range.

    Uses center-based representation with normalized coordinates relative to
    image dimensions. Commonly used for ML model input/output.

    Attributes:
        x_center: Normalized center x coordinate (0.0 to 1.0)
        y_center: Normalized center y coordinate (0.0 to 1.0)
        width: Normalized width (0.0 to 1.0)
        height: Normalized height (0.0 to 1.0)

    Raises:
        ValueError: If any coordinate is outside [0, 1] range.

    Example:
        >>> norm_bbox = NormalizedBBox(x_center=0.5, y_center=0.5, width=0.2, height=0.1)
        >>> norm_bbox.to_tuple()
        (0.5, 0.5, 0.2, 0.1)
    """

    x_center: float
    y_center: float
    width: float
    height: float

    def __post_init__(self) -> None:
        """Validate normalized coordinates are in [0, 1] range."""
        if not (0.0 <= self.x_center <= 1.0):
            raise ValueError(f"x_center must be in [0, 1] range, got {self.x_center}")
        if not (0.0 <= self.y_center <= 1.0):
            raise ValueError(f"y_center must be in [0, 1] range, got {self.y_center}")
        if not (0.0 <= self.width <= 1.0):
            raise ValueError(f"width must be in [0, 1] range, got {self.width}")
        if not (0.0 <= self.height <= 1.0):
            raise ValueError(f"height must be in [0, 1] range, got {self.height}")

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """
        Convert to (x_center, y_center, width, height) tuple.

        Returns:
            Tuple of (x_center, y_center, width, height)
        """
        return (self.x_center, self.y_center, self.width, self.height)


def calculate_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.

    Args:
        bbox1: First bounding box
        bbox2: Second bounding box

    Returns:
        IoU score (0.0 to 1.0)

    Example:
        >>> bbox1 = BoundingBox(x=0, y=0, width=10, height=10)
        >>> bbox2 = BoundingBox(x=5, y=5, width=10, height=10)
        >>> calculate_iou(bbox1, bbox2)
        0.25
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

    Example:
        >>> calculate_distance((0, 0), (3, 4))
        5.0
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

    Example:
        >>> outer = BoundingBox(x=0, y=0, width=100, height=100)
        >>> inner = BoundingBox(x=10, y=10, width=20, height=20)
        >>> bbox_contains(outer, inner)
        True
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

    Example:
        >>> bbox1 = BoundingBox(x=0, y=0, width=10, height=10)
        >>> bbox2 = BoundingBox(x=5, y=5, width=10, height=10)
        >>> bbox_overlaps(bbox1, bbox2)
        True
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


def expand_bbox(
    bbox: BoundingBox,
    pixels: int,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
) -> BoundingBox:
    """
    Expand bounding box by N pixels in all directions.

    Expands the box symmetrically around its center. If max dimensions are
    provided, clips the result to stay within those bounds.

    Args:
        bbox: Original bounding box
        pixels: Number of pixels to expand (can be negative to shrink)
        max_width: Optional maximum width to clip expansion (image width)
        max_height: Optional maximum height to clip expansion (image height)

    Returns:
        Expanded bounding box clipped to boundaries if max dimensions provided.
        May return a zero-area box if shrinkage or clipping results in no area.

    Example:
        >>> bbox = BoundingBox(x=5, y=5, width=10, height=10)
        >>> expand_bbox(bbox, pixels=10, max_width=100, max_height=100)
        BoundingBox(x=0, y=0, width=25, height=25)
    """
    # Calculate expanded corners first (before clipping)
    x1 = bbox.x - pixels
    y1 = bbox.y - pixels
    x2 = bbox.x + bbox.width + pixels
    y2 = bbox.y + bbox.height + pixels

    # Clip to boundaries
    x1_clipped = max(0, x1)
    y1_clipped = max(0, y1)

    if max_width is not None:
        x2_clipped = min(x2, max_width)
    else:
        x2_clipped = max(x1_clipped, x2)  # Ensure x2 >= x1

    if max_height is not None:
        y2_clipped = min(y2, max_height)
    else:
        y2_clipped = max(y1_clipped, y2)  # Ensure y2 >= y1

    # Calculate final dimensions from clipped corners
    width = x2_clipped - x1_clipped
    height = y2_clipped - y1_clipped

    return BoundingBox(x=x1_clipped, y=y1_clipped, width=width, height=height)


def merge_bboxes(bboxes: List[BoundingBox]) -> BoundingBox:
    """
    Merge multiple bounding boxes into single bounding box.

    Args:
        bboxes: List of bounding boxes to merge

    Returns:
        Merged bounding box that contains all input boxes. Returns a zero-area
        bounding box at origin if input list is empty.

    Example:
        >>> boxes = [BoundingBox(0, 0, 10, 10), BoundingBox(20, 20, 10, 10)]
        >>> merge_bboxes(boxes)
        BoundingBox(x=0, y=0, width=30, height=30)
        >>> merge_bboxes([])
        BoundingBox(x=0, y=0, width=0, height=0)
    """
    if not bboxes:
        return BoundingBox(x=0, y=0, width=0, height=0)

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


def normalize_bbox(
    bbox: BoundingBox, image_width: int, image_height: int
) -> NormalizedBBox:
    """
    Normalize bounding box to [0, 1] coordinates relative to image size.

    Args:
        bbox: Bounding box in pixel coordinates
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        Normalized bounding box

    Raises:
        ValueError: If image dimensions are zero or negative

    Example:
        >>> bbox = BoundingBox(x=50, y=100, width=100, height=50)
        >>> normalize_bbox(bbox, 200, 200)
        NormalizedBBox(x_center=0.5, y_center=0.625, width=0.5, height=0.25)
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"Image dimensions must be positive, got width={image_width}, "
            f"height={image_height}"
        )

    # Calculate center in pixels (using float precision from bbox.center())
    center_x_px, center_y_px = bbox.center()

    # Normalize
    x_center = center_x_px / image_width
    y_center = center_y_px / image_height
    width = bbox.width / image_width
    height = bbox.height / image_height

    # Clamp to [0, 1] range to handle edge cases
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

    return NormalizedBBox(
        x_center=x_center, y_center=y_center, width=width, height=height
    )


def denormalize_bbox(
    norm_bbox: NormalizedBBox, image_width: int, image_height: int
) -> BoundingBox:
    """
    Convert normalized bounding box back to pixel coordinates.

    Uses rounding for better geometric accuracy when converting from
    floating-point normalized coordinates to integer pixel coordinates.

    Args:
        norm_bbox: Normalized bounding box
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        Bounding box in pixel coordinates

    Raises:
        ValueError: If image dimensions are zero or negative

    Example:
        >>> norm = NormalizedBBox(x_center=0.5, y_center=0.5, width=0.5, height=0.25)
        >>> denormalize_bbox(norm, 200, 200)
        BoundingBox(x=50, y=75, width=100, height=50)
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"Image dimensions must be positive, got width={image_width}, "
            f"height={image_height}"
        )

    # Convert center to pixels
    center_x_px = norm_bbox.x_center * image_width
    center_y_px = norm_bbox.y_center * image_height

    # Convert size to pixels
    width_px = norm_bbox.width * image_width
    height_px = norm_bbox.height * image_height

    # Calculate top-left corner with rounding for better accuracy
    x = int(round(center_x_px - width_px / 2.0))
    y = int(round(center_y_px - height_px / 2.0))

    return BoundingBox(
        x=max(0, x),
        y=max(0, y),
        width=int(round(width_px)),
        height=int(round(height_px)),
    )


def clip_bbox_to_image(
    bbox: BoundingBox, image_width: int, image_height: int
) -> BoundingBox:
    """
    Clip bounding box to image boundaries.

    If the bounding box is completely outside the image, returns a zero-area
    bounding box. Callers can check result.area() == 0 to detect this case.

    Args:
        bbox: Original bounding box
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        Clipped bounding box within image bounds (may have zero area)

    Example:
        >>> bbox = BoundingBox(x=-10, y=-10, width=50, height=50)
        >>> clipped = clip_bbox_to_image(bbox, 100, 100)
        >>> clipped
        BoundingBox(x=0, y=0, width=40, height=40)
    """
    x1, y1, x2, y2 = bbox.to_corners()

    # Clip to boundaries
    x1 = max(0, min(x1, image_width))
    y1 = max(0, min(y1, image_height))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))

    width = x2 - x1
    height = y2 - y1

    return BoundingBox(x=x1, y=y1, width=width, height=height)


def bbox_intersection(bbox1: BoundingBox, bbox2: BoundingBox) -> BoundingBox:
    """
    Calculate intersection rectangle of two bounding boxes.

    Args:
        bbox1: First bounding box
        bbox2: Second bounding box

    Returns:
        Intersection bounding box. If boxes don't overlap, returns a zero-area
        bounding box at origin. Callers can check result.area() == 0.

    Example:
        >>> bbox1 = BoundingBox(x=0, y=0, width=10, height=10)
        >>> bbox2 = BoundingBox(x=5, y=5, width=10, height=10)
        >>> intersection = bbox_intersection(bbox1, bbox2)
        >>> intersection
        BoundingBox(x=5, y=5, width=5, height=5)
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
        return BoundingBox(x=0, y=0, width=0, height=0)

    return BoundingBox(
        x=x1_inter,
        y=y1_inter,
        width=x2_inter - x1_inter,
        height=y2_inter - y1_inter,
    )


def calculate_aspect_ratio(bbox: BoundingBox) -> float:
    """
    Calculate aspect ratio of bounding box.

    Args:
        bbox: Bounding box

    Returns:
        Aspect ratio (width / height)

    Raises:
        ValueError: If height is zero

    Example:
        >>> bbox = BoundingBox(x=0, y=0, width=16, height=9)
        >>> calculate_aspect_ratio(bbox)
        1.7777777777777777
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
        True if point is inside bbox (inclusive of edges)

    Example:
        >>> bbox = BoundingBox(x=0, y=0, width=10, height=10)
        >>> point_in_bbox((5, 5), bbox)
        True
        >>> point_in_bbox((15, 15), bbox)
        False
    """
    x, y = point
    x1, y1, x2, y2 = bbox.to_corners()

    return x1 <= x <= x2 and y1 <= y <= y2


def bbox_distance(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """
    Calculate minimum edge-to-edge distance between two bounding boxes.

    Returns 0.0 if boxes overlap. For non-overlapping boxes, calculates the
    shortest distance between any two edges.

    Args:
        bbox1: First bounding box
        bbox2: Second bounding box

    Returns:
        Minimum distance in pixels (0.0 if boxes overlap)

    Example:
        >>> bbox1 = BoundingBox(x=0, y=0, width=10, height=10)
        >>> bbox2 = BoundingBox(x=20, y=0, width=10, height=10)
        >>> bbox_distance(bbox1, bbox2)
        10.0
    """
    if bbox_overlaps(bbox1, bbox2):
        return 0.0

    # Get corner coordinates
    x1_1, y1_1, x2_1, y2_1 = bbox1.to_corners()
    x1_2, y1_2, x2_2, y2_2 = bbox2.to_corners()

    # Calculate horizontal and vertical distances
    # Horizontal distance
    if x2_1 < x1_2:
        dx = x1_2 - x2_1
    elif x2_2 < x1_1:
        dx = x1_1 - x2_2
    else:
        dx = 0

    # Vertical distance
    if y2_1 < y1_2:
        dy = y1_2 - y2_1
    elif y2_2 < y1_1:
        dy = y1_1 - y2_2
    else:
        dy = 0

    # Calculate Euclidean distance
    return math.sqrt(dx * dx + dy * dy)


def scale_bbox(bbox: BoundingBox, scale_factor: float) -> BoundingBox:
    """
    Scale bounding box by a factor around its center.

    Args:
        bbox: Original bounding box
        scale_factor: Scale factor (e.g., 1.5 for 150%, 0.5 for 50%)

    Returns:
        Scaled bounding box

    Raises:
        ValueError: If scale_factor is negative

    Example:
        >>> bbox = BoundingBox(x=10, y=10, width=20, height=20)
        >>> scale_bbox(bbox, scale_factor=2.0)
        BoundingBox(x=0, y=0, width=40, height=40)
    """
    if scale_factor < 0:
        raise ValueError(f"Scale factor must be non-negative, got {scale_factor}")

    center_x, center_y = bbox.center()

    new_width = int(round(bbox.width * scale_factor))
    new_height = int(round(bbox.height * scale_factor))

    new_x = int(round(center_x - new_width / 2.0))
    new_y = int(round(center_y - new_height / 2.0))

    return BoundingBox(
        x=max(0, new_x), y=max(0, new_y), width=new_width, height=new_height
    )
