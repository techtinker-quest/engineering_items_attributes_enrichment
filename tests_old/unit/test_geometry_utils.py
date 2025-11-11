"""
Unit tests for geometry_utils module.
"""

import pytest
from src.drawing_intelligence.utils.geometry_utils import (
    BoundingBox,
    NormalizedBBox,
    calculate_iou,
    calculate_distance,
    bbox_contains,
    bbox_overlaps,
    expand_bbox,
    merge_bboxes,
    normalize_bbox,
    denormalize_bbox,
    clip_bbox_to_image,
    calculate_aspect_ratio,
    point_in_bbox,
    scale_bbox,
)


class TestBoundingBox:
    """Tests for BoundingBox class."""

    def test_creation(self):
        """Test creating a bounding box."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)

        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 50

    def test_center(self):
        """Test center calculation."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        center = bbox.center()

        assert center == (60, 45)  # (10+100/2, 20+50/2)

    def test_area(self):
        """Test area calculation."""
        bbox = BoundingBox(x=0, y=0, width=100, height=50)
        area = bbox.area()

        assert area == 5000

    def test_to_tuple(self):
        """Test conversion to tuple."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        tuple_repr = bbox.to_tuple()

        assert tuple_repr == (10, 20, 100, 50)

    def test_to_corners(self):
        """Test conversion to corner coordinates."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        corners = bbox.to_corners()

        assert corners == (10, 20, 110, 70)  # (x1, y1, x2, y2)


class TestCalculateIoU:
    """Tests for calculate_iou function."""

    def test_identical_boxes(self):
        """Test IoU of identical boxes."""
        bbox1 = BoundingBox(x=0, y=0, width=100, height=100)
        bbox2 = BoundingBox(x=0, y=0, width=100, height=100)

        iou = calculate_iou(bbox1, bbox2)

        assert iou == 1.0

    def test_no_overlap(self):
        """Test IoU of non-overlapping boxes."""
        bbox1 = BoundingBox(x=0, y=0, width=50, height=50)
        bbox2 = BoundingBox(x=100, y=100, width=50, height=50)

        iou = calculate_iou(bbox1, bbox2)

        assert iou == 0.0

    def test_partial_overlap(self):
        """Test IoU of partially overlapping boxes."""
        bbox1 = BoundingBox(x=0, y=0, width=100, height=100)
        bbox2 = BoundingBox(x=50, y=50, width=100, height=100)

        iou = calculate_iou(bbox1, bbox2)

        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU: 2500/17500 = 0.142857...
        assert 0.14 < iou < 0.15

    def test_contained_box(self):
        """Test IoU when one box contains another."""
        bbox1 = BoundingBox(x=0, y=0, width=100, height=100)
        bbox2 = BoundingBox(x=25, y=25, width=50, height=50)

        iou = calculate_iou(bbox1, bbox2)

        # Intersection: 2500
        # Union: 10000 + 2500 - 2500 = 10000
        # IoU: 2500/10000 = 0.25
        assert iou == 0.25


class TestCalculateDistance:
    """Tests for calculate_distance function."""

    def test_same_point(self):
        """Test distance between same point."""
        distance = calculate_distance((0, 0), (0, 0))
        assert distance == 0.0

    def test_horizontal_distance(self):
        """Test horizontal distance."""
        distance = calculate_distance((0, 0), (3, 0))
        assert distance == 3.0

    def test_vertical_distance(self):
        """Test vertical distance."""
        distance = calculate_distance((0, 0), (0, 4))
        assert distance == 4.0

    def test_diagonal_distance(self):
        """Test diagonal distance (Pythagorean theorem)."""
        distance = calculate_distance((0, 0), (3, 4))
        assert distance == 5.0  # 3-4-5 triangle


class TestBboxContains:
    """Tests for bbox_contains function."""

    def test_fully_contained(self):
        """Test when inner box is fully inside outer."""
        outer = BoundingBox(x=0, y=0, width=100, height=100)
        inner = BoundingBox(x=25, y=25, width=50, height=50)

        assert bbox_contains(outer, inner) is True

    def test_not_contained(self):
        """Test when boxes don't contain each other."""
        outer = BoundingBox(x=0, y=0, width=100, height=100)
        inner = BoundingBox(x=50, y=50, width=100, height=100)

        assert bbox_contains(outer, inner) is False

    def test_edge_touching(self):
        """Test when inner box touches edge of outer."""
        outer = BoundingBox(x=0, y=0, width=100, height=100)
        inner = BoundingBox(x=0, y=0, width=100, height=100)

        assert bbox_contains(outer, inner) is True


class TestBboxOverlaps:
    """Tests for bbox_overlaps function."""

    def test_overlapping_boxes(self):
        """Test overlapping boxes."""
        bbox1 = BoundingBox(x=0, y=0, width=100, height=100)
        bbox2 = BoundingBox(x=50, y=50, width=100, height=100)

        assert bbox_overlaps(bbox1, bbox2) is True

    def test_non_overlapping_boxes(self):
        """Test non-overlapping boxes."""
        bbox1 = BoundingBox(x=0, y=0, width=50, height=50)
        bbox2 = BoundingBox(x=100, y=100, width=50, height=50)

        assert bbox_overlaps(bbox1, bbox2) is False

    def test_edge_touching_boxes(self):
        """Test boxes that touch at edges (no overlap)."""
        bbox1 = BoundingBox(x=0, y=0, width=50, height=50)
        bbox2 = BoundingBox(x=50, y=0, width=50, height=50)

        # Edge touching is not overlap
        assert bbox_overlaps(bbox1, bbox2) is False


class TestExpandBbox:
    """Tests for expand_bbox function."""

    def test_expand_by_10_pixels(self):
        """Test expanding bbox by 10 pixels."""
        bbox = BoundingBox(x=50, y=50, width=100, height=100)
        expanded = expand_bbox(bbox, 10)

        assert expanded.x == 40
        assert expanded.y == 40
        assert expanded.width == 120
        assert expanded.height == 120

    def test_expand_near_zero(self):
        """Test expanding bbox near zero (shouldn't go negative)."""
        bbox = BoundingBox(x=5, y=5, width=10, height=10)
        expanded = expand_bbox(bbox, 10)

        assert expanded.x == 0  # Clamped to 0
        assert expanded.y == 0  # Clamped to 0
        assert expanded.width == 30
        assert expanded.height == 30


class TestMergeBboxes:
    """Tests for merge_bboxes function."""

    def test_merge_two_boxes(self):
        """Test merging two bounding boxes."""
        bbox1 = BoundingBox(x=0, y=0, width=50, height=50)
        bbox2 = BoundingBox(x=100, y=100, width=50, height=50)

        merged = merge_bboxes([bbox1, bbox2])

        assert merged.x == 0
        assert merged.y == 0
        assert merged.width == 150  # 0 to 150
        assert merged.height == 150  # 0 to 150

    def test_merge_overlapping_boxes(self):
        """Test merging overlapping boxes."""
        bbox1 = BoundingBox(x=0, y=0, width=100, height=100)
        bbox2 = BoundingBox(x=50, y=50, width=100, height=100)

        merged = merge_bboxes([bbox1, bbox2])

        assert merged.x == 0
        assert merged.y == 0
        assert merged.width == 150
        assert merged.height == 150

    def test_merge_empty_list_raises_error(self):
        """Test that merging empty list raises error."""
        with pytest.raises(ValueError):
            merge_bboxes([])


class TestNormalizeDenormalizeBbox:
    """Tests for normalize_bbox and denormalize_bbox functions."""

    def test_normalize_bbox(self):
        """Test normalizing bbox to [0,1] coordinates."""
        bbox = BoundingBox(x=50, y=100, width=100, height=200)
        normalized = normalize_bbox(bbox, image_width=400, image_height=800)

        # Center: (100, 200)
        # Normalized center: (100/400=0.25, 200/800=0.25)
        # Normalized size: (100/400=0.25, 200/800=0.25)
        assert normalized.x_center == 0.25
        assert normalized.y_center == 0.25
        assert normalized.width == 0.25
        assert normalized.height == 0.25

    def test_denormalize_bbox(self):
        """Test denormalizing bbox back to pixel coordinates."""
        norm_bbox = NormalizedBBox(x_center=0.5, y_center=0.5, width=0.5, height=0.5)
        bbox = denormalize_bbox(norm_bbox, image_width=400, image_height=400)

        # Center: (200, 200)
        # Size: (200, 200)
        # Top-left: (100, 100)
        assert bbox.x == 100
        assert bbox.y == 100
        assert bbox.width == 200
        assert bbox.height == 200

    def test_normalize_denormalize_round_trip(self):
        """Test that normalize->denormalize gives back original."""
        original = BoundingBox(x=50, y=100, width=100, height=200)
        normalized = normalize_bbox(original, 400, 800)
        back = denormalize_bbox(normalized, 400, 800)

        # Account for integer rounding
        assert abs(back.x - original.x) <= 1
        assert abs(back.y - original.y) <= 1
        assert abs(back.width - original.width) <= 1
        assert abs(back.height - original.height) <= 1


class TestClipBboxToImage:
    """Tests for clip_bbox_to_image function."""

    def test_clip_bbox_outside_image(self):
        """Test clipping bbox that extends outside image."""
        bbox = BoundingBox(x=350, y=350, width=100, height=100)
        clipped = clip_bbox_to_image(bbox, image_width=400, image_height=400)

        assert clipped.x == 350
        assert clipped.y == 350
        assert clipped.width == 50  # Clipped to image edge
        assert clipped.height == 50  # Clipped to image edge

    def test_clip_negative_coordinates(self):
        """Test clipping bbox with negative coordinates."""
        bbox = BoundingBox(x=-50, y=-50, width=100, height=100)
        clipped = clip_bbox_to_image(bbox, image_width=400, image_height=400)

        assert clipped.x == 0
        assert clipped.y == 0
        assert clipped.width == 50
        assert clipped.height == 50

    def test_bbox_fully_inside_unchanged(self):
        """Test that bbox fully inside image is unchanged."""
        bbox = BoundingBox(x=50, y=50, width=100, height=100)
        clipped = clip_bbox_to_image(bbox, image_width=400, image_height=400)

        assert clipped.x == bbox.x
        assert clipped.y == bbox.y
        assert clipped.width == bbox.width
        assert clipped.height == bbox.height


class TestCalculateAspectRatio:
    """Tests for calculate_aspect_ratio function."""

    def test_square_aspect_ratio(self):
        """Test aspect ratio of square."""
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        ratio = calculate_aspect_ratio(bbox)

        assert ratio == 1.0

    def test_horizontal_rectangle(self):
        """Test aspect ratio of horizontal rectangle."""
        bbox = BoundingBox(x=0, y=0, width=200, height=100)
        ratio = calculate_aspect_ratio(bbox)

        assert ratio == 2.0

    def test_vertical_rectangle(self):
        """Test aspect ratio of vertical rectangle."""
        bbox = BoundingBox(x=0, y=0, width=100, height=200)
        ratio = calculate_aspect_ratio(bbox)

        assert ratio == 0.5

    def test_zero_height_raises_error(self):
        """Test that zero height raises error."""
        bbox = BoundingBox(x=0, y=0, width=100, height=0)

        with pytest.raises(ValueError):
            calculate_aspect_ratio(bbox)


class TestPointInBbox:
    """Tests for point_in_bbox function."""

    def test_point_inside(self):
        """Test point inside bbox."""
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        assert point_in_bbox((50, 50), bbox) is True

    def test_point_outside(self):
        """Test point outside bbox."""
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        assert point_in_bbox((150, 150), bbox) is False

    def test_point_on_edge(self):
        """Test point on bbox edge."""
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        assert point_in_bbox((100, 100), bbox) is True


class TestScaleBbox:
    """Tests for scale_bbox function."""

    def test_scale_up(self):
        """Test scaling bbox up."""
        bbox = BoundingBox(x=50, y=50, width=100, height=100)
        scaled = scale_bbox(bbox, 2.0)

        # Center stays same: (100, 100)
        # New size: 200x200
        # New top-left: (0, 0)
        assert scaled.width == 200
        assert scaled.height == 200
        center = scaled.center()
        assert center == (100, 100)

    def test_scale_down(self):
        """Test scaling bbox down."""
        bbox = BoundingBox(x=50, y=50, width=100, height=100)
        scaled = scale_bbox(bbox, 0.5)

        assert scaled.width == 50
        assert scaled.height == 50
        center = scaled.center()
        assert center == (100, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
