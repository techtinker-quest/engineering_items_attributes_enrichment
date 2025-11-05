"""
Unit tests for DataAssociator.
"""

import pytest
import numpy as np
from src.drawing_intelligence.processing.data_associator import (
    DataAssociator,
    AssociationConfig,
)
from src.drawing_intelligence.models.data_structures import (
    TextBlock,
    Detection,
    BoundingBox,
    NormalizedBBox,
)


@pytest.fixture
def associator_config():
    """Create test configuration."""
    return AssociationConfig(
        label_distance_threshold=200,
        dimension_distance_threshold=500,
        min_association_confidence=0.6,
        enable_obstacle_detection=True,
    )


@pytest.fixture
def associator(associator_config):
    """Create DataAssociator instance."""
    return DataAssociator(associator_config)


@pytest.fixture
def sample_text_blocks():
    """Create sample text blocks."""
    return [
        TextBlock(
            text_id="TXT-001",
            content="Ø25.4mm",
            bbox=BoundingBox(x=100, y=100, width=80, height=20),
            confidence=0.92,
            ocr_engine="paddleocr",
            region_type="dimension",
        ),
        TextBlock(
            text_id="TXT-002",
            content="BOLT",
            bbox=BoundingBox(x=300, y=150, width=50, height=20),
            confidence=0.95,
            ocr_engine="paddleocr",
            region_type="label",
        ),
    ]


@pytest.fixture
def sample_detections():
    """Create sample detections."""
    return [
        Detection(
            detection_id="DET-001",
            class_name="bolt",
            confidence=0.88,
            bbox=BoundingBox(x=200, y=200, width=100, height=100),
            bbox_normalized=NormalizedBBox(
                x_center=0.3125, y_center=0.4167, width=0.125, height=0.1667
            ),
        ),
        Detection(
            detection_id="DET-002",
            class_name="gear",
            confidence=0.91,
            bbox=BoundingBox(x=500, y=200, width=120, height=120),
            bbox_normalized=NormalizedBBox(
                x_center=0.7, y_center=0.4167, width=0.15, height=0.2
            ),
        ),
    ]


class TestDataAssociator:
    """Test DataAssociator class."""

    def test_initialization(self, associator, associator_config):
        """Test associator initialization."""
        assert associator.label_threshold == 200
        assert associator.dimension_threshold == 500
        assert associator.min_confidence == 0.6

    def test_associate_text_to_shapes(
        self, associator, sample_text_blocks, sample_detections
    ):
        """Test text-to-shape association."""
        associations = associator.associate_text_to_shapes(
            sample_text_blocks, sample_detections
        )

        assert len(associations) > 0
        assert all(hasattr(a, "text_id") for a in associations)
        assert all(hasattr(a, "shape_id") for a in associations)
        assert all(hasattr(a, "confidence") for a in associations)
        assert all(0.0 <= a.confidence <= 1.0 for a in associations)

    def test_calculate_distance(self, associator):
        """Test distance calculation."""
        bbox1 = BoundingBox(x=100, y=100, width=50, height=50)
        bbox2 = BoundingBox(x=200, y=200, width=50, height=50)

        distance = associator._calculate_distance(bbox1, bbox2)

        assert distance > 0
        assert isinstance(distance, float)

        # Distance should be approximately 141.42 (sqrt of 100^2 + 100^2)
        assert 140 < distance < 145

    def test_classify_text_type(self, associator):
        """Test text type classification."""
        assert associator._classify_text_type("Ø25.4mm") == "dimension"
        assert associator._classify_text_type("BOLT") == "label"
        assert associator._classify_text_type("NOTE: Check clearance") == "note"

    def test_identify_multi_view_groups(self, associator, sample_detections):
        """Test multi-view grouping."""
        # Add more detections with similar characteristics
        aligned_detections = sample_detections + [
            Detection(
                detection_id="DET-003",
                class_name="bolt",
                confidence=0.87,
                bbox=BoundingBox(x=200, y=400, width=105, height=105),
                bbox_normalized=NormalizedBBox(
                    x_center=0.315, y_center=0.6, width=0.13, height=0.175
                ),
            ),
        ]

        groups = associator.identify_multi_view_groups(aligned_detections)

        assert isinstance(groups, list)
        # May or may not find groups depending on alignment
        for group in groups:
            assert hasattr(group, "component_class")
            assert hasattr(group, "shape_ids")

    def test_empty_inputs(self, associator):
        """Test with empty inputs."""
        associations = associator.associate_text_to_shapes([], [])
        assert len(associations) == 0

        groups = associator.identify_multi_view_groups([])
        assert len(groups) == 0

    def test_confidence_calculation(self, associator):
        """Test confidence decreases with distance."""
        text = TextBlock(
            text_id="TXT-001",
            content="Test",
            bbox=BoundingBox(x=100, y=100, width=50, height=20),
            confidence=0.9,
            ocr_engine="test",
            region_type="label",
        )

        shape_near = Detection(
            detection_id="DET-001",
            class_name="bolt",
            confidence=0.9,
            bbox=BoundingBox(x=150, y=110, width=50, height=50),
            bbox_normalized=NormalizedBBox(0.2, 0.2, 0.1, 0.1),
        )

        shape_far = Detection(
            detection_id="DET-002",
            class_name="bolt",
            confidence=0.9,
            bbox=BoundingBox(x=400, y=400, width=50, height=50),
            bbox_normalized=NormalizedBBox(0.5, 0.5, 0.1, 0.1),
        )

        associations = associator.associate_text_to_shapes(
            [text], [shape_near, shape_far]
        )

        # Should associate with near shape, not far shape
        assert len(associations) == 1
        assert associations[0].shape_id == "DET-001"
        assert associations[0].confidence > 0.6
