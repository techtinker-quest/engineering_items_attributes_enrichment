"""
Unit tests for shape_detector module.

Tests YOLOv8 shape detection, batch processing, and NMS filtering.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.drawing_intelligence.processing.shape_detector import (
    ShapeDetector,
    DetectionConfig,
)
from src.drawing_intelligence.models.data_structures import (
    Detection,
    DetectionResult,
    DetectionSummary,
    BoundingBox,
    NormalizedBBox,
)
from src.drawing_intelligence.utils.error_handlers import ShapeDetectionError


@pytest.fixture
def detection_config(tmp_path):
    """Create test detection configuration."""
    # Create dummy model file
    model_path = tmp_path / "test_model.pt"
    model_path.write_bytes(b"fake model data")

    return DetectionConfig(
        model_path=str(model_path),
        confidence_threshold=0.45,
        nms_threshold=0.45,
        device="cpu",
        batch_size=8,
        image_size=640,
    )


@pytest.fixture
def test_image():
    """Create test image."""
    # Create image with some patterns
    image = np.ones((640, 640, 3), dtype=np.uint8) * 200
    # Add some "shape" regions
    image[100:200, 100:200] = 50  # Dark square
    image[300:400, 300:450] = 100  # Rectangle
    return image


@pytest.fixture
def mock_yolo():
    """Create mock YOLO model."""
    with patch("src.drawing_intelligence.processing.shape_detector.YOLO") as mock:
        yield mock


class TestShapeDetectorInitialization:
    """Test shape detector initialization."""

    def test_init_with_config(self, detection_config, mock_yolo):
        """Test initialization with configuration."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(detection_config)

        assert detector.config == detection_config
        assert detector.config.confidence_threshold == 0.45
        assert detector.config.device == "cpu"

    def test_init_with_default_config(self, tmp_path, mock_yolo):
        """Test initialization with default configuration."""
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")

        config = DetectionConfig(model_path=str(model_path))
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(config)

        assert detector.config.confidence_threshold == 0.45
        assert detector.config.nms_threshold == 0.45

    def test_init_loads_model(self, detection_config, mock_yolo):
        """Test model is loaded on initialization."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(detection_config)

        mock_yolo.assert_called_once_with(detection_config.model_path)

    def test_init_sets_device(self, detection_config, mock_yolo):
        """Test device is set correctly."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(detection_config)

        mock_model.to.assert_called_with("cpu")

    def test_init_invalid_model_path(self, mock_yolo):
        """Test initialization with invalid model path."""
        config = DetectionConfig(model_path="nonexistent.pt")

        with pytest.raises((ShapeDetectionError, FileNotFoundError)):
            detector = ShapeDetector(config)


class TestSingleImageDetection:
    """Test detection on single images."""

    def test_detect_shapes_success(self, detection_config, test_image, mock_yolo):
        """Test successful shape detection."""
        # Mock YOLO results
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.xyxy = np.array([[100, 150, 200, 250]])  # x1, y1, x2, y2
        mock_box.conf = np.array([0.85])
        mock_box.cls = np.array([0])  # Class 0 = bolt
        mock_result.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_model.names = {0: "bolt", 1: "screw", 2: "gear"}
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(detection_config)
        result = detector.detect_shapes(test_image)

        assert isinstance(result, DetectionResult)
        assert len(result.detections) == 1
        assert result.detections[0].class_name == "bolt"
        assert result.detections[0].confidence == 0.85

    def test_detect_multiple_shapes(self, detection_config, test_image, mock_yolo):
        """Test detection of multiple shapes."""
        mock_result = MagicMock()

        # Multiple detections
        mock_box = MagicMock()
        mock_box.xyxy = np.array(
            [
                [100, 150, 200, 250],  # bolt
                [300, 350, 450, 500],  # gear
                [500, 100, 600, 200],  # screw
            ]
        )
        mock_box.conf = np.array([0.85, 0.90, 0.75])
        mock_box.cls = np.array([0, 2, 1])
        mock_result.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_model.names = {0: "bolt", 1: "screw", 2: "gear"}
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(detection_config)
        result = detector.detect_shapes(test_image)

        assert len(result.detections) == 3
        assert result.detections[0].class_name == "bolt"
        assert result.detections[1].class_name == "gear"
        assert result.detections[2].class_name == "screw"

    def test_detect_no_shapes(self, detection_config, test_image, mock_yolo):
        """Test detection with no shapes found."""
        mock_result = MagicMock()
        mock_result.boxes = []

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(detection_config)
        result = detector.detect_shapes(test_image)

        assert isinstance(result, DetectionResult)
        assert len(result.detections) == 0

    def test_detect_shapes_creates_bbox(self, detection_config, test_image, mock_yolo):
        """Test detection creates proper bounding boxes."""
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.xyxy = np.array(
            [[100, 150, 250, 300]]
        )  # x1=100, y1=150, x2=250, y2=300
        mock_box.conf = np.array([0.85])
        mock_box.cls = np.array([0])
        mock_result.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_model.names = {0: "bolt"}
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(detection_config)
        result = detector.detect_shapes(test_image)

        bbox = result.detections[0].bbox
        assert isinstance(bbox, BoundingBox)
        assert bbox.x == 100
        assert bbox.y == 150
        assert bbox.width == 150  # x2 - x1
        assert bbox.height == 150  # y2 - y1


class TestBatchDetection:
    """Test batch processing functionality."""

    def test_detect_shapes_batch(self, detection_config, mock_yolo):
        """Test batch detection on multiple images."""
        images = [
            np.ones((640, 640, 3), dtype=np.uint8) * 100,
            np.ones((640, 640, 3), dtype=np.uint8) * 150,
            np.ones((640, 640, 3), dtype=np.uint8) * 200,
        ]

        # Mock results for each image
        mock_results = []
        for i in range(3):
            mock_result = MagicMock()
            mock_box = MagicMock()
            mock_box.xyxy = np.array([[100, 150, 200, 250]])
            mock_box.conf = np.array([0.85])
            mock_box.cls = np.array([i])  # Different class for each
            mock_result.boxes = [mock_box]
            mock_results.append(mock_result)

        mock_model = MagicMock()
        mock_model.predict.return_value = mock_results
        mock_model.names = {0: "bolt", 1: "screw", 2: "gear"}
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(detection_config)
        results = detector.detect_shapes_batch(images)

        assert len(results) == 3
        assert all(isinstance(r, DetectionResult) for r in results)

    def test_batch_processing_efficiency(self, detection_config, mock_yolo):
        """Test batch processing is more efficient than sequential."""
        images = [np.ones((640, 640, 3), dtype=np.uint8)] * 8

        mock_model = MagicMock()
        mock_model.predict.return_value = [MagicMock(boxes=[]) for _ in range(8)]
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(detection_config)
        results = detector.detect_shapes_batch(images)

        # Should call predict once with batch
        assert mock_model.predict.call_count == 1


class TestConfidenceFiltering:
    """Test confidence-based filtering."""

    def test_filter_by_confidence_threshold(
        self, detection_config, test_image, mock_yolo
    ):
        """Test filtering detections by confidence."""
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.xyxy = np.array(
            [
                [100, 150, 200, 250],  # conf 0.85 - keep
                [300, 350, 450, 500],  # conf 0.30 - filter
                [500, 100, 600, 200],  # conf 0.50 - keep
            ]
        )
        mock_box.conf = np.array([0.85, 0.30, 0.50])
        mock_box.cls = np.array([0, 0, 0])
        mock_result.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_model.names = {0: "bolt"}
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(detection_config)
        result = detector.detect_shapes(test_image)

        # Should only keep detections >= 0.45
        assert len(result.detections) == 2
        assert all(d.confidence >= 0.45 for d in result.detections)

    def test_custom_confidence_filter(self, detection_config, test_image, mock_yolo):
        """Test filtering with custom threshold."""
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.xyxy = np.array(
            [
                [100, 150, 200, 250],  # 0.85
                [300, 350, 450, 500],  # 0.60
                [500, 100, 600, 200],  # 0.50
            ]
        )
        mock_box.conf = np.array([0.85, 0.60, 0.50])
        mock_box.cls = np.array([0, 0, 0])
        mock_result.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_model.names = {0: "bolt"}
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(detection_config)
        result = detector.detect_shapes(test_image)

        # Filter with higher threshold
        filtered = detector.filter_by_confidence(result.detections, threshold=0.70)

        assert len(filtered) == 1
        assert filtered[0].confidence == 0.85


class TestNMS:
    """Test Non-Maximum Suppression."""

    def test_nms_removes_duplicates(self, detection_config, test_image, mock_yolo):
        """Test NMS removes duplicate detections."""
        mock_result = MagicMock()
        mock_box = MagicMock()
        # Two overlapping detections of same class
        mock_box.xyxy = np.array(
            [
                [100, 150, 200, 250],  # conf 0.85
                [105, 155, 205, 255],  # conf 0.75 - should be suppressed (IoU > 0.45)
                [400, 400, 500, 500],  # conf 0.80 - different location
            ]
        )
        mock_box.conf = np.array([0.85, 0.75, 0.80])
        mock_box.cls = np.array([0, 0, 0])
        mock_result.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_model.names = {0: "bolt"}
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(detection_config)
        result = detector.detect_shapes(test_image)

        # Should keep 2 detections (remove the duplicate)
        assert len(result.detections) <= 2

    def test_nms_threshold(self, detection_config, mock_yolo):
        """Test NMS threshold configuration."""
        # Set low NMS threshold (more aggressive suppression)
        config = detection_config
        config.nms_threshold = 0.3

        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(config)

        assert detector.config.nms_threshold == 0.3


class TestBboxNormalization:
    """Test bounding box normalization."""

    def test_normalized_bbox_creation(self, detection_config, test_image, mock_yolo):
        """Test normalized bbox is created."""
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.xyxy = np.array([[100, 150, 200, 250]])
        mock_box.conf = np.array([0.85])
        mock_box.cls = np.array([0])
        mock_result.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_model.names = {0: "bolt"}
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(detection_config)
        result = detector.detect_shapes(test_image)

        detection = result.detections[0]
        assert isinstance(detection.bbox_normalized, NormalizedBBox)

    def test_normalized_coordinates_range(
        self, detection_config, test_image, mock_yolo
    ):
        """Test normalized coordinates are in [0, 1] range."""
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.xyxy = np.array([[100, 150, 200, 250]])
        mock_box.conf = np.array([0.85])
        mock_box.cls = np.array([0])
        mock_result.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_model.names = {0: "bolt"}
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(detection_config)
        result = detector.detect_shapes(test_image)

        norm_bbox = result.detections[0].bbox_normalized
        assert 0.0 <= norm_bbox.x_center <= 1.0
        assert 0.0 <= norm_bbox.y_center <= 1.0
        assert 0.0 <= norm_bbox.width <= 1.0
        assert 0.0 <= norm_bbox.height <= 1.0


class TestComponentClasses:
    """Test different component class detections."""

    def test_detect_fasteners(self, detection_config, test_image, mock_yolo):
        """Test detection of fastener types."""
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.xyxy = np.array(
            [
                [100, 100, 150, 150],  # bolt
                [200, 200, 250, 250],  # screw
                [300, 300, 350, 350],  # nut
            ]
        )
        mock_box.conf = np.array([0.85, 0.80, 0.90])
        mock_box.cls = np.array([0, 1, 2])
        mock_result.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_model.names = {0: "bolt", 1: "screw", 2: "nut"}
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(detection_config)
        result = detector.detect_shapes(test_image)

        classes = [d.class_name for d in result.detections]
        assert "bolt" in classes
        assert "screw" in classes
        assert "nut" in classes

    def test_detect_mechanical_components(
        self, detection_config, test_image, mock_yolo
    ):
        """Test detection of mechanical components."""
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.xyxy = np.array(
            [[100, 100, 200, 200], [250, 250, 350, 350]]  # gear  # bearing
        )
        mock_box.conf = np.array([0.85, 0.80])
        mock_box.cls = np.array([5, 6])
        mock_result.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_model.names = {5: "gear", 6: "bearing"}
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(detection_config)
        result = detector.detect_shapes(test_image)

        assert result.detections[0].class_name == "gear"
        assert result.detections[1].class_name == "bearing"


class TestDetectionSummary:
    """Test detection summary generation."""

    def test_summary_statistics(self, detection_config, test_image, mock_yolo):
        """Test summary contains correct statistics."""
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.xyxy = np.array(
            [[100, 100, 200, 200], [250, 250, 350, 350], [400, 400, 500, 500]]
        )
        mock_box.conf = np.array([0.85, 0.70, 0.90])
        mock_box.cls = np.array([0, 0, 1])
        mock_result.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_model.names = {0: "bolt", 1: "gear"}
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(detection_config)
        result = detector.detect_shapes(test_image)

        assert isinstance(result.summary, DetectionSummary)
        assert result.summary.total_detections == 3
        assert result.summary.detections_by_class["bolt"] == 2
        assert result.summary.detections_by_class["gear"] == 1

    def test_summary_average_confidence(self, detection_config, test_image, mock_yolo):
        """Test average confidence calculation."""
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.xyxy = np.array([[100, 100, 200, 200]])
        mock_box.conf = np.array([0.85])
        mock_box.cls = np.array([0])
        mock_result.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_model.names = {0: "bolt"}
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(detection_config)
        result = detector.detect_shapes(test_image)

        assert result.summary.average_confidence == 0.85

    def test_summary_confidence_distribution(
        self, detection_config, test_image, mock_yolo
    ):
        """Test confidence distribution in summary."""
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.xyxy = np.array(
            [
                [100, 100, 200, 200],  # 0.95 - high
                [250, 250, 350, 350],  # 0.70 - medium
                [400, 400, 500, 500],  # 0.50 - medium
            ]
        )
        mock_box.conf = np.array([0.95, 0.70, 0.50])
        mock_box.cls = np.array([0, 0, 0])
        mock_result.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_model.names = {0: "bolt"}
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(detection_config)
        result = detector.detect_shapes(test_image)

        assert result.summary.confidence_distribution["high"] == 1
        assert result.summary.confidence_distribution["medium"] == 2
        assert result.summary.confidence_distribution["low"] == 0


class TestDeviceHandling:
    """Test GPU/CPU device handling."""

    def test_cpu_device(self, tmp_path, mock_yolo):
        """Test CPU device configuration."""
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")

        config = DetectionConfig(model_path=str(model_path), device="cpu")
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(config)

        mock_model.to.assert_called_with("cpu")

    def test_cuda_device(self, tmp_path, mock_yolo):
        """Test CUDA device configuration."""
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")

        config = DetectionConfig(model_path=str(model_path), device="cuda")
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(config)

        mock_model.to.assert_called_with("cuda")


class TestInferenceTime:
    """Test inference time tracking."""

    def test_inference_time_recorded(self, detection_config, test_image, mock_yolo):
        """Test inference time is recorded."""
        mock_result = MagicMock()
        mock_result.boxes = []

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(detection_config)
        result = detector.detect_shapes(test_image)

        assert hasattr(result, "inference_time_seconds")
        assert result.inference_time_seconds >= 0.0


class TestErrorHandling:
    """Test error handling and recovery."""

    def test_invalid_image_input(self, detection_config, mock_yolo):
        """Test handling of invalid image."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(detection_config)

        with pytest.raises((ShapeDetectionError, ValueError)):
            detector.detect_shapes(None)

    def test_model_inference_failure(self, detection_config, test_image, mock_yolo):
        """Test handling of inference failure."""
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Inference failed")
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(detection_config)

        with pytest.raises(ShapeDetectionError, match="Detection failed"):
            detector.detect_shapes(test_image)

    def test_empty_batch(self, detection_config, mock_yolo):
        """Test handling of empty batch."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(detection_config)

        with pytest.raises(ValueError, match="Empty batch"):
            detector.detect_shapes_batch([])


class TestModelInfo:
    """Test model information retrieval."""

    def test_get_model_info(self, detection_config, mock_yolo):
        """Test getting model information."""
        mock_model = MagicMock()
        mock_model.names = {0: "bolt", 1: "screw"}
        mock_yolo.return_value = mock_model

        detector = ShapeDetector(detection_config)
        info = detector.get_model_info()

        assert "model_path" in info
        assert "device" in info
        assert "classes" in info
        assert info["classes"] == {0: "bolt", 1: "screw"}


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
