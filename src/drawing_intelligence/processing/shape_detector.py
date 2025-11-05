"""
Shape detection module for the Drawing Intelligence System.

Uses YOLOv8 for detecting and classifying component shapes.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from ..models.data_structures import Detection, DetectionResult, DetectionSummary
from ..utils.geometry_utils import (
    BoundingBox,
    NormalizedBBox,
    normalize_bbox,
    calculate_iou,
)
from ..utils.file_utils import generate_unique_id
from ..utils.error_handlers import ShapeDetectionError
from ..utils.validation_utils import validate_model_path, validate_image_array

logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    """
    Configuration for shape detection.

    Attributes:
        model_path: Path to YOLOv8 model weights
        confidence_threshold: Minimum confidence for detections (default: 0.45)
        nms_threshold: IoU threshold for NMS (default: 0.45)
        device: Device for inference ('cuda', 'cpu', 'mps')
        batch_size: Batch size for inference (default: 8)
        max_det: Maximum detections per image (default: 300)
    """

    model_path: str
    confidence_threshold: float = 0.45
    nms_threshold: float = 0.45
    device: str = "cuda"
    batch_size: int = 8
    max_det: int = 300


class ShapeDetector:
    """
    Detect and classify component shapes using YOLOv8.

    Component classes (20 types):
    - bearing, gear, bolt, screw, nut, washer, pin, shaft
    - motor, sensor, controller, valve, pump, cylinder
    - spring, pulley, bracket, housing, connector, fastener
    """

    def __init__(self, config: DetectionConfig):
        """
        Initialize shape detector.

        Args:
            config: Detection configuration

        Raises:
            ShapeDetectionError: If initialization fails
        """
        self.config = config

        # Validate model path
        is_valid, error_msg = validate_model_path(config.model_path)
        if not is_valid:
            raise ShapeDetectionError(error_msg)

        # Lazy load model
        self._model = None
        self._model_version = None

        logger.info(f"ShapeDetector initialized (device: {config.device})")

    def detect_shapes(self, image: np.ndarray) -> DetectionResult:
        """
        Run shape detection on image.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            DetectionResult with all detections

        Raises:
            ShapeDetectionError: If detection fails
        """
        # Validate image
        is_valid, error_msg = validate_image_array(image)
        if not is_valid:
            raise ShapeDetectionError(f"Invalid input image: {error_msg}")

        # Load model if not loaded
        if self._model is None:
            self._load_model()

        try:
            import time

            start_time = time.time()

            # Run inference
            results = self._model.predict(
                source=image,
                conf=self.config.confidence_threshold,
                iou=self.config.nms_threshold,
                max_det=self.config.max_det,
                verbose=False,
            )

            inference_time = (time.time() - start_time) * 1000  # ms

            # Process results
            detections = self._postprocess_detections(results[0], image.shape)

            # Apply additional NMS if needed
            detections = self._apply_nms(detections, self.config.nms_threshold)

            # Filter by confidence
            detections = self.filter_by_confidence(
                detections, self.config.confidence_threshold
            )

            # Calculate summary
            summary = self._calculate_summary(detections)

            result = DetectionResult(
                detections=detections,
                summary=summary,
                inference_time_ms=inference_time,
                model_version=self._model_version,
            )

            logger.info(
                f"Shape detection complete: {len(detections)} shapes detected "
                f"in {inference_time:.1f}ms"
            )

            return result

        except Exception as e:
            raise ShapeDetectionError(
                f"Shape detection failed: {e}", model_version=self._model_version
            )

    def _load_model(self):
        """Lazy load YOLOv8 model."""
        try:
            from ultralytics import YOLO

            logger.info(f"Loading YOLOv8 model from {self.config.model_path}")
            self._model = YOLO(self.config.model_path)

            # Set device
            self._model.to(self.config.device)

            # Get model version/name
            self._model_version = f"YOLOv8_{self.config.model_path.split('/')[-1]}"

            logger.info(f"Model loaded successfully on {self.config.device}")

        except ImportError:
            raise ShapeDetectionError(
                "ultralytics not installed. Install with: pip install ultralytics"
            )
        except FileNotFoundError:
            raise ShapeDetectionError(f"Model file not found: {self.config.model_path}")
        except Exception as e:
            raise ShapeDetectionError(f"Failed to load model: {e}")

    def _postprocess_detections(
        self, results: Any, image_shape: tuple
    ) -> List[Detection]:
        """
        Convert YOLO results to Detection objects.

        Args:
            results: YOLO results object
            image_shape: Original image shape (H, W, C)

        Returns:
            List of Detection objects
        """
        detections = []

        # Get image dimensions
        height, width = image_shape[:2]

        # Extract boxes, scores, and classes
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            # Get box coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Get confidence and class
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())

            # Get class name
            class_name = results.names[class_id]

            # Create bounding box
            bbox = BoundingBox(
                x=int(x1), y=int(y1), width=int(x2 - x1), height=int(y2 - y1)
            )

            # Create normalized bbox
            bbox_normalized = normalize_bbox(bbox, width, height)

            # Create detection
            detection = Detection(
                detection_id=generate_unique_id("DET"),
                class_name=class_name,
                confidence=confidence,
                bbox=bbox,
                bbox_normalized=bbox_normalized,
                model_version=self._model_version,
            )

            detections.append(detection)

        return detections

    def filter_by_confidence(
        self, detections: List[Detection], threshold: float
    ) -> List[Detection]:
        """
        Filter detections by confidence threshold.

        Args:
            detections: List of detections
            threshold: Minimum confidence

        Returns:
            Filtered list of detections
        """
        filtered = [d for d in detections if d.confidence >= threshold]

        if len(filtered) < len(detections):
            logger.debug(
                f"Filtered {len(detections) - len(filtered)} "
                f"low-confidence detections"
            )

        return filtered

    def _apply_nms(
        self, detections: List[Detection], iou_threshold: float
    ) -> List[Detection]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections.

        Args:
            detections: List of detections
            iou_threshold: IoU threshold for suppression

        Returns:
            List of detections after NMS
        """
        if not detections:
            return detections

        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        keep = []

        while detections:
            # Keep highest confidence detection
            current = detections.pop(0)
            keep.append(current)

            # Remove overlapping detections
            detections = [
                d
                for d in detections
                if calculate_iou(current.bbox, d.bbox) < iou_threshold
            ]

        return keep

    def _calculate_summary(self, detections: List[Detection]) -> DetectionSummary:
        """
        Calculate summary statistics for detections.

        Args:
            detections: List of detections

        Returns:
            DetectionSummary object
        """
        # Count by class
        detections_by_class = {}
        for det in detections:
            detections_by_class[det.class_name] = (
                detections_by_class.get(det.class_name, 0) + 1
            )

        # Average confidence
        avg_confidence = (
            sum(d.confidence for d in detections) / len(detections)
            if detections
            else 0.0
        )

        # Confidence distribution
        high_conf = len([d for d in detections if d.confidence >= 0.7])
        medium_conf = len([d for d in detections if 0.5 <= d.confidence < 0.7])
        low_conf = len([d for d in detections if d.confidence < 0.5])

        total = len(detections) if detections else 1
        confidence_distribution = {
            "high": high_conf / total,
            "medium": medium_conf / total,
            "low": low_conf / total,
        }

        return DetectionSummary(
            total_detections=len(detections),
            detections_by_class=detections_by_class,
            average_confidence=avg_confidence,
            confidence_distribution=confidence_distribution,
        )

    def detect_shapes_batch(self, images: List[np.ndarray]) -> List[DetectionResult]:
        """
        Run shape detection on multiple images (batch processing).

        Args:
            images: List of input images

        Returns:
            List of DetectionResult objects
        """
        if not images:
            return []

        # Load model if not loaded
        if self._model is None:
            self._load_model()

        results_list = []

        # Process in batches
        for i in range(0, len(images), self.config.batch_size):
            batch = images[i : i + self.config.batch_size]

            try:
                # Run batch inference
                batch_results = self._model.predict(
                    source=batch,
                    conf=self.config.confidence_threshold,
                    iou=self.config.nms_threshold,
                    max_det=self.config.max_det,
                    verbose=False,
                )

                # Process each result
                for idx, results in enumerate(batch_results):
                    image_shape = batch[idx].shape
                    detections = self._postprocess_detections(results, image_shape)
                    summary = self._calculate_summary(detections)

                    result = DetectionResult(
                        detections=detections,
                        summary=summary,
                        inference_time_ms=0.0,  # Not tracked in batch mode
                        model_version=self._model_version,
                    )
                    results_list.append(result)

            except Exception as e:
                logger.error(f"Batch detection failed for batch {i}: {e}")
                # Add empty results for failed images
                for _ in batch:
                    results_list.append(
                        DetectionResult(
                            detections=[],
                            summary=DetectionSummary(
                                total_detections=0,
                                detections_by_class={},
                                average_confidence=0.0,
                                confidence_distribution={
                                    "high": 0,
                                    "medium": 0,
                                    "low": 0,
                                },
                            ),
                            inference_time_ms=0.0,
                            model_version=self._model_version,
                        )
                    )

        logger.info(f"Batch detection complete: {len(images)} images processed")
        return results_list

    def get_model_info(self) -> dict:
        """
        Get information about loaded model.

        Returns:
            Dictionary with model information
        """
        if self._model is None:
            return {"loaded": False, "model_path": self.config.model_path}

        return {
            "loaded": True,
            "model_path": self.config.model_path,
            "model_version": self._model_version,
            "device": self.config.device,
            "num_classes": len(self._model.names),
            "class_names": list(self._model.names.values()),
        }
