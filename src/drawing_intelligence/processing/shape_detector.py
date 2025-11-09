"""
Shape detection module for the Drawing Intelligence System.

This module provides YOLOv8-based object detection for identifying and classifying
mechanical component shapes in engineering drawings. Supports both single-image
and batch processing with configurable confidence thresholds and NMS parameters.

Classes:
    DetectionConfig: Configuration dataclass for shape detection parameters.
    ShapeDetector: Main detector class implementing YOLOv8 inference pipeline.

Typical usage example:
    config = DetectionConfig(
        model_path="models/yolov8_shapes.pt",
        confidence_threshold=0.5,
        device="cuda"
    )
    detector = ShapeDetector(config)

    # Single image processing
    result = detector.detect_shapes(image)

    # Batch processing with context manager (recommended for GPU)
    with detector:
        results = detector.detect_shapes_batch(images)
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

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
    """Configuration parameters for YOLOv8 shape detection.

    Attributes:
        model_path: Absolute or relative path to YOLOv8 model weights file (.pt).
        confidence_threshold: Minimum confidence score for detections during
            inference. Must be in range [0.0, 1.0]. Default: 0.45.
        nms_threshold: Intersection-over-Union threshold for Non-Maximum
            Suppression. Must be in range [0.0, 1.0]. Default: 0.45.
        device: Compute device for inference. Must be one of: 'cuda', 'cpu',
            'mps'. Default: 'cuda'. Falls back to 'cpu' if unavailable.
        batch_size: Number of images to process simultaneously in batch mode.
            Must be >= 1. Default: 8.
        max_det: Maximum number of detections to keep per image. Prevents
            memory issues with extremely complex drawings. Must be >= 1.
            Default: 300.
        confidence_high_threshold: Threshold for classifying detections as
            "high confidence" in summary statistics. Default: 0.7.
        confidence_medium_threshold: Threshold for classifying detections as
            "medium confidence" in summary statistics. Default: 0.5.
        check_batch_consistency: If True, validates that all images in a batch
            have the same dimensions. Default: True.
        progress_callback: Optional callback function for batch progress
            reporting. Called with (current_index, total_images). Default: None.

    Raises:
        ValueError: If any threshold values are outside valid ranges or if
            device is not a supported value.
    """

    model_path: str
    confidence_threshold: float = 0.45
    nms_threshold: float = 0.45
    device: str = "cuda"
    batch_size: int = 8
    max_det: int = 300
    confidence_high_threshold: float = 0.7
    confidence_medium_threshold: float = 0.5
    check_batch_consistency: bool = True
    progress_callback: Optional[Callable[[int, int], None]] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        # Validate confidence thresholds
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0], "
                f"got {self.confidence_threshold}"
            )
        if not 0.0 <= self.nms_threshold <= 1.0:
            raise ValueError(
                f"nms_threshold must be in [0.0, 1.0], got {self.nms_threshold}"
            )
        if not 0.0 <= self.confidence_high_threshold <= 1.0:
            raise ValueError(
                f"confidence_high_threshold must be in [0.0, 1.0], "
                f"got {self.confidence_high_threshold}"
            )
        if not 0.0 <= self.confidence_medium_threshold <= 1.0:
            raise ValueError(
                f"confidence_medium_threshold must be in [0.0, 1.0], "
                f"got {self.confidence_medium_threshold}"
            )

        # Validate logical threshold ordering
        if self.confidence_medium_threshold > self.confidence_high_threshold:
            raise ValueError(
                f"confidence_medium_threshold ({self.confidence_medium_threshold}) "
                f"must be <= confidence_high_threshold "
                f"({self.confidence_high_threshold})"
            )

        # Validate integer parameters
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.max_det < 1:
            raise ValueError(f"max_det must be >= 1, got {self.max_det}")

        # Validate device
        valid_devices = {"cuda", "cpu", "mps"}
        if self.device not in valid_devices:
            raise ValueError(
                f"device must be one of {valid_devices}, got '{self.device}'"
            )


class ShapeDetector:
    """YOLOv8-based shape detector for mechanical component recognition.

    Detects and classifies 20 types of mechanical components including:
    fasteners (bolt, screw, nut, washer, pin), transmission elements (bearing,
    gear, pulley, shaft), electromechanical (motor, sensor, controller),
    hydraulic/pneumatic (valve, pump, cylinder), and structural (bracket,
    housing, connector, spring, fastener).

    The detector uses lazy model loading to minimize memory usage and supports
    both single-image and batch processing modes. Implements context manager
    protocol for automatic resource cleanup.

    Attributes:
        config: Detection configuration parameters.

    Example:
        >>> config = DetectionConfig(model_path="yolov8n.pt")
        >>> detector = ShapeDetector(config)
        >>> result = detector.detect_shapes(image)
        >>> print(f"Found {result.summary.total_detections} shapes")

        >>> # Recommended for batch processing with GPU
        >>> with ShapeDetector(config) as detector:
        ...     results = detector.detect_shapes_batch(images)

    Note:
        Input images should be NumPy arrays with:
        - Format: BGR (H, W, 3), RGB (H, W, 3), or grayscale (H, W)
        - Dtype: uint8
        - Value range: [0, 255]

        Common pitfalls:
        - OpenCV loads images in BGR format by default
        - Normalized float images [0.0, 1.0] will fail validation
        - RGBA images (4 channels) are not supported
    """

    def __init__(self, config: DetectionConfig) -> None:
        """Initialize shape detector with configuration.

        Args:
            config: Detection configuration containing model path, thresholds,
                and device settings.

        Raises:
            ShapeDetectionError: If model path is invalid or inaccessible.
        """
        self.config = config

        # Validate model path
        is_valid, error_msg = validate_model_path(config.model_path)
        if not is_valid:
            raise ShapeDetectionError(error_msg)

        # Lazy load model (with thread-safe initialization)
        self._model: Optional[Any] = None
        self._model_version: Optional[str] = None
        self._actual_device: Optional[str] = None
        self._model_lock = threading.Lock()

        logger.info(f"ShapeDetector initialized (requested device: {config.device})")

    def __enter__(self) -> "ShapeDetector":
        """Enter context manager.

        Returns:
            Self for use in with statement.
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any],
    ) -> None:
        """Exit context manager and cleanup resources.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        self.cleanup()

    def cleanup(self) -> None:
        """Release model resources and free GPU memory.

        This method is called automatically when using the detector as a
        context manager. For manual resource management, call this method
        explicitly when done with the detector.

        Note:
            After calling cleanup(), the detector can still be used - the
            model will be reloaded on the next inference call.
        """
        if self._model is not None:
            logger.info("Releasing model resources")
            with self._model_lock:
                del self._model
                self._model = None
                self._actual_device = None

                # Clear GPU cache if using CUDA
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.debug("GPU cache cleared")
                except ImportError:
                    pass  # PyTorch not installed, skip GPU cleanup

    def detect_shapes(self, image: np.ndarray) -> DetectionResult:
        """Detect and classify shapes in a single image.

        Performs the full detection pipeline: model loading (if needed),
        inference, post-processing, and summary statistics calculation.

        IMPORTANT: YOLOv8's inference already applies confidence thresholding
        using config.confidence_threshold. No additional filtering is performed.

        Args:
            image: Input image as NumPy array. Must be:
                - Format: BGR (H, W, 3), RGB (H, W, 3), or grayscale (H, W)
                - Dtype: uint8
                - Value range: [0, 255]

        Returns:
            DetectionResult containing:
                - List of Detection objects with bounding boxes and confidence
                - DetectionSummary with aggregate statistics
                - Inference time in milliseconds
                - Model version identifier

        Raises:
            ShapeDetectionError: If image validation fails, model loading fails,
                or inference encounters an error. Error includes model version
                and context for debugging.

        Note:
            First call triggers lazy model loading, which may take several
            seconds. Subsequent calls use the cached model instance.
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

            # Run inference (confidence filtering applied by YOLO)
            results = self._model.predict(
                source=image,
                conf=self.config.confidence_threshold,
                iou=self.config.nms_threshold,
                max_det=self.config.max_det,
                verbose=False,
            )

            inference_time = (time.time() - start_time) * 1000  # ms

            # Process results (no additional filtering needed)
            detections = self._postprocess_detections(results[0], image.shape)

            # Calculate summary
            summary = self._calculate_summary(detections)

            result = DetectionResult(
                detections=detections,
                summary=summary,
                inference_time_ms=inference_time,
                model_version=self._model_version or "unknown",
            )

            logger.info(
                f"Shape detection complete: {len(detections)} shapes detected "
                f"in {inference_time:.1f}ms (device: {self._actual_device})"
            )

            return result

        except Exception as e:
            raise ShapeDetectionError(
                f"Shape detection failed: {e}",
                model_version=self._model_version,
            ) from e

    def _load_model(self) -> None:
        """Lazy-load YOLOv8 model from configured path.

        Loads model weights, transfers to specified device (with fallback),
        and caches model version identifier. This method is called automatically
        on first inference request.

        Raises:
            ShapeDetectionError: If ultralytics package is not installed,
                model file doesn't exist, or loading fails for any reason.

        Note:
            Model is stored in self._model for reuse across multiple detections.
            Thread-safe via lock to prevent duplicate loading.
        """
        with self._model_lock:
            # Double-check pattern: another thread may have loaded while waiting
            if self._model is not None:
                return

            try:
                from ultralytics import YOLO

                logger.info(f"Loading YOLOv8 model from {self.config.model_path}")
                self._model = YOLO(self.config.model_path)

                # Set device with fallback
                try:
                    self._model.to(self.config.device)
                    self._actual_device = self.config.device
                    logger.info(f"Model loaded successfully on {self._actual_device}")
                except Exception as device_error:
                    logger.warning(
                        f"Failed to set device to '{self.config.device}': "
                        f"{device_error}. Falling back to 'cpu'."
                    )
                    self._model.to("cpu")
                    self._actual_device = "cpu"
                    logger.info("Model loaded successfully on cpu (fallback)")

                # Get model version/name
                model_file = self.config.model_path.split("/")[-1]
                self._model_version = f"YOLOv8_{model_file}"

            except ImportError as e:
                raise ShapeDetectionError(
                    "ultralytics not installed. Install with: "
                    "pip install ultralytics"
                ) from e
            except FileNotFoundError as e:
                raise ShapeDetectionError(
                    f"Model file not found: {self.config.model_path}"
                ) from e
            except Exception as e:
                raise ShapeDetectionError(f"Failed to load model: {e}") from e

    def _postprocess_detections(
        self, results: Any, image_shape: tuple
    ) -> List[Detection]:
        """Convert YOLO raw results to Detection dataclass objects.

        Extracts bounding boxes, confidence scores, and class predictions from
        YOLO output format, converting to absolute pixel coordinates and creating
        both pixel-space and normalized bounding boxes.

        Args:
            results: YOLO results object from model.predict() containing boxes,
                scores, and class predictions. Type is ultralytics Results but
                using Any to avoid hard dependency.
            image_shape: Original image dimensions as (height, width, channels).
                Used to compute normalized bounding boxes.

        Returns:
            List of Detection objects with unique IDs, bounding boxes in both
            coordinate systems, and model version metadata. Returns empty list
            if no detections found.
        """
        detections: List[Detection] = []

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
                model_version=self._model_version or "unknown",
            )

            detections.append(detection)

        return detections

    def _calculate_summary(self, detections: List[Detection]) -> DetectionSummary:
        """Compute aggregate statistics for a set of detections.

        Uses configurable thresholds from DetectionConfig to classify
        detections into high/medium/low confidence categories.

        Args:
            detections: List of Detection objects to summarize.

        Returns:
            DetectionSummary containing:
                - total_detections: Count of all detections
                - detections_by_class: Dict mapping class names to counts
                - average_confidence: Mean confidence across all detections
                - confidence_distribution: Dict with 'high', 'medium', and 'low'
                  confidence proportions based on configurable thresholds
        """
        # Count by class
        detections_by_class: Dict[str, int] = {}
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

        # Confidence distribution using configurable thresholds
        if detections:
            high_conf = len(
                [
                    d
                    for d in detections
                    if d.confidence >= self.config.confidence_high_threshold
                ]
            )
            medium_conf = len(
                [
                    d
                    for d in detections
                    if self.config.confidence_medium_threshold
                    <= d.confidence
                    < self.config.confidence_high_threshold
                ]
            )
            low_conf = len(
                [
                    d
                    for d in detections
                    if d.confidence < self.config.confidence_medium_threshold
                ]
            )
            total = len(detections)
            confidence_distribution = {
                "high": high_conf / total,
                "medium": medium_conf / total,
                "low": low_conf / total,
            }
        else:
            confidence_distribution = {"high": 0.0, "medium": 0.0, "low": 0.0}

        return DetectionSummary(
            total_detections=len(detections),
            detections_by_class=detections_by_class,
            average_confidence=avg_confidence,
            confidence_distribution=confidence_distribution,
        )

    def detect_shapes_batch(self, images: List[np.ndarray]) -> List[DetectionResult]:
        """Detect shapes in multiple images using batch processing.

        Processes images in batches (size determined by config.batch_size) to
        improve throughput. Tracks total batch time and distributes across
        images for approximate per-image timing.

        Args:
            images: List of input images. Each image must be:
                - Format: BGR (H, W, 3), RGB (H, W, 3), or grayscale (H, W)
                - Dtype: uint8
                - Value range: [0, 255]

        Returns:
            List of DetectionResult objects, one per input image. Order matches
            input order. Failed batch images receive empty DetectionResult
            with error logged.

        Note:
            **Important timing caveat:** The `inference_time_ms` in each result
            is an APPROXIMATION calculated by dividing the total batch inference
            time equally among all images in the batch. Actual per-image times
            may vary based on image complexity, but batch processing prevents
            individual timing measurements.

            Batch processing is significantly faster than repeated single-image
            calls (typically 2-5x speedup). Individual image failures within a
            batch are logged but don't halt processing of remaining images.
        """
        if not images:
            return []

        # Check batch consistency if enabled
        if self.config.check_batch_consistency and len(images) > 1:
            first_shape = images[0].shape
            inconsistent = [
                idx for idx, img in enumerate(images) if img.shape != first_shape
            ]
            if inconsistent:
                logger.warning(
                    f"Inconsistent image shapes detected in batch. "
                    f"First image shape: {first_shape}. "
                    f"Mismatched indices: {inconsistent[:5]}"
                    f"{'...' if len(inconsistent) > 5 else ''}. "
                    f"This may cause YOLO inference errors."
                )

        # Load model if not loaded
        if self._model is None:
            self._load_model()

        results_list: List[DetectionResult] = []

        # Process in batches
        for i in range(0, len(images), self.config.batch_size):
            batch = images[i : i + self.config.batch_size]

            # Progress callback
            if self.config.progress_callback:
                self.config.progress_callback(i, len(images))

            try:
                import time

                batch_start = time.time()

                # Run batch inference (confidence filtering applied by YOLO)
                batch_results = self._model.predict(
                    source=batch,
                    conf=self.config.confidence_threshold,
                    iou=self.config.nms_threshold,
                    max_det=self.config.max_det,
                    verbose=False,
                )

                batch_time = (time.time() - batch_start) * 1000  # ms
                avg_time_per_image = batch_time / len(batch)

                # Process each result (no additional filtering)
                for idx, results in enumerate(batch_results):
                    image_shape = batch[idx].shape
                    detections = self._postprocess_detections(results, image_shape)
                    summary = self._calculate_summary(detections)

                    result = DetectionResult(
                        detections=detections,
                        summary=summary,
                        inference_time_ms=avg_time_per_image,
                        model_version=self._model_version or "unknown",
                    )
                    results_list.append(result)

            except Exception as e:
                logger.error(
                    f"Batch detection failed for batch starting at index {i}: {e}",
                    exc_info=True,
                )
                # Add empty results for failed images in this batch
                for _ in batch:
                    results_list.append(
                        DetectionResult(
                            detections=[],
                            summary=DetectionSummary(
                                total_detections=0,
                                detections_by_class={},
                                average_confidence=0.0,
                                confidence_distribution={
                                    "high": 0.0,
                                    "medium": 0.0,
                                    "low": 0.0,
                                },
                            ),
                            inference_time_ms=0.0,
                            model_version=self._model_version or "unknown",
                        )
                    )

        # Final progress callback
        if self.config.progress_callback:
            self.config.progress_callback(len(images), len(images))

        logger.info(
            f"Batch detection complete: {len(images)} images processed "
            f"(device: {self._actual_device})"
        )
        return results_list

    def get_model_info(self) -> Dict[str, Union[bool, str, int, List[str]]]:
        """Retrieve metadata about the loaded detection model.

        Returns:
            Dictionary containing:
                - loaded (bool): Whether model is currently loaded in memory
                - model_path (str): Path to model weights file
                - model_version (str): Model identifier (only if loaded)
                - requested_device (str): Device specified in config
                - actual_device (str): Device actually in use (only if loaded)
                - num_classes (int): Number of detectable classes (only if
                  loaded)
                - class_names (List[str]): Names of detectable classes (only
                  if loaded)

        Note:
            If model is not loaded, only 'loaded', 'model_path', and
            'requested_device' keys are present.
        """
        base_info: Dict[str, Union[bool, str, int, List[str]]] = {
            "loaded": self._model is not None,
            "model_path": self.config.model_path,
            "requested_device": self.config.device,
        }

        if self._model is None:
            return base_info

        base_info.update(
            {
                "model_version": self._model_version or "unknown",
                "actual_device": self._actual_device or "unknown",
                "num_classes": len(self._model.names),
                "class_names": list(self._model.names.values()),
            }
        )

        return base_info
