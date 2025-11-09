"""Data association module for the Drawing Intelligence System.

Links text annotations to detected shapes using spatial relationships, multi-view
grouping, and dimension-to-feature linking with configurable confidence scoring.

This module uses spatial indexing (KD-tree), advanced obstruction detection, and
clustering algorithms (DBSCAN) for robust association in complex engineering drawings.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

from ..models.data_structures import (
    Association,
    Detection,
    DimensionLink,
    Entity,
    TextBlock,
    ViewGroup,
)
from ..utils.file_utils import generate_unique_id
from ..utils.geometry_utils import (
    bbox_overlaps,
    calculate_distance,
    point_in_bbox,
    line_intersects_bbox,
)

logger = logging.getLogger(__name__)


class ObstacleDetectionMode(Enum):
    """Obstacle detection complexity levels."""

    OFF = "off"
    SIMPLE = "simple"  # Midpoint check only
    ADVANCED = "advanced"  # Line intersection with multiple sample points


@dataclass
class AssociationConfig:
    """Configuration for data association with adaptive thresholds.

    Attributes:
        label_distance_threshold: Maximum distance for label associations (pixels).
        dimension_distance_threshold: Maximum distance for dimension associations.
        min_association_confidence: Minimum confidence score for associations.
        obstacle_detection_mode: Obstacle detection complexity level.
        alignment_tolerance_px: Tolerance for shape alignment detection.
        size_similarity_ratio: Minimum size ratio for multi-view grouping.
        max_dimension_distance_px: Maximum distance for dimension-feature links.
        confidence_multipliers: Text type confidence multipliers (dimension, label, note).
        dbscan_eps: DBSCAN epsilon parameter for shape clustering.
        dbscan_min_samples: DBSCAN minimum samples for cluster formation.
        adaptive_thresholds: Enable DPI-based threshold scaling.
        drawing_dpi: Drawing DPI for threshold normalization (if adaptive).
    """

    label_distance_threshold: int = 200
    dimension_distance_threshold: int = 500
    min_association_confidence: float = 0.6
    obstacle_detection_mode: ObstacleDetectionMode = ObstacleDetectionMode.SIMPLE
    alignment_tolerance_px: int = 50
    size_similarity_ratio: float = 0.5
    max_dimension_distance_px: int = 500
    confidence_multipliers: Dict[str, float] = field(
        default_factory=lambda: {"dimension": 0.9, "label": 1.0, "note": 0.8}
    )
    dbscan_eps: float = 100.0
    dbscan_min_samples: int = 2
    adaptive_thresholds: bool = False
    drawing_dpi: int = 300

    def __post_init__(self) -> None:
        """Normalize thresholds if adaptive mode is enabled."""
        if self.adaptive_thresholds and self.drawing_dpi != 300:
            scale_factor = self.drawing_dpi / 300.0
            self.label_distance_threshold = int(
                self.label_distance_threshold * scale_factor
            )
            self.dimension_distance_threshold = int(
                self.dimension_distance_threshold * scale_factor
            )
            self.max_dimension_distance_px = int(
                self.max_dimension_distance_px * scale_factor
            )
            self.alignment_tolerance_px = int(
                self.alignment_tolerance_px * scale_factor
            )
            logger.info(
                f"Thresholds scaled by {scale_factor:.2f}x for {self.drawing_dpi} DPI"
            )


class DataAssociator:
    """Link text annotations to detected shapes using advanced spatial analysis.

    Uses KD-tree spatial indexing for efficient nearest-neighbor search,
    configurable obstacle detection, DBSCAN clustering for multi-view grouping,
    and multi-factor confidence scoring.

    Attributes:
        config: Association configuration parameters.
        _dimension_pattern: Compiled regex for dimension detection.
        _radius_pattern: Compiled regex for radius detection.
        _diameter_pattern: Compiled regex for diameter detection.

    Example:
        >>> config = AssociationConfig(
        ...     label_distance_threshold=150,
        ...     obstacle_detection_mode=ObstacleDetectionMode.ADVANCED
        ... )
        >>> associator = DataAssociator(config)
        >>> associations = associator.associate_text_to_shapes(texts, shapes)
    """

    def __init__(self, config: AssociationConfig) -> None:
        """Initialize data associator with compiled patterns.

        Args:
            config: Association configuration parameters.
        """
        self.config = config

        # Compile regex patterns for feature inference
        self._dimension_pattern = re.compile(r"\d+\.?\d*\s*(mm|cm|in|inch|\")")
        self._radius_pattern = re.compile(r"r\s*\d+\.?\d*", re.IGNORECASE)
        self._diameter_pattern = re.compile(
            r"(ø|diameter|dia\.?)\s*\d+\.?\d*", re.IGNORECASE
        )

        logger.info(
            f"DataAssociator initialized with {config.obstacle_detection_mode.value} "
            f"obstacle detection"
        )

    def associate_text_to_shapes(
        self, text_blocks: List[TextBlock], detections: List[Detection]
    ) -> List[Association]:
        """Link text blocks to nearest shapes using spatial indexing.

        Uses KD-tree for efficient nearest-neighbor search and multi-factor
        confidence scoring including distance, size ratio, and semantic compatibility.

        Args:
            text_blocks: OCR-extracted text blocks with entity types.
            detections: Shape detections from YOLO model.

        Returns:
            List of Association objects with confidence scores and relationship types.

        Raises:
            ValueError: If inputs are invalid or empty.
        """
        # Validate inputs
        if not text_blocks:
            logger.warning("No text blocks provided for association")
            return []
        if not detections:
            logger.warning("No detections provided for association")
            return []

        associations = []

        # Build KD-tree for spatial indexing
        shape_centers = np.array([det.bbox.center() for det in detections])
        kdtree = KDTree(shape_centers)

        for text_block in text_blocks:
            # Use entity type from EntityExtractor instead of heuristic classification
            text_type = self._get_entity_type(text_block)

            # Find nearest shape using KD-tree
            nearest_shape, distance = self._find_nearest_shape_kdtree(
                text_block, detections, kdtree
            )

            if nearest_shape is None:
                logger.debug(f"No valid shape found for text '{text_block.content}'")
                continue

            # Get distance threshold based on text type
            threshold = self._get_distance_threshold(text_type)

            if distance > threshold:
                logger.debug(
                    f"Text '{text_block.content}' exceeds threshold "
                    f"({distance:.1f} > {threshold}px)"
                )
                continue

            # Check for obstacles if enabled
            if self.config.obstacle_detection_mode != ObstacleDetectionMode.OFF:
                has_obstacle = self._check_obstacles(
                    text_block, nearest_shape, detections
                )
                if has_obstacle:
                    logger.debug(
                        f"Obstacle detected between text '{text_block.content}' "
                        f"and shape {nearest_shape.class_name}"
                    )
                    continue

            # Calculate multi-factor confidence
            confidence = self._calculate_association_confidence(
                text_block, nearest_shape, distance, threshold, text_type
            )

            if confidence < self.config.min_association_confidence:
                logger.debug(
                    f"Low confidence ({confidence:.2f}) for "
                    f"'{text_block.content}' → {nearest_shape.class_name}"
                )
                continue

            # Create association
            association = Association(
                association_id=generate_unique_id("ASSOC"),
                text_id=text_block.text_id,
                shape_id=nearest_shape.detection_id,
                relationship_type=text_type,
                confidence=confidence,
                distance_pixels=distance,
            )

            associations.append(association)
            logger.debug(
                f"Associated '{text_block.content}' → {nearest_shape.class_name} "
                f"(distance: {distance:.1f}px, conf: {confidence:.2f})"
            )

        logger.info(f"Created {len(associations)} text-shape associations")
        return associations

    def _get_entity_type(self, text_block: TextBlock) -> str:
        """Extract entity type from TextBlock (set by EntityExtractor).

        Args:
            text_block: Text block with entity metadata.

        Returns:
            Entity type string: 'dimension', 'label', 'note', or fallback.
        """
        # Check if entity_type attribute exists (from EntityExtractor)
        if hasattr(text_block, "entity_type") and text_block.entity_type:
            return text_block.entity_type.lower()

        # Fallback: Use content-based classification only if entity_type missing
        logger.warning(
            f"Text block '{text_block.content}' missing entity_type, using fallback"
        )
        return self._classify_text_type_fallback(text_block.content)

    def _classify_text_type_fallback(self, content: str) -> str:
        """Fallback text classification using regex patterns.

        Args:
            content: Text string to classify.

        Returns:
            Classification string: 'dimension', 'label', or 'note'.
        """
        content_lower = content.lower()

        # Check for dimension pattern
        if self._dimension_pattern.search(content):
            return "dimension"

        # Label patterns (short, alphabetic)
        if len(content) <= 10 and content.replace(" ", "").isalpha():
            return "label"

        # Everything else is a note
        return "note"

    def _find_nearest_shape_kdtree(
        self,
        text_block: TextBlock,
        detections: List[Detection],
        kdtree: KDTree,
    ) -> Tuple[Optional[Detection], float]:
        """Find nearest shape using KD-tree spatial index.

        Calculates distance from text center to nearest point on shape edge
        (not center-to-center) for improved accuracy.

        Args:
            text_block: Text block to find nearest shape for.
            detections: List of all shape detections.
            kdtree: Pre-built KD-tree of shape centers.

        Returns:
            Tuple containing nearest Detection and edge distance (or None, inf).
        """
        if not detections:
            return None, float("inf")

        text_center = np.array(text_block.bbox.center())

        # Query KD-tree for k nearest candidates
        k = min(5, len(detections))
        distances_center, indices = kdtree.query(text_center, k=k)

        # Refine by calculating distance to shape edge
        min_edge_distance = float("inf")
        nearest_shape = None

        for idx in indices:
            detection = detections[idx]
            edge_distance = self._distance_to_bbox_edge(text_center, detection.bbox)

            if edge_distance < min_edge_distance:
                min_edge_distance = edge_distance
                nearest_shape = detection

        return nearest_shape, min_edge_distance

    def _distance_to_bbox_edge(self, point: np.ndarray, bbox) -> float:
        """Calculate minimum distance from point to bounding box edge.

        Args:
            point: (x, y) coordinates as numpy array.
            bbox: BoundingBox object.

        Returns:
            Minimum distance to bbox edge in pixels.
        """
        px, py = point

        # Check if point is inside bbox
        if bbox.x_min <= px <= bbox.x_max and bbox.y_min <= py <= bbox.y_max:
            # Point inside: distance to nearest edge
            return min(
                px - bbox.x_min,
                bbox.x_max - px,
                py - bbox.y_min,
                bbox.y_max - py,
            )

        # Point outside: calculate perpendicular distance to edges
        dx = max(bbox.x_min - px, 0, px - bbox.x_max)
        dy = max(bbox.y_min - py, 0, py - bbox.y_max)

        return np.sqrt(dx**2 + dy**2)

    def _get_distance_threshold(self, text_type: str) -> int:
        """Get distance threshold based on text type.

        Args:
            text_type: Type of text ('dimension', 'label', 'note').

        Returns:
            Distance threshold in pixels.
        """
        if text_type == "label":
            return self.config.label_distance_threshold
        elif text_type == "dimension":
            return self.config.dimension_distance_threshold
        else:
            # Notes use label threshold as default
            return self.config.label_distance_threshold

    def _check_obstacles(
        self,
        text_block: TextBlock,
        target_shape: Detection,
        all_shapes: List[Detection],
    ) -> bool:
        """Check if shapes obstruct text-to-shape line of sight.

        Uses configurable detection mode: simple (midpoint) or advanced
        (line intersection with multiple sample points).

        Args:
            text_block: Source text block.
            target_shape: Target shape for association.
            all_shapes: All shape detections to check as obstacles.

        Returns:
            True if obstruction detected, False otherwise.
        """
        text_center = text_block.bbox.center()
        shape_center = target_shape.bbox.center()

        if self.config.obstacle_detection_mode == ObstacleDetectionMode.SIMPLE:
            return self._check_obstacles_simple(
                text_center, shape_center, target_shape, all_shapes
            )
        else:  # ADVANCED
            return self._check_obstacles_advanced(
                text_center, shape_center, target_shape, all_shapes
            )

    def _check_obstacles_simple(
        self,
        text_center: Tuple[int, int],
        shape_center: Tuple[int, int],
        target_shape: Detection,
        all_shapes: List[Detection],
    ) -> bool:
        """Simple obstacle detection using midpoint check.

        Args:
            text_center: Text block center coordinates.
            shape_center: Target shape center coordinates.
            target_shape: Target shape detection.
            all_shapes: All shapes to check as obstacles.

        Returns:
            True if midpoint falls inside an obstacle bbox.
        """
        midpoint = (
            (text_center[0] + shape_center[0]) // 2,
            (text_center[1] + shape_center[1]) // 2,
        )

        for shape in all_shapes:
            if shape.detection_id == target_shape.detection_id:
                continue

            if point_in_bbox(midpoint, shape.bbox):
                return True

        return False

    def _check_obstacles_advanced(
        self,
        text_center: Tuple[int, int],
        shape_center: Tuple[int, int],
        target_shape: Detection,
        all_shapes: List[Detection],
    ) -> bool:
        """Advanced obstacle detection using line intersection.

        Samples multiple points along text-to-shape line and checks for
        line segment intersection with obstacle bounding boxes.

        Args:
            text_center: Text block center coordinates.
            shape_center: Target shape center coordinates.
            target_shape: Target shape detection.
            all_shapes: All shapes to check as obstacles.

        Returns:
            True if line intersects any obstacle bbox.
        """
        # Sample 5 points along the line
        sample_points = []
        for t in np.linspace(0, 1, 5):
            x = int(text_center[0] + t * (shape_center[0] - text_center[0]))
            y = int(text_center[1] + t * (shape_center[1] - text_center[1]))
            sample_points.append((x, y))

        for shape in all_shapes:
            if shape.detection_id == target_shape.detection_id:
                continue

            # Check if any sample point is inside obstacle
            for point in sample_points:
                if point_in_bbox(point, shape.bbox):
                    return True

            # Check for line-bbox intersection
            if line_intersects_bbox(text_center, shape_center, shape.bbox):
                return True

        return False

    def _calculate_association_confidence(
        self,
        text_block: TextBlock,
        shape: Detection,
        distance: float,
        threshold: float,
        text_type: str,
    ) -> float:
        """Calculate multi-factor confidence score for association.

        Considers:
        1. Distance (exponential decay)
        2. Text-to-shape size ratio
        3. Alignment quality
        4. Semantic compatibility

        Args:
            text_block: Source text block.
            shape: Target shape detection.
            distance: Distance between text and shape edge.
            threshold: Maximum allowed distance.
            text_type: Text classification type.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        # Factor 1: Distance confidence (exponential decay)
        normalized_distance = min(distance / threshold, 1.0)
        distance_conf = np.exp(-3 * normalized_distance)

        # Factor 2: Size ratio confidence
        text_area = text_block.bbox.area()
        shape_area = shape.bbox.area()
        size_ratio = min(text_area, shape_area) / max(text_area, shape_area)
        size_conf = min(1.0, size_ratio + 0.5)  # Boost small ratios

        # Factor 3: Alignment quality (check if text is aligned with shape edges)
        alignment_conf = self._calculate_alignment_confidence(text_block, shape)

        # Factor 4: Semantic compatibility
        semantic_conf = self._calculate_semantic_confidence(text_type, shape.class_name)

        # Weighted combination
        base_confidence = (
            0.5 * distance_conf
            + 0.2 * size_conf
            + 0.15 * alignment_conf
            + 0.15 * semantic_conf
        )

        # Apply text type multiplier from config
        multiplier = self.config.confidence_multipliers.get(text_type, 0.8)
        final_confidence = base_confidence * multiplier

        return max(0.0, min(1.0, final_confidence))

    def _calculate_alignment_confidence(
        self, text_block: TextBlock, shape: Detection
    ) -> float:
        """Calculate alignment quality between text and shape.

        Args:
            text_block: Text block.
            shape: Shape detection.

        Returns:
            Alignment confidence (0.0-1.0).
        """
        text_center = text_block.bbox.center()
        shape_bbox = shape.bbox

        # Check horizontal/vertical alignment with shape edges
        h_aligned = (
            abs(text_center[1] - shape_bbox.y_min) < 20
            or abs(text_center[1] - shape_bbox.y_max) < 20
        )
        v_aligned = (
            abs(text_center[0] - shape_bbox.x_min) < 20
            or abs(text_center[0] - shape_bbox.x_max) < 20
        )

        if h_aligned or v_aligned:
            return 1.0
        else:
            return 0.5  # Neutral for non-aligned

    def _calculate_semantic_confidence(self, text_type: str, shape_class: str) -> float:
        """Calculate semantic compatibility between text type and shape class.

        Args:
            text_type: Text classification type.
            shape_class: Shape class name.

        Returns:
            Semantic confidence (0.0-1.0).
        """
        # Define semantic compatibility rules
        if text_type == "dimension":
            # Dimensions compatible with all shapes
            return 1.0
        elif text_type == "label":
            # Labels prefer specific components (bolts, gears, etc.)
            if shape_class.lower() in [
                "bolt",
                "nut",
                "screw",
                "gear",
                "bearing",
            ]:
                return 1.0
            else:
                return 0.7
        else:  # note
            # Notes are generic
            return 0.6

    def identify_multi_view_groups(
        self, detections: List[Detection]
    ) -> List[ViewGroup]:
        """Group shapes representing orthographic views using DBSCAN clustering.

        Uses DBSCAN to cluster shapes of the same class that are spatially
        aligned and have similar sizes, indicating multi-view representations.

        Args:
            detections: All shape detections from drawing.

        Returns:
            List of ViewGroup objects with inferred view types.

        Note:
            DBSCAN parameters (eps, min_samples) are configurable via
            AssociationConfig for tuning to specific drawing styles.
        """
        view_groups = []

        if len(detections) < 2:
            return view_groups

        # Group by class
        by_class: Dict[str, List[Detection]] = {}
        for det in detections:
            if det.class_name not in by_class:
                by_class[det.class_name] = []
            by_class[det.class_name].append(det)

        # For each class, find aligned groups using DBSCAN
        for class_name, shapes in by_class.items():
            if len(shapes) < 2:
                continue

            # Cluster shapes using DBSCAN
            groups = self._cluster_shapes_dbscan(shapes)

            for group in groups:
                if len(group) >= 2:
                    # Create view group
                    view_group = ViewGroup(
                        group_id=generate_unique_id("VIEW"),
                        component_class=class_name,
                        shape_ids=[s.detection_id for s in group],
                        views=self._infer_view_types(group),
                        primary_shape_id=self._select_primary_shape(group),
                    )
                    view_groups.append(view_group)

        logger.info(f"Identified {len(view_groups)} multi-view groups")
        return view_groups

    def _cluster_shapes_dbscan(self, shapes: List[Detection]) -> List[List[Detection]]:
        """Cluster shapes using DBSCAN based on spatial features.

        Args:
            shapes: List of shapes of same class.

        Returns:
            List of shape clusters (groups).
        """
        if len(shapes) < 2:
            return []

        # Extract features: center coordinates + normalized size
        features = []
        for shape in shapes:
            center = shape.bbox.center()
            area = shape.bbox.area()
            features.append([center[0], center[1], np.sqrt(area)])

        features = np.array(features)

        # Normalize features for consistent clustering
        features_normalized = (features - features.mean(axis=0)) / (
            features.std(axis=0) + 1e-8
        )

        # Apply DBSCAN
        clustering = DBSCAN(
            eps=self.config.dbscan_eps / 100.0,  # Normalized scale
            min_samples=self.config.dbscan_min_samples,
        ).fit(features_normalized)

        # Group shapes by cluster label
        clusters: Dict[int, List[Detection]] = {}
        for idx, label in enumerate(clustering.labels_):
            if label == -1:  # Noise
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(shapes[idx])

        # Filter clusters by size similarity
        valid_groups = []
        for cluster_shapes in clusters.values():
            if self._shapes_have_similar_size(cluster_shapes):
                valid_groups.append(cluster_shapes)

        return valid_groups

    def _shapes_have_similar_size(self, shapes: List[Detection]) -> bool:
        """Check if shapes in group have similar sizes.

        Args:
            shapes: List of shapes to check.

        Returns:
            True if all shapes have similar size ratios.
        """
        if len(shapes) < 2:
            return True

        areas = [s.bbox.area() for s in shapes]
        min_area = min(areas)
        max_area = max(areas)

        size_ratio = min_area / max_area
        return size_ratio >= self.config.size_similarity_ratio

    def _infer_view_types(self, shapes: List[Detection]) -> List[str]:
        """Infer view types based on relative spatial positions.

        Args:
            shapes: List of aligned shapes.

        Returns:
            List of view type strings based on positioning.
        """
        if len(shapes) == 1:
            return ["single"]

        # Sort shapes by position
        shapes_sorted = sorted(shapes, key=lambda s: s.bbox.center())

        # Determine layout (horizontal or vertical)
        x_coords = [s.bbox.center()[0] for s in shapes]
        y_coords = [s.bbox.center()[1] for s in shapes]

        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)

        if x_range > y_range:
            # Horizontal layout
            if len(shapes) == 2:
                return ["front", "side"]
            else:
                return ["front", "side", "top"][: len(shapes)]
        else:
            # Vertical layout
            if len(shapes) == 2:
                return ["top", "front"]
            else:
                return ["top", "front", "side"][: len(shapes)]

    def _select_primary_shape(self, shapes: List[Detection]) -> str:
        """Select primary shape from group (largest area).

        Args:
            shapes: List of shapes in group.

        Returns:
            detection_id of primary shape.
        """
        largest = max(shapes, key=lambda s: s.bbox.area())
        return largest.detection_id

    def link_dimensions_to_features(
        self, dimensions: List[Entity], shapes: List[Detection]
    ) -> List[DimensionLink]:
        """Link dimension entities to shape features with contextual inference.

        Uses shape geometry, dimension orientation, and spatial context to
        infer which feature (diameter, radius, length) the dimension describes.

        Args:
            dimensions: Dimension entities extracted by EntityExtractor.
            shapes: Shape detections from YOLO model.

        Returns:
            List of DimensionLink objects with confidence scores.

        Note:
            Uses configurable max_dimension_distance_px threshold. Feature
            type inference considers both dimension text and shape context.
        """
        links = []

        if not dimensions or not shapes:
            logger.warning("No dimensions or shapes for linking")
            return links

        # Build KD-tree for efficient search
        shape_centers = np.array([s.bbox.center() for s in shapes])
        kdtree = KDTree(shape_centers)

        for dimension in dimensions:
            dim_center = np.array(dimension.bbox.center())

            # Query nearest shapes
            k = min(3, len(shapes))
            distances, indices = kdtree.query(dim_center, k=k)

            nearest_shape = None
            min_distance = float("inf")

            # Find nearest shape within threshold
            for dist, idx in zip(distances, indices):
                if dist < self.config.max_dimension_distance_px:
                    if dist < min_distance:
                        min_distance = dist
                        nearest_shape = shapes[idx]

            if nearest_shape:
                # Infer feature type using context
                feature_type = self._infer_feature_type_contextual(
                    dimension, nearest_shape
                )

                # Calculate confidence (exponential decay)
                normalized_dist = min_distance / self.config.max_dimension_distance_px
                confidence = np.exp(-2 * normalized_dist)

                link = DimensionLink(
                    entity_id=dimension.entity_id,
                    shape_id=nearest_shape.detection_id,
                    feature_type=feature_type,
                    confidence=confidence,
                )
                links.append(link)
                logger.debug(
                    f"Linked dimension '{dimension.value}' → "
                    f"{nearest_shape.class_name} ({feature_type})"
                )

        logger.info(f"Created {len(links)} dimension-shape links")
        return links

    def _infer_feature_type_contextual(
        self, dimension: Entity, shape: Detection
    ) -> str:
        """Infer feature type using dimension text and shape context.

        Args:
            dimension: Dimension entity.
            shape: Associated shape detection.

        Returns:
            Feature type string ('diameter', 'radius', 'length_width', etc.).
        """
        value_str = str(dimension.value).lower()

        # Check explicit patterns first
        if self._diameter_pattern.search(value_str):
            return "diameter"
        elif self._radius_pattern.search(value_str):
            return "radius"

        # Contextual inference based on shape class
        shape_class = shape.class_name.lower()

        # Circular shapes likely have diameter/radius
        if shape_class in ["circle", "hole", "cylinder", "shaft"]:
            if "x" in value_str:
                return "length_width"
            else:
                return "diameter"

        # Rectangular shapes likely have length/width
        if shape_class in ["rectangle", "block", "plate"]:
            if "x" in value_str:
                return "length_width"
            else:
                return "dimension"

        # Threaded features
        if shape_class in ["bolt", "screw", "thread"]:
            if "m" in value_str or "thread" in value_str:
                return "thread_size"
            else:
                return "length"

        # Default: generic dimension
        if "x" in value_str:
            return "length_width"
        else:
            return "dimension"
