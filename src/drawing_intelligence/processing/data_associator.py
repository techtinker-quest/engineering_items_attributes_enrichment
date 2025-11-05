"""
Data association module for the Drawing Intelligence System.

Links text annotations to detected shapes based on spatial relationships.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
import numpy as np

from ..models.data_structures import (
    TextBlock,
    Detection,
    Association,
    ViewGroup,
    DimensionLink,
)
from ..utils.geometry_utils import calculate_distance, bbox_overlaps, point_in_bbox
from ..utils.file_utils import generate_unique_id

logger = logging.getLogger(__name__)


@dataclass
class AssociationConfig:
    """
    Configuration for data association.

    Attributes:
        label_distance_threshold: Max distance for labels (default: 200 pixels)
        dimension_distance_threshold: Max distance for dimensions (default: 500 pixels)
        min_association_confidence: Minimum confidence for associations (default: 0.6)
        enable_obstacle_detection: Check for obstacles between text and shape (default: True)
    """

    label_distance_threshold: int = 200
    dimension_distance_threshold: int = 500
    min_association_confidence: float = 0.6
    enable_obstacle_detection: bool = True


class DataAssociator:
    """
    Link text annotations to detected shapes.

    Uses spatial proximity and relationship analysis.
    """

    def __init__(self, config: AssociationConfig):
        """
        Initialize data associator.

        Args:
            config: Association configuration
        """
        self.config = config
        logger.info("DataAssociator initialized")

    def associate_text_to_shapes(
        self, text_blocks: List[TextBlock], detections: List[Detection]
    ) -> List[Association]:
        """
        Link text blocks to nearest shapes based on proximity.

        Args:
            text_blocks: List of OCR text blocks
            detections: List of shape detections

        Returns:
            List of text-shape associations
        """
        associations = []

        if not text_blocks or not detections:
            logger.warning("No text blocks or detections to associate")
            return associations

        for text_block in text_blocks:
            # Classify text type
            text_type = self._classify_text_type(text_block.content)

            # Find nearest shape
            nearest_shape, distance = self._find_nearest_shape(text_block, detections)

            if nearest_shape is None:
                continue

            # Get distance threshold based on text type
            threshold = self._get_distance_threshold(text_type)

            if distance > threshold:
                continue

            # Check for obstacles if enabled
            if self.config.enable_obstacle_detection:
                has_obstacle = self._check_obstacles(
                    text_block, nearest_shape, detections
                )
                if has_obstacle:
                    logger.debug(
                        f"Obstacle detected between text '{text_block.content}' "
                        f"and shape {nearest_shape.class_name}"
                    )
                    continue

            # Calculate confidence based on distance
            confidence = self._calculate_association_confidence(
                distance, threshold, text_type
            )

            if confidence < self.config.min_association_confidence:
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

    def _find_nearest_shape(
        self, text_block: TextBlock, detections: List[Detection]
    ) -> Tuple[Detection, float]:
        """
        Find the nearest shape to a text block.

        Args:
            text_block: Text block to find shape for
            detections: List of shape detections

        Returns:
            Tuple of (nearest_detection, distance) or (None, inf)
        """
        text_center = text_block.bbox.center()

        min_distance = float("inf")
        nearest_shape = None

        for detection in detections:
            shape_center = detection.bbox.center()
            distance = calculate_distance(text_center, shape_center)

            if distance < min_distance:
                min_distance = distance
                nearest_shape = detection

        return nearest_shape, min_distance

    def _classify_text_type(self, content: str) -> str:
        """
        Classify text as dimension, label, or note.

        Args:
            content: Text content

        Returns:
            Type string: 'dimension', 'label', or 'note'
        """
        content_lower = content.lower()

        # Dimension patterns
        dimension_indicators = [
            "mm",
            "cm",
            "inch",
            "in",
            '"',
            "ø",
            "diameter",
            "±",
            "tolerance",
            "m",
            "r",
            "x",
            "°",
        ]

        # Check if text contains dimension indicators
        has_number = any(c.isdigit() for c in content)
        has_dim_indicator = any(ind in content_lower for ind in dimension_indicators)

        if has_number and has_dim_indicator:
            return "dimension"

        # Label patterns (short, alphabetic)
        if len(content) <= 10 and content.replace(" ", "").isalpha():
            return "label"

        # Everything else is a note
        return "note"

    def _get_distance_threshold(self, text_type: str) -> int:
        """
        Get distance threshold based on text type.

        Args:
            text_type: Type of text ('dimension', 'label', 'note')

        Returns:
            Distance threshold in pixels
        """
        if text_type == "label":
            return self.config.label_distance_threshold
        elif text_type == "dimension":
            return self.config.dimension_distance_threshold
        else:
            return self.config.label_distance_threshold

    def _check_obstacles(
        self,
        text_block: TextBlock,
        target_shape: Detection,
        all_shapes: List[Detection],
    ) -> bool:
        """
        Check if other shapes obstruct the text-shape association.

        Args:
            text_block: Text block
            target_shape: Target shape detection
            all_shapes: All shape detections

        Returns:
            True if obstacle detected
        """
        text_center = text_block.bbox.center()
        shape_center = target_shape.bbox.center()

        # Check if any other shape intersects the line between text and shape
        for shape in all_shapes:
            if shape.detection_id == target_shape.detection_id:
                continue

            # Simple check: if shape bbox contains midpoint
            midpoint = (
                (text_center[0] + shape_center[0]) // 2,
                (text_center[1] + shape_center[1]) // 2,
            )

            if point_in_bbox(midpoint, shape.bbox):
                return True

        return False

    def _calculate_association_confidence(
        self, distance: float, threshold: float, text_type: str
    ) -> float:
        """
        Calculate confidence score for association.

        Confidence decreases with distance.

        Args:
            distance: Distance in pixels
            threshold: Maximum threshold
            text_type: Type of text

        Returns:
            Confidence score (0.0-1.0)
        """
        # Base confidence: inverse of normalized distance
        normalized_distance = min(distance / threshold, 1.0)
        base_confidence = 1.0 - normalized_distance

        # Adjust based on text type
        if text_type == "dimension":
            # Dimensions can be further away
            confidence = base_confidence * 0.9
        elif text_type == "label":
            # Labels should be close
            confidence = base_confidence * 1.0
        else:
            # Notes are less reliable
            confidence = base_confidence * 0.8

        return max(0.0, min(1.0, confidence))

    def identify_multi_view_groups(
        self, detections: List[Detection]
    ) -> List[ViewGroup]:
        """
        Group shapes representing different views of same component.

        Uses spatial alignment and size similarity.

        Args:
            detections: List of shape detections

        Returns:
            List of ViewGroups
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

        # For each class, find aligned shapes
        for class_name, shapes in by_class.items():
            if len(shapes) < 2:
                continue

            # Find aligned groups
            groups = self._find_aligned_shapes(shapes)

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

    def _find_aligned_shapes(self, shapes: List[Detection]) -> List[List[Detection]]:
        """
        Find shapes that are spatially aligned.

        Args:
            shapes: List of shapes of same class

        Returns:
            List of aligned shape groups
        """
        groups = []
        used: Set[str] = set()

        for i, shape1 in enumerate(shapes):
            if shape1.detection_id in used:
                continue

            group = [shape1]
            used.add(shape1.detection_id)

            center1 = shape1.bbox.center()

            # Find other shapes aligned horizontally or vertically
            for shape2 in shapes[i + 1 :]:
                if shape2.detection_id in used:
                    continue

                center2 = shape2.bbox.center()

                # Check horizontal alignment
                y_diff = abs(center1[1] - center2[1])
                x_diff = abs(center1[0] - center2[0])

                # Aligned if Y-coords similar (horizontal alignment)
                # or X-coords similar (vertical alignment)
                if y_diff < 50 or x_diff < 50:
                    # Check size similarity
                    size1 = shape1.bbox.width * shape1.bbox.height
                    size2 = shape2.bbox.width * shape2.bbox.height
                    size_ratio = min(size1, size2) / max(size1, size2)

                    if size_ratio > 0.5:  # Similar size
                        group.append(shape2)
                        used.add(shape2.detection_id)

            if len(group) >= 2:
                groups.append(group)

        return groups

    def _infer_view_types(self, shapes: List[Detection]) -> List[str]:
        """
        Infer view types (front, side, top) based on positions.

        Args:
            shapes: List of aligned shapes

        Returns:
            List of view type strings
        """
        # Simplified inference based on position
        views = []

        if len(shapes) == 2:
            views = ["front", "side"]
        elif len(shapes) == 3:
            views = ["front", "side", "top"]
        else:
            views = [f"view_{i+1}" for i in range(len(shapes))]

        return views

    def _select_primary_shape(self, shapes: List[Detection]) -> str:
        """
        Select primary shape from group (largest/clearest).

        Args:
            shapes: List of shapes in group

        Returns:
            detection_id of primary shape
        """
        # Select largest shape
        largest = max(shapes, key=lambda s: s.bbox.area())
        return largest.detection_id

    def link_dimensions_to_features(
        self, dimensions: List, shapes: List[Detection]
    ) -> List[DimensionLink]:
        """
        Link dimension entities to shape features.

        Args:
            dimensions: List of dimension entities
            shapes: List of shape detections

        Returns:
            List of dimension-shape links
        """
        links = []

        for dimension in dimensions:
            # Find nearest shape
            dim_center = dimension.bbox.center()

            nearest_shape = None
            min_distance = float("inf")

            for shape in shapes:
                shape_center = shape.bbox.center()
                distance = calculate_distance(dim_center, shape_center)

                if distance < min_distance:
                    min_distance = distance
                    nearest_shape = shape

            if nearest_shape and min_distance < 500:
                # Infer feature type from dimension
                feature_type = self._infer_feature_type(dimension)

                # Calculate confidence
                confidence = 1.0 - (min_distance / 500.0)

                link = DimensionLink(
                    entity_id=dimension.entity_id,
                    shape_id=nearest_shape.detection_id,
                    feature_type=feature_type,
                    confidence=confidence,
                )
                links.append(link)

        logger.info(f"Created {len(links)} dimension-shape links")
        return links

    def _infer_feature_type(self, dimension) -> str:
        """
        Infer feature type from dimension entity.

        Args:
            dimension: Dimension entity

        Returns:
            Feature type string
        """
        value_str = str(dimension.value).lower()

        if "ø" in value_str or "diameter" in value_str:
            return "diameter"
        elif "r" in value_str and len(value_str) < 5:
            return "radius"
        elif "x" in value_str:
            return "length_width"
        else:
            return "dimension"
