"""
Hierarchy builder module for the Drawing Intelligence System.

Infers assembly relationships and component hierarchies.
"""

import logging
from typing import List, Dict, Set

from ..models.data_structures import (
    Detection,
    Association,
    ComponentHierarchy,
    Assembly,
)
from ..utils.geometry_utils import bbox_contains
from ..utils.file_utils import generate_unique_id

logger = logging.getLogger(__name__)


class HierarchyBuilder:
    """
    Build component hierarchy from detected shapes.

    Infers relationships:
    - Containment (housing contains bearing)
    - Fastening (bolt fastens parts together)
    - Support (bracket supports component)
    """

    def __init__(self):
        """Initialize hierarchy builder."""
        logger.info("HierarchyBuilder initialized")

        # Define fastener component types
        self.fastener_types = {"bolt", "screw", "nut", "washer", "pin", "fastener"}

    def build_hierarchy(
        self, detections: List[Detection], associations: List[Association]
    ) -> ComponentHierarchy:
        """
        Build component hierarchy from detections.

        Args:
            detections: List of shape detections
            associations: Text-shape associations (for context)

        Returns:
            ComponentHierarchy with inferred relationships
        """
        if not detections:
            logger.warning("No detections to build hierarchy")
            return ComponentHierarchy(
                root_component_id="", assemblies=[], hierarchy_tree={}
            )

        # Find root component (largest/most central)
        root = self._find_root_component(detections, associations)

        # Build assemblies
        assemblies = []
        processed: Set[str] = set()
        processed.add(root.detection_id)

        # Find containment relationships
        for detection in detections:
            if detection.detection_id == root.detection_id:
                continue

            if bbox_contains(root.bbox, detection.bbox):
                # Root contains this component
                assembly = Assembly(
                    parent_shape_id=root.detection_id,
                    child_shape_ids=[detection.detection_id],
                    relationship_type="contains",
                    confidence=0.9,
                )
                assemblies.append(assembly)
                processed.add(detection.detection_id)

        # Find fastening relationships
        fasteners = [d for d in detections if d.class_name in self.fastener_types]
        for fastener in fasteners:
            if fastener.detection_id in processed:
                continue

            # Find what the fastener connects
            connected = self._find_connected_components(fastener, detections)

            if connected:
                assembly = Assembly(
                    parent_shape_id=fastener.detection_id,
                    child_shape_ids=[c.detection_id for c in connected],
                    relationship_type="fastens",
                    confidence=0.8,
                )
                assemblies.append(assembly)
                processed.add(fastener.detection_id)

        # Build hierarchy tree
        hierarchy_tree = self._build_tree(root, assemblies, detections)

        hierarchy = ComponentHierarchy(
            root_component_id=root.detection_id,
            assemblies=assemblies,
            hierarchy_tree=hierarchy_tree,
        )

        logger.info(
            f"Built hierarchy: root={root.class_name}, " f"{len(assemblies)} assemblies"
        )

        return hierarchy

    def _find_root_component(
        self, detections: List[Detection], associations: List[Association]
    ) -> Detection:
        """
        Find root component (largest or most associated).

        Args:
            detections: List of detections
            associations: List of associations

        Returns:
            Root detection
        """
        # Count associations per detection
        association_counts = {}
        for assoc in associations:
            association_counts[assoc.shape_id] = (
                association_counts.get(assoc.shape_id, 0) + 1
            )

        # Score each detection
        best_score = -1
        root = detections[0]

        for detection in detections:
            # Score based on size and associations
            size = detection.bbox.area()
            assoc_count = association_counts.get(detection.detection_id, 0)

            # Weighted score
            score = size * 0.7 + assoc_count * 1000 * 0.3

            if score > best_score:
                best_score = score
                root = detection

        return root

    def _find_connected_components(
        self, fastener: Detection, detections: List[Detection]
    ) -> List[Detection]:
        """
        Find components connected by a fastener.

        Args:
            fastener: Fastener detection
            detections: All detections

        Returns:
            List of connected components
        """
        from ..utils.geometry_utils import calculate_distance

        connected = []
        fastener_center = fastener.bbox.center()

        # Find nearby components (within reasonable distance)
        for detection in detections:
            if detection.detection_id == fastener.detection_id:
                continue

            if detection.class_name in self.fastener_types:
                continue

            shape_center = detection.bbox.center()
            distance = calculate_distance(fastener_center, shape_center)

            # Fastener should be close to connected components
            if distance < 200:  # pixels
                connected.append(detection)

        return connected

    def _build_tree(
        self, root: Detection, assemblies: List[Assembly], detections: List[Detection]
    ) -> Dict:
        """
        Build hierarchical tree structure.

        Args:
            root: Root component
            assemblies: List of assemblies
            detections: All detections

        Returns:
            Nested dictionary representing tree
        """
        detection_by_id = {d.detection_id: d for d in detections}

        def build_node(node_id: str, visited: Set[str]) -> Dict:
            if node_id in visited:
                return {}

            visited.add(node_id)
            detection = detection_by_id.get(node_id)

            node = {
                "id": node_id,
                "class": detection.class_name if detection else "unknown",
                "children": [],
            }

            # Find assemblies where this is parent
            for assembly in assemblies:
                if assembly.parent_shape_id == node_id:
                    for child_id in assembly.child_shape_ids:
                        child_node = build_node(child_id, visited)
                        if child_node:
                            child_node["relationship"] = assembly.relationship_type
                            node["children"].append(child_node)

            return node

        tree = build_node(root.detection_id, set())
        return tree
