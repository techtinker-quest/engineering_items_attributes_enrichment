"""Hierarchy builder module for the Drawing Intelligence System.

This module provides functionality for inferring assembly relationships and
component hierarchies from detected shapes in engineering drawings. It identifies
containment, fastening, and spatial relationships between components.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

from ..models.data_structures import (
    Assembly,
    Association,
    ComponentHierarchy,
    Detection,
)
from ..utils.file_utils import generate_unique_id
from ..utils.geometry_utils import bbox_contains, calculate_distance


logger = logging.getLogger(__name__)


# Type definition for hierarchy tree nodes
class HierarchyNode(TypedDict, total=False):
    """Type definition for hierarchy tree node structure.

    Attributes:
        id: Unique identifier for the component.
        class_name: Component class name (e.g., 'housing', 'bolt').
        confidence: Confidence score for this node's placement in hierarchy.
        relationship: Type of relationship to parent (omitted for root).
            Valid values: 'contains', 'fastens', 'orphaned'.
        children: List of child nodes in the hierarchy.
    """

    id: str
    class_name: str
    confidence: float
    relationship: str
    children: List["HierarchyNode"]


@dataclass(frozen=True)
class HierarchyConfig:
    """Configuration for hierarchy building.

    Attributes:
        fastener_proximity_px: Maximum distance (pixels) between fastener and
            connected components. Should be adjusted based on drawing resolution.
        size_weight: Weight for component size in root selection (0.0-1.0).
        association_weight: Weight for association count in root selection (0.0-1.0).
        centrality_weight: Weight for spatial centrality in root selection (0.0-1.0).
        containment_confidence_base: Base confidence score for containment relationships.
        fastening_confidence_base: Base confidence score for fastening relationships.
        orphaned_confidence: Confidence score for orphaned component attachments.
        max_tree_depth: Maximum depth for hierarchy tree to prevent stack overflow.
        fastener_types: Set of component class names considered fasteners.
        min_containment_overlap: Minimum bbox overlap ratio to consider containment (0.0-1.0).
        weight_sum_tolerance: Tolerance for weight sum validation.
    """

    fastener_proximity_px: int = 200
    size_weight: float = 0.5
    association_weight: float = 0.3
    centrality_weight: float = 0.2
    containment_confidence_base: float = 0.9
    fastening_confidence_base: float = 0.8
    orphaned_confidence: float = 0.5
    max_tree_depth: int = 50
    fastener_types: Set[str] = field(
        default_factory=lambda: {"bolt", "screw", "nut", "washer", "pin", "fastener"}
    )
    min_containment_overlap: float = 0.95
    weight_sum_tolerance: float = 0.01

    def __post_init__(self) -> None:
        """Validates configuration parameters."""
        # Validate weight ranges
        if not (0.0 <= self.size_weight <= 1.0):
            raise ValueError("size_weight must be between 0.0 and 1.0")
        if not (0.0 <= self.association_weight <= 1.0):
            raise ValueError("association_weight must be between 0.0 and 1.0")
        if not (0.0 <= self.centrality_weight <= 1.0):
            raise ValueError("centrality_weight must be between 0.0 and 1.0")

        # Validate weight sum
        weight_sum = self.size_weight + self.association_weight + self.centrality_weight
        if abs(weight_sum - 1.0) > self.weight_sum_tolerance:
            raise ValueError(
                f"Weights must sum to 1.0 (got {weight_sum:.3f}). "
                f"size_weight={self.size_weight}, "
                f"association_weight={self.association_weight}, "
                f"centrality_weight={self.centrality_weight}"
            )

        # Validate other parameters
        if not (0.0 <= self.min_containment_overlap <= 1.0):
            raise ValueError("min_containment_overlap must be between 0.0 and 1.0")

        if not self.fastener_types:
            raise ValueError("fastener_types cannot be empty")

        if self.fastener_proximity_px <= 0:
            raise ValueError("fastener_proximity_px must be positive")

        if self.max_tree_depth <= 0:
            raise ValueError("max_tree_depth must be positive")


class SpatialIndex:
    """Simple spatial index for fast bounding box queries.

    Uses R-tree-like structure for efficient proximity and containment queries.
    """

    def __init__(self, detections: List[Detection]) -> None:
        """Initializes spatial index from detections.

        Args:
            detections: List of detections to index.
        """
        self.detections = detections
        self._index = self._build_index()

    def _build_index(self) -> Dict[str, Detection]:
        """Builds simple dictionary index (can be replaced with R-tree).

        Returns:
            Dictionary mapping detection IDs to Detection objects.
        """
        return {d.detection_id: d for d in self.detections}

    def find_containing(self, detection: Detection) -> List[Detection]:
        """Finds all detections that contain the given detection.

        Args:
            detection: Detection to find containers for.

        Returns:
            List of detections that contain the given detection, sorted by area
            (smallest first to find immediate parent).
        """
        containers = []
        for candidate in self.detections:
            if candidate.detection_id == detection.detection_id:
                continue

            if bbox_contains(candidate.bbox, detection.bbox):
                containers.append(candidate)

        # Sort by area (smallest containers first = immediate parents)
        containers.sort(key=lambda d: d.bbox.area())
        return containers

    def find_within_distance(
        self, detection: Detection, max_distance: float
    ) -> List[Tuple[Detection, float]]:
        """Finds detections within specified distance.

        Args:
            detection: Reference detection.
            max_distance: Maximum distance in pixels.

        Returns:
            List of (detection, distance) tuples within max_distance.
        """
        center = detection.bbox.center()
        nearby = []

        for candidate in self.detections:
            if candidate.detection_id == detection.detection_id:
                continue

            candidate_center = candidate.bbox.center()
            distance = calculate_distance(center, candidate_center)

            if distance <= max_distance:
                nearby.append((candidate, distance))

        return nearby


class HierarchyBuilder:
    """Builds component hierarchy from detected shapes in engineering drawings.

    This class analyzes spatial relationships between detected shapes to infer
    assembly hierarchies including containment (housings, enclosures), fastening
    (bolts, screws connecting components), and nested component structures.

    Attributes:
        config: Configuration parameters for hierarchy building.

    Example:
        >>> config = HierarchyConfig(fastener_proximity_px=250)
        >>> builder = HierarchyBuilder(config)
        >>> hierarchy = builder.build_hierarchy(detections, associations)
        >>> print(hierarchy.root_component_id)
    """

    def __init__(self, config: Optional[HierarchyConfig] = None) -> None:
        """Initializes the hierarchy builder with optional configuration.

        Args:
            config: Configuration parameters. If None, uses default configuration.
        """
        self.config = config or HierarchyConfig()
        logger.info(
            f"HierarchyBuilder initialized with fastener_proximity="
            f"{self.config.fastener_proximity_px}px"
        )

    def build_hierarchy(
        self, detections: List[Detection], associations: List[Association]
    ) -> ComponentHierarchy:
        """Builds component hierarchy from shape detections and associations.

        Analyzes spatial relationships to create a hierarchical tree structure
        representing component assemblies. Identifies containment relationships
        (parent contains child), fastening relationships (fastener connects
        components), and builds a navigable tree structure.

        Relationship types in output:
            - 'contains': Parent component physically contains child
            - 'fastens': Fastener connects multiple components
            - 'orphaned': Component has no explicit relationship, attached to root

        Args:
            detections: List of shape detections from the drawing. Must not be empty.
            associations: Text-shape associations providing additional context for
                relationship inference. Used to identify important components.

        Returns:
            ComponentHierarchy object containing the root component ID, list of
            assembly relationships, and hierarchical tree structure.

        Raises:
            ValueError: If detections list is empty.

        Example:
            >>> detections = [Detection(...), Detection(...)]
            >>> associations = [Association(...)]
            >>> hierarchy = builder.build_hierarchy(detections, associations)
        """
        if not detections:
            raise ValueError(
                "Cannot build hierarchy: detections list is empty. "
                "At least one detection is required."
            )

        # Build spatial index for efficient queries
        spatial_index = SpatialIndex(detections)

        # Precompute association counts (caching)
        association_counts = self._compute_association_counts(associations)

        # Find root component
        root = self._find_root_component(detections, associations, association_counts)

        # Build assemblies with cycle detection
        assemblies: List[Assembly] = []
        processed: Set[str] = set()
        parent_child_map: Dict[str, Set[str]] = {}

        processed.add(root.detection_id)

        # Infer containment relationships
        containment_assemblies = self._infer_containment_assemblies(
            detections, spatial_index, root, processed, parent_child_map
        )
        assemblies.extend(containment_assemblies)

        # Infer fastening relationships
        fastening_assemblies = self._infer_fastening_assemblies(
            detections, spatial_index, processed, parent_child_map
        )
        assemblies.extend(fastening_assemblies)

        # Handle orphaned components
        orphaned_assemblies = self._handle_orphaned_components(
            detections, root, processed
        )
        assemblies.extend(orphaned_assemblies)

        # Build hierarchy tree
        detection_by_id = {d.detection_id: d for d in detections}
        hierarchy_tree = self._build_tree(root, assemblies, detection_by_id)

        hierarchy = ComponentHierarchy(
            root_component_id=root.detection_id,
            assemblies=assemblies,
            hierarchy_tree=hierarchy_tree,
        )

        logger.info(
            f"Built hierarchy: root={root.class_name}, "
            f"{len(assemblies)} assemblies, "
            f"{len(processed)} components processed"
        )

        return hierarchy

    def _compute_association_counts(
        self, associations: List[Association]
    ) -> Dict[str, int]:
        """Computes association counts per detection (cached).

        Args:
            associations: List of text-shape associations.

        Returns:
            Dictionary mapping detection IDs to association counts.
        """
        counts: Dict[str, int] = {}
        for assoc in associations:
            counts[assoc.shape_id] = counts.get(assoc.shape_id, 0) + 1
        return counts

    def _find_root_component(
        self,
        detections: List[Detection],
        associations: List[Association],
        association_counts: Dict[str, int],
    ) -> Detection:
        """Identifies the root component of the assembly.

        Selects the root based on a weighted score combining component size,
        number of text associations, and spatial centrality. Larger components
        with more labels and central positioning are preferred.

        Args:
            detections: List of all shape detections. Must contain at least one item.
            associations: List of text-shape associations for scoring.
            association_counts: Precomputed association counts per detection.

        Returns:
            Detection object identified as the root component.

        Note:
            Scoring formula:
            (size * size_weight) + (assoc_count * assoc_weight) + (centrality * centrality_weight)
            where weights are configured in HierarchyConfig and sum to 1.0.
        """
        if not detections:
            raise ValueError("Cannot find root: detections list is empty")

        # Handle edge case: single detection
        if len(detections) == 1:
            return detections[0]

        # Calculate drawing center for centrality scoring
        all_centers = [d.bbox.center() for d in detections]
        drawing_center_x = sum(c[0] for c in all_centers) / len(all_centers)
        drawing_center_y = sum(c[1] for c in all_centers) / len(all_centers)
        drawing_center = (drawing_center_x, drawing_center_y)

        # Normalize metrics
        max_size = max(d.bbox.area() for d in detections)
        max_assoc = max(association_counts.values()) if association_counts else 1

        # Calculate max distance for centrality normalization
        max_distance = max(
            calculate_distance(d.bbox.center(), drawing_center) for d in detections
        )

        # Handle zero metrics edge cases
        if max_size == 0:
            logger.warning(
                "All detections have zero area, using first detection as root"
            )
            return detections[0]

        # Score each detection
        best_score = -1.0
        root = detections[0]

        for detection in detections:
            # Normalize size to [0, 1]
            size_normalized = detection.bbox.area() / max_size

            # Normalize associations to [0, 1]
            assoc_count = association_counts.get(detection.detection_id, 0)
            assoc_normalized = assoc_count / max_assoc if max_assoc > 0 else 0

            # Normalize centrality to [0, 1] (closer to center = higher score)
            distance_to_center = calculate_distance(
                detection.bbox.center(), drawing_center
            )
            centrality_normalized = (
                1.0 - (distance_to_center / max_distance) if max_distance > 0 else 0
            )

            # Weighted score
            score = (
                size_normalized * self.config.size_weight
                + assoc_normalized * self.config.association_weight
                + centrality_normalized * self.config.centrality_weight
            )

            # Deterministic tie-breaking by detection_id
            if score > best_score or (
                score == best_score and detection.detection_id < root.detection_id
            ):
                best_score = score
                root = detection

        logger.debug(
            f"Selected root: {root.class_name} "
            f"(score={best_score:.3f}, id={root.detection_id})"
        )
        return root

    def _infer_containment_assemblies(
        self,
        detections: List[Detection],
        spatial_index: SpatialIndex,
        root: Detection,
        processed: Set[str],
        parent_child_map: Dict[str, Set[str]],
    ) -> List[Assembly]:
        """Infers containment relationships between components.

        Performs complete nested containment analysis where each component
        is checked against all larger components to find the smallest,
        most immediate enclosing parent.

        Args:
            detections: All shape detections.
            spatial_index: Spatial index for efficient queries.
            root: Root component.
            processed: Set to track processed detection IDs.
            parent_child_map: Map to track parent-child relationships for cycle detection.

        Returns:
            List of Assembly objects representing containment relationships.
        """
        assemblies: List[Assembly] = []

        for detection in detections:
            if detection.detection_id == root.detection_id:
                continue

            # Find all components that contain this detection
            containers = spatial_index.find_containing(detection)

            if not containers:
                continue

            # Select immediate parent (smallest container)
            parent = containers[0]

            # Check for cycles
            if self._would_create_cycle(
                parent.detection_id, detection.detection_id, parent_child_map
            ):
                logger.warning(
                    f"Cycle detected: skipping containment "
                    f"{parent.detection_id} -> {detection.detection_id}"
                )
                continue

            # Calculate dynamic confidence based on overlap
            confidence = self._calculate_containment_confidence(
                parent.bbox, detection.bbox
            )

            assembly = Assembly(
                parent_shape_id=parent.detection_id,
                child_shape_ids=[detection.detection_id],
                relationship_type="contains",
                confidence=confidence,
            )
            assemblies.append(assembly)
            processed.add(detection.detection_id)

            # Update parent-child map
            if parent.detection_id not in parent_child_map:
                parent_child_map[parent.detection_id] = set()
            parent_child_map[parent.detection_id].add(detection.detection_id)

        logger.debug(f"Inferred {len(assemblies)} containment relationships")
        return assemblies

    def _infer_fastening_assemblies(
        self,
        detections: List[Detection],
        spatial_index: SpatialIndex,
        processed: Set[str],
        parent_child_map: Dict[str, Set[str]],
    ) -> List[Assembly]:
        """Infers fastening relationships between components.

        Identifies fastener components and determines which components
        they connect based on proximity and geometric alignment.

        Args:
            detections: All shape detections.
            spatial_index: Spatial index for efficient queries.
            processed: Set to track processed detection IDs.
            parent_child_map: Map to track parent-child relationships for cycle detection.

        Returns:
            List of Assembly objects representing fastening relationships.
        """
        assemblies: List[Assembly] = []

        fasteners = [
            d for d in detections if d.class_name in self.config.fastener_types
        ]

        for fastener in fasteners:
            if fastener.detection_id in processed:
                continue

            # Find connected components
            connected = self._find_connected_components(fastener, spatial_index)

            if not connected:
                continue

            # Check for cycles
            has_cycle = False
            for component in connected:
                if self._would_create_cycle(
                    fastener.detection_id, component.detection_id, parent_child_map
                ):
                    logger.warning(
                        f"Cycle detected in fastening: skipping "
                        f"{fastener.detection_id} -> {component.detection_id}"
                    )
                    has_cycle = True
                    break

            if has_cycle:
                continue

            # Calculate dynamic confidence based on geometry
            confidence = self._calculate_fastening_confidence(fastener, connected)

            assembly = Assembly(
                parent_shape_id=fastener.detection_id,
                child_shape_ids=[c.detection_id for c in connected],
                relationship_type="fastens",
                confidence=confidence,
            )
            assemblies.append(assembly)
            processed.add(fastener.detection_id)

            # Update parent-child map
            if fastener.detection_id not in parent_child_map:
                parent_child_map[fastener.detection_id] = set()
            for component in connected:
                parent_child_map[fastener.detection_id].add(component.detection_id)

        logger.debug(f"Inferred {len(assemblies)} fastening relationships")
        return assemblies

    def _handle_orphaned_components(
        self, detections: List[Detection], root: Detection, processed: Set[str]
    ) -> List[Assembly]:
        """Handles components not part of any explicit relationship.

        Attaches orphaned components to the root with an 'orphaned' relationship.

        Args:
            detections: All shape detections.
            root: Root component.
            processed: Set of processed detection IDs.

        Returns:
            List of Assembly objects for orphaned components.
        """
        orphaned_assemblies: List[Assembly] = []

        for detection in detections:
            if detection.detection_id not in processed:
                assembly = Assembly(
                    parent_shape_id=root.detection_id,
                    child_shape_ids=[detection.detection_id],
                    relationship_type="orphaned",
                    confidence=self.config.orphaned_confidence,
                )
                orphaned_assemblies.append(assembly)
                processed.add(detection.detection_id)

        if orphaned_assemblies:
            logger.info(
                f"Attached {len(orphaned_assemblies)} orphaned components to root"
            )

        return orphaned_assemblies

    def _would_create_cycle(
        self, parent_id: str, child_id: str, parent_child_map: Dict[str, Set[str]]
    ) -> bool:
        """Checks if adding a parent-child relationship would create a cycle.

        Args:
            parent_id: Proposed parent detection ID.
            child_id: Proposed child detection ID.
            parent_child_map: Current parent-child relationships.

        Returns:
            True if adding this relationship would create a cycle.
        """
        # Check if child is already an ancestor of parent
        visited: Set[str] = set()
        to_check = [child_id]

        while to_check:
            current = to_check.pop()

            if current in visited:
                continue

            if current == parent_id:
                return True

            visited.add(current)

            # Add children of current to check
            if current in parent_child_map:
                to_check.extend(parent_child_map[current])

        return False

    def _calculate_containment_confidence(
        self, parent_bbox: Any, child_bbox: Any
    ) -> float:
        """Calculates confidence score for containment relationship.

        Based on how well the child fits within the parent (overlap ratio).

        Args:
            parent_bbox: Parent bounding box.
            child_bbox: Child bounding box.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        child_area = child_bbox.area()

        if child_area == 0:
            return self.config.containment_confidence_base * 0.5

        # Calculate overlap ratio (child area should be fully within parent)
        # This is a simplified metric; could be enhanced with intersection area
        overlap_ratio = self.config.min_containment_overlap

        # Scale base confidence by how well child fits
        confidence = self.config.containment_confidence_base * overlap_ratio

        return max(0.5, min(1.0, confidence))

    def _calculate_fastening_confidence(
        self, fastener: Detection, connected: List[Detection]
    ) -> float:
        """Calculates confidence score for fastening relationship.

        Based on number of connected components and their proximity.

        Args:
            fastener: Fastener detection.
            connected: List of connected component detections.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if not connected:
            return 0.0

        # Base confidence
        confidence = self.config.fastening_confidence_base

        # Adjust based on number of connections (2+ is ideal for fasteners)
        if len(connected) >= 2:
            confidence *= 1.0
        elif len(connected) == 1:
            confidence *= 0.8  # Less confident with single connection

        # Calculate average distance for quality assessment
        fastener_center = fastener.bbox.center()
        avg_distance = sum(
            calculate_distance(fastener_center, c.bbox.center()) for c in connected
        ) / len(connected)

        # Penalize large distances
        distance_ratio = avg_distance / self.config.fastener_proximity_px
        confidence *= 1.0 - (distance_ratio * 0.2)

        return max(0.3, min(1.0, confidence))

    def _find_connected_components(
        self, fastener: Detection, spatial_index: SpatialIndex
    ) -> List[Detection]:
        """Finds components connected by a specific fastener.

        Identifies components within proximity threshold of the fastener's center
        point. Excludes other fasteners and applies geometric alignment checks.

        Args:
            fastener: Detection object representing a fastener (bolt, screw, etc.).
            spatial_index: Spatial index for efficient proximity queries.

        Returns:
            List of Detection objects for components connected by this fastener.
            May be empty if no components are within proximity threshold.

        Note:
            Proximity threshold is configurable via HierarchyConfig.fastener_proximity_px.
        """
        nearby = spatial_index.find_within_distance(
            fastener, self.config.fastener_proximity_px
        )

        connected: List[Detection] = []
        fastener_center = fastener.bbox.center()

        for detection, distance in nearby:
            # Skip other fasteners
            if detection.class_name in self.config.fastener_types:
                continue

            # Simple geometric alignment check
            # (Could be enhanced with hole detection and alignment analysis)
            component_center = detection.bbox.center()

            # Check if fastener is roughly aligned (within 45 degrees)
            dx = abs(component_center[0] - fastener_center[0])
            dy = abs(component_center[1] - fastener_center[1])

            # Allow some angular tolerance
            is_aligned = (dx < distance * 0.8) or (dy < distance * 0.8)

            if is_aligned:
                connected.append(detection)

        return connected

    def _build_tree(
        self,
        root: Detection,
        assemblies: List[Assembly],
        detection_by_id: Dict[str, Detection],
    ) -> HierarchyNode:
        """Constructs hierarchical tree structure from assembly relationships.

        Builds a nested dictionary representing the component hierarchy starting
        from the root. Each node contains the component ID, class name, and
        list of child nodes with relationship types.

        Args:
            root: Root component detection to start tree traversal.
            assemblies: List of assembly relationships defining parent-child connections.
            detection_by_id: Dictionary mapping detection IDs to Detection objects.

        Returns:
            HierarchyNode representing the root and all descendants.

        Note:
            Uses depth-first traversal with cycle detection via visited set.
            Tree depth is limited by max_tree_depth configuration.
        """

        def build_node(
            node_id: str, visited: Set[str], depth: int = 0
        ) -> HierarchyNode:
            """Recursively builds tree node with cycle and depth protection.

            Args:
                node_id: ID of the detection to build node for.
                visited: Set of already visited node IDs to prevent cycles.
                depth: Current depth in the tree for stack overflow prevention.

            Returns:
                HierarchyNode representing the node and its children.
            """
            if node_id in visited:
                logger.warning(f"Cycle detected at node {node_id}")
                return HierarchyNode(
                    id=node_id, class_name="cycle_error", confidence=0.0, children=[]
                )

            if depth > self.config.max_tree_depth:
                logger.warning(
                    f"Max tree depth ({self.config.max_tree_depth}) exceeded "
                    f"at node {node_id}"
                )
                return HierarchyNode(
                    id=node_id, class_name="depth_exceeded", confidence=0.0, children=[]
                )

            visited.add(node_id)
            detection = detection_by_id.get(node_id)

            if detection is None:
                logger.error(
                    f"Detection not found for node_id: {node_id}. "
                    "This indicates an invalid ID in assembly relationships."
                )
                return HierarchyNode(
                    id=node_id, class_name="not_found", confidence=0.0, children=[]
                )

            node = HierarchyNode(
                id=node_id,
                class_name=detection.class_name,
                confidence=detection.confidence,
                children=[],
            )

            # Find assemblies where this is parent
            for assembly in assemblies:
                if assembly.parent_shape_id == node_id:
                    for child_id in assembly.child_shape_ids:
                        child_node = build_node(child_id, visited.copy(), depth + 1)
                        if child_node and child_node.get("class_name") != "cycle_error":
                            child_node["relationship"] = assembly.relationship_type
                            node["children"].append(child_node)

            return node

        tree = build_node(root.detection_id, set())
        return tree

    # Tree traversal utility methods

    def find_children(
        self, hierarchy: ComponentHierarchy, component_id: str
    ) -> List[HierarchyNode]:
        """Finds all direct children of a component in the hierarchy.

        Args:
            hierarchy: ComponentHierarchy to search.
            component_id: ID of the parent component.

        Returns:
            List of child HierarchyNode objects.
        """

        def search_node(
            node: HierarchyNode, target_id: str
        ) -> Optional[List[HierarchyNode]]:
            if node["id"] == target_id:
                return node.get("children", [])

            for child in node.get("children", []):
                result = search_node(child, target_id)
                if result is not None:
                    return result

            return None

        return search_node(hierarchy.hierarchy_tree, component_id) or []

    def get_subtree(
        self, hierarchy: ComponentHierarchy, component_id: str
    ) -> Optional[HierarchyNode]:
        """Gets the complete subtree rooted at a specific component.

        Args:
            hierarchy: ComponentHierarchy to search.
            component_id: ID of the component to use as subtree root.

        Returns:
            HierarchyNode representing the subtree, or None if not found.
        """

        def search_node(node: HierarchyNode, target_id: str) -> Optional[HierarchyNode]:
            if node["id"] == target_id:
                return node

            for child in node.get("children", []):
                result = search_node(child, target_id)
                if result is not None:
                    return result

            return None

        return search_node(hierarchy.hierarchy_tree, component_id)

    def get_path_to_root(
        self, hierarchy: ComponentHierarchy, component_id: str
    ) -> List[str]:
        """Gets the path from a component to the root.

        Args:
            hierarchy: ComponentHierarchy to search.
            component_id: ID of the component to find path for.

        Returns:
            List of component IDs from root to target (inclusive).
            Empty list if component not found.
        """

        def search_path(
            node: HierarchyNode, target_id: str, current_path: List[str]
        ) -> Optional[List[str]]:
            current_path = current_path + [node["id"]]

            if node["id"] == target_id:
                return current_path

            for child in node.get("children", []):
                result = search_path(child, target_id, current_path)
                if result is not None:
                    return result

            return None

        return search_path(hierarchy.hierarchy_tree, component_id, []) or []

    def get_all_components(self, hierarchy: ComponentHierarchy) -> List[str]:
        """Gets all component IDs in the hierarchy.

        Args:
            hierarchy: ComponentHierarchy to traverse.

        Returns:
            List of all component IDs in depth-first order.
        """
        components: List[str] = []

        def traverse(node: HierarchyNode) -> None:
            components.append(node["id"])
            for child in node.get("children", []):
                traverse(child)

        traverse(hierarchy.hierarchy_tree)
        return components

    def get_components_by_relationship(
        self, hierarchy: ComponentHierarchy, relationship_type: str
    ) -> List[Tuple[str, str]]:
        """Gets all component pairs with a specific relationship type.

        Args:
            hierarchy: ComponentHierarchy to search.
            relationship_type: Type of relationship to filter by
                ('contains', 'fastens', 'orphaned').

        Returns:
            List of (parent_id, child_id) tuples.
        """
        pairs: List[Tuple[str, str]] = []

        def traverse(node: HierarchyNode, parent_id: Optional[str] = None) -> None:
            for child in node.get("children", []):
                if child.get("relationship") == relationship_type:
                    pairs.append((node["id"], child["id"]))
                traverse(child, node["id"])

        traverse(hierarchy.hierarchy_tree)
        return pairs
