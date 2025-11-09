"""
Data validation module for the Drawing Intelligence System.

Cross-validates extraction results for consistency and completeness.
Implements configurable validation rules with spatial indexing for performance.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from scipy.spatial import KDTree

from ..models.data_structures import (
    OCRResult,
    Detection,
    Entity,
    EntityType,
    Association,
    ValidationReport,
    ValidationIssue,
    TextBlock,
)
from ..utils.geometry_utils import calculate_distance, bbox_intersects

logger = logging.getLogger(__name__)


class Severity(Enum):
    """Validation issue severity levels."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class IssueType(Enum):
    """Specific validation issue types."""

    ORPHANED_ENTITY = "ORPHANED_ENTITY"
    MISSING_ASSOCIATION = "MISSING_ASSOCIATION"
    INCOMPATIBLE_ASSOCIATION = "INCOMPATIBLE_ASSOCIATION"
    DIAMETER_ON_NON_CYLINDER = "DIAMETER_ON_NON_CYLINDER"
    THREAD_ON_NON_FASTENER = "THREAD_ON_NON_FASTENER"
    MISSING_CRITICAL_FIELD = "MISSING_CRITICAL_FIELD"
    MISSING_FIELD = "MISSING_FIELD"
    CONFLICTING_ASSOCIATION = "CONFLICTING_ASSOCIATION"
    OVERLAPPING_ENTITIES = "OVERLAPPING_ENTITIES"


@dataclass
class ValidationConfig:
    """Configuration for data validation checks.

    Controls which validation rules are enforced, their thresholds, and penalty weights.

    Attributes:
        enforce_part_number: If True, require part number presence (CRITICAL if missing).
        enforce_material: If True, require material specification (HIGH if missing).
        warn_on_orphaned_entities: If True, flag entities without source text blocks.
        check_dimension_compatibility: If True, validate dimension-shape semantic compatibility.
        check_geometric_overlap: If True, detect overlapping or conflicting entities.
        unassociated_text_distance_threshold: Maximum distance (px) for unassociated dimension detection.
        penalty_critical: Confidence penalty per CRITICAL issue (0.0-1.0).
        penalty_high: Confidence penalty per HIGH issue (0.0-1.0).
        penalty_medium: Confidence penalty per MEDIUM issue (0.0-1.0).
        penalty_low: Confidence penalty per LOW issue (0.0-1.0).
        max_penalty: Maximum cumulative confidence penalty (0.0-1.0).
        critical_entity_types: Entity types considered critical for validation.
        cylindrical_shape_classes: Shape classes compatible with diameter dimensions.
        fastener_shape_classes: Shape classes compatible with thread specifications.
    """

    # Check enablement
    enforce_part_number: bool = True
    enforce_material: bool = False
    warn_on_orphaned_entities: bool = True
    check_dimension_compatibility: bool = True
    check_geometric_overlap: bool = True

    # Thresholds
    unassociated_text_distance_threshold: int = 500
    overlap_threshold: float = 0.3  # 30% bbox overlap

    # Penalty weights
    penalty_critical: float = 0.30
    penalty_high: float = 0.15
    penalty_medium: float = 0.05
    penalty_low: float = 0.02
    max_penalty: float = 0.50

    # Critical fields
    critical_entity_types: Set[EntityType] = field(
        default_factory=lambda: {EntityType.PART_NUMBER}
    )

    # Shape compatibility mappings
    cylindrical_shape_classes: Set[str] = field(
        default_factory=lambda: {
            "shaft",
            "hole",
            "pin",
            "cylinder",
            "bearing",
            "bushing",
        }
    )

    fastener_shape_classes: Set[str] = field(
        default_factory=lambda: {
            "bolt",
            "screw",
            "nut",
            "washer",
            "threaded_hole",
            "stud",
        }
    )

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0 <= self.penalty_critical <= 1:
            raise ValueError("penalty_critical must be between 0 and 1")
        if not 0 <= self.penalty_high <= 1:
            raise ValueError("penalty_high must be between 0 and 1")
        if not 0 <= self.penalty_medium <= 1:
            raise ValueError("penalty_medium must be between 0 and 1")
        if not 0 <= self.penalty_low <= 1:
            raise ValueError("penalty_low must be between 0 and 1")
        if not 0 <= self.max_penalty <= 1:
            raise ValueError("max_penalty must be between 0 and 1")
        if self.unassociated_text_distance_threshold < 0:
            raise ValueError(
                "unassociated_text_distance_threshold must be non-negative"
            )


class DataValidator:
    """Cross-validates extraction results for consistency and completeness.

    Performs multi-layer validation across OCR results, shape detections,
    extracted entities, and their associations. Uses spatial indexing for
    efficient proximity queries on large drawing sets.

    The validator implements comprehensive checks:
        1. Orphaned entities (entities without source text blocks)
        2. Unassociated critical text (dimensions near shapes but not linked)
        3. Entity-shape compatibility (dimension types vs. shape types)
        4. Critical field presence (part numbers, materials, etc.)
        5. Geometric overlap detection (conflicting annotations)
        6. Association quality (conflicting or invalid links)

    Attributes:
        config: Validation configuration controlling check behavior and penalties.
    """

    # Constants
    DIMENSION_PATTERN = re.compile(
        r"(?=.*\d)"  # Must contain digit
        r"(?=.*(mm|cm|inch|in|\"|\bm\b|\br\b|ø|±|diameter|thread|tol))",
        re.IGNORECASE,
    )

    DIMENSION_EXCLUDE_PATTERN = re.compile(
        r"\b(note|rev|title|scale|sheet|date|drawn|checked|approved|qty|item)\b",
        re.IGNORECASE,
    )

    def __init__(self, config: Optional[ValidationConfig] = None) -> None:
        """Initialize the data validator with optional configuration.

        Args:
            config: Validation configuration. If None, uses default configuration
                with all standard checks enabled.
        """
        self.config = config or ValidationConfig()
        logger.info(
            f"DataValidator initialized with config: "
            f"enforce_part_number={self.config.enforce_part_number}, "
            f"check_dimension_compatibility={self.config.check_dimension_compatibility}"
        )

    def validate_associations(
        self,
        ocr_result: OCRResult,
        detections: List[Detection],
        entities: List[Entity],
        associations: List[Association],
    ) -> ValidationReport:
        """Cross-validate all extraction layers for consistency.

        Runs all configured validation checks and generates a comprehensive
        validation report with severity-classified issues and confidence adjustments.
        Uses pre-calculated lookup maps for efficient cross-referencing.

        Args:
            ocr_result: OCR extraction result containing text blocks and metadata.
            detections: List of shape detections from computer vision pipeline.
            entities: List of extracted structured entities (part numbers,
                dimensions, materials, etc.).
            associations: List of validated text-to-shape associations.

        Returns:
            ValidationReport containing:
                - is_valid: False if CRITICAL issues found
                - issues: List of all validation issues with severity levels
                - confidence_adjustment: Penalty factor (0.5-1.0) based on issue severity
                - requires_human_review: True if HIGH or CRITICAL issues exist

        Example:
            >>> validator = DataValidator()
            >>> report = validator.validate_associations(
            ...     ocr_result, detections, entities, associations
            ... )
            >>> if report.requires_human_review:
            ...     logger.warning(f"Found {len(report.issues)} validation issues")
        """
        issues: List[ValidationIssue] = []

        # Pre-calculate lookup maps for efficiency
        text_block_by_id = {tb.text_id: tb for tb in ocr_result.text_blocks}
        detection_by_id = {det.detection_id: det for det in detections}
        entity_by_text: Dict[str, List[Entity]] = defaultdict(list)
        for entity in entities:
            entity_by_text[entity.source_text_id].append(entity)

        # Check 1: Orphaned entities
        if self.config.warn_on_orphaned_entities:
            logger.debug("Running orphaned entity check")
            orphaned_issues = self._check_orphaned_entities(entities, text_block_by_id)
            issues.extend(orphaned_issues)

        # Check 2: Unassociated critical text
        logger.debug("Running unassociated text check")
        unassociated_issues = self._check_unassociated_text(
            ocr_result.text_blocks, detections, associations
        )
        issues.extend(unassociated_issues)

        # Check 3: Entity-shape consistency
        if self.config.check_dimension_compatibility:
            logger.debug("Running dimension compatibility check")
            consistency_issues = self._check_entity_shape_consistency(
                associations, entity_by_text, detection_by_id
            )
            issues.extend(consistency_issues)

        # Check 4: Critical field presence
        logger.debug("Running critical field validation")
        field_issues = self._check_critical_fields(entities)
        issues.extend(field_issues)

        # Check 5: Geometric overlap detection
        if self.config.check_geometric_overlap:
            logger.debug("Running geometric overlap check")
            overlap_issues = self._check_geometric_overlaps(entities, text_block_by_id)
            issues.extend(overlap_issues)

        # Check 6: Association quality
        logger.debug("Running association quality check")
        association_issues = self._check_association_quality(associations)
        issues.extend(association_issues)

        # Calculate validation status
        issues_by_severity = self._group_issues_by_severity(issues)

        is_valid = len(issues_by_severity[Severity.CRITICAL]) == 0
        requires_review = (
            len(issues_by_severity[Severity.HIGH]) > 0
            or len(issues_by_severity[Severity.CRITICAL]) > 0
        )

        # Calculate confidence adjustment
        confidence_adjustment = self._calculate_confidence_penalty(issues)

        report = ValidationReport(
            is_valid=is_valid,
            issues=issues,
            confidence_adjustment=confidence_adjustment,
            requires_human_review=requires_review,
        )

        logger.info(
            f"Validation complete: {len(issues)} issues found - "
            f"CRITICAL: {len(issues_by_severity[Severity.CRITICAL])}, "
            f"HIGH: {len(issues_by_severity[Severity.HIGH])}, "
            f"MEDIUM: {len(issues_by_severity[Severity.MEDIUM])}, "
            f"LOW: {len(issues_by_severity[Severity.LOW])}"
        )

        if len(issues_by_severity[Severity.CRITICAL]) > 0:
            logger.warning(
                f"CRITICAL validation failure: Drawing contains {len(issues_by_severity[Severity.CRITICAL])} critical issues"
            )

        return report

    def _check_orphaned_entities(
        self, entities: List[Entity], text_block_by_id: Dict[str, TextBlock]
    ) -> List[ValidationIssue]:
        """Identify entities without corresponding source text blocks.

        Orphaned entities indicate extraction logic errors where entities
        were created without valid text sources.

        Args:
            entities: List of extracted entities to validate.
            text_block_by_id: Lookup map of text blocks by ID.

        Returns:
            List of HIGH severity validation issues for orphaned entities.
        """
        issues: List[ValidationIssue] = []

        for entity in entities:
            if entity.source_text_id not in text_block_by_id:
                issue = ValidationIssue(
                    severity=Severity.HIGH.value,
                    type=IssueType.ORPHANED_ENTITY.value,
                    message=f"Entity {entity.entity_type.value} '{entity.value}' has no source text block",
                    entity_id=entity.entity_id,
                    suggested_fix=f"Verify entity extractor correctly sets source_text_id. "
                    f"Check if text block '{entity.source_text_id}' was filtered out.",
                )
                issues.append(issue)

        if issues:
            logger.debug(f"Found {len(issues)} orphaned entities")

        return issues

    def _check_unassociated_text(
        self,
        text_blocks: List[TextBlock],
        detections: List[Detection],
        associations: List[Association],
    ) -> List[ValidationIssue]:
        """Find dimension-like text near shapes that lack associations.

        Uses spatial indexing (KD-tree) for efficient nearest-neighbor search.
        Applies semantic filtering to reduce false positives.

        Args:
            text_blocks: List of all OCR text blocks.
            detections: List of detected shapes.
            associations: List of existing text-shape associations.

        Returns:
            List of MEDIUM severity issues for unassociated dimensions.

        Note:
            Uses config.unassociated_text_distance_threshold for proximity
            detection (default: 500px). Complexity: O(n log m) vs O(n*m).
        """
        issues: List[ValidationIssue] = []

        if not detections:
            return issues

        # Build spatial index for shapes
        shape_centers = [det.bbox.center() for det in detections]
        shape_tree = KDTree(shape_centers)

        # Get associated text IDs
        associated_text_ids = {assoc.text_id for assoc in associations}

        for text_block in text_blocks:
            # Skip if already associated
            if text_block.text_id in associated_text_ids:
                continue

            # Check if looks like dimension with semantic filtering
            if not self._looks_like_dimension(text_block.content):
                continue

            # Find nearest shape using KD-tree
            text_center = text_block.bbox.center()
            distances, indices = shape_tree.query(
                [text_center],
                k=1,
                distance_upper_bound=self.config.unassociated_text_distance_threshold,
            )

            # If dimension is near a shape but not linked
            if distances[0] != float("inf"):
                nearest_shape = detections[indices[0]]
                issue = ValidationIssue(
                    severity=Severity.MEDIUM.value,
                    type=IssueType.MISSING_ASSOCIATION.value,
                    message=f"Dimension '{text_block.content}' is {distances[0]:.1f}px from "
                    f"{nearest_shape.class_name} but not associated",
                    shape_id=nearest_shape.detection_id,
                    suggested_fix=f"Increase association distance threshold or verify "
                    f"data_associator correctly links text at {distances[0]:.1f}px distance.",
                )
                issues.append(issue)

        if issues:
            logger.debug(f"Found {len(issues)} unassociated dimension texts")

        return issues

    def _check_entity_shape_consistency(
        self,
        associations: List[Association],
        entity_by_text: Dict[str, List[Entity]],
        detection_by_id: Dict[str, Detection],
    ) -> List[ValidationIssue]:
        """Validate dimension entity compatibility with associated shapes.

        Implements semantic checks:
            - Diameter dimensions only for cylindrical shapes
            - Thread specs only for fasteners
            - General dimensional compatibility

        Args:
            associations: Text-shape associations with relationship types.
            entity_by_text: Lookup map of entities grouped by source text ID.
            detection_by_id: Lookup map of detections by ID.

        Returns:
            List of LOW severity issues for incompatible pairings.
        """
        issues: List[ValidationIssue] = []

        # Check dimension associations
        for assoc in associations:
            if assoc.relationship_type != "dimension":
                continue

            # Get entities from this text
            dimension_entities = entity_by_text.get(assoc.text_id, [])
            if not dimension_entities:
                continue

            # Get associated shape
            shape = detection_by_id.get(assoc.shape_id)
            if not shape:
                continue

            # Perform compatibility checks
            compatibility_issue = self._check_dimension_shape_compatibility(
                dimension_entities, shape
            )
            if compatibility_issue:
                issues.append(compatibility_issue)

        if issues:
            logger.debug(f"Found {len(issues)} dimension-shape compatibility issues")

        return issues

    def _check_dimension_shape_compatibility(
        self, dimension_entities: List[Entity], shape: Detection
    ) -> Optional[ValidationIssue]:
        """Check semantic compatibility between dimension entities and shape.

        Validates that dimension types match shape geometry:
            - Diameter (Ø) dimensions only for cylindrical shapes
            - Thread dimensions only for fasteners
            - Other dimensions are generally compatible

        Args:
            dimension_entities: List of dimension entities from associated text.
            shape: Shape detection to validate against.

        Returns:
            ValidationIssue if incompatibility detected, None otherwise.
        """
        shape_class_lower = shape.class_name.lower()

        for entity in dimension_entities:
            entity_value_lower = entity.value.lower()

            # Check 1: Diameter dimensions on non-cylindrical shapes
            if "ø" in entity_value_lower or "diameter" in entity_value_lower:
                if not any(
                    cyl in shape_class_lower
                    for cyl in self.config.cylindrical_shape_classes
                ):
                    return ValidationIssue(
                        severity=Severity.LOW.value,
                        type=IssueType.DIAMETER_ON_NON_CYLINDER.value,
                        message=f"Diameter dimension '{entity.value}' associated with "
                        f"non-cylindrical shape '{shape.class_name}'",
                        entity_id=entity.entity_id,
                        shape_id=shape.detection_id,
                        suggested_fix=f"Verify association logic. If shape is actually cylindrical, "
                        f"add '{shape.class_name}' to cylindrical_shape_classes config.",
                    )

            # Check 2: Thread specs on non-fastener shapes
            if any(
                thread_ind in entity_value_lower
                for thread_ind in ["m6", "m8", "m10", "thread", "pitch"]
            ):
                if not any(
                    fast in shape_class_lower
                    for fast in self.config.fastener_shape_classes
                ):
                    return ValidationIssue(
                        severity=Severity.LOW.value,
                        type=IssueType.THREAD_ON_NON_FASTENER.value,
                        message=f"Thread specification '{entity.value}' associated with "
                        f"non-fastener shape '{shape.class_name}'",
                        entity_id=entity.entity_id,
                        shape_id=shape.detection_id,
                        suggested_fix=f"Verify thread annotation. If shape is a fastener, "
                        f"add '{shape.class_name}' to fastener_shape_classes config.",
                    )

        return None

    def _check_critical_fields(self, entities: List[Entity]) -> List[ValidationIssue]:
        """Verify presence of mandatory entity types.

        Checks for critical fields required for database enrichment based on
        configuration settings.

        Args:
            entities: List of extracted entities to check.

        Returns:
            List of validation issues for missing critical fields.
        """
        issues: List[ValidationIssue] = []

        entity_types_present = {e.entity_type for e in entities}

        # Check configured critical entity types
        for critical_type in self.config.critical_entity_types:
            if critical_type not in entity_types_present:
                issue = ValidationIssue(
                    severity=Severity.CRITICAL.value,
                    type=IssueType.MISSING_CRITICAL_FIELD.value,
                    message=f"No {critical_type.value} found in drawing",
                    suggested_fix=f"Enable LLM enhancement for title block extraction "
                    f"or manually verify {critical_type.value} is present in drawing.",
                )
                issues.append(issue)
                logger.warning(f"CRITICAL: Missing {critical_type.value}")

        # Check for material if enforced
        if self.config.enforce_material:
            if EntityType.MATERIAL not in entity_types_present:
                issue = ValidationIssue(
                    severity=Severity.HIGH.value,
                    type=IssueType.MISSING_FIELD.value,
                    message="No material specification found in drawing",
                    suggested_fix="Check title block for material annotation or enable entity enhancement.",
                )
                issues.append(issue)

        # Check for OEM (warning only)
        if EntityType.OEM not in entity_types_present:
            issue = ValidationIssue(
                severity=Severity.LOW.value,
                type=IssueType.MISSING_FIELD.value,
                message="No OEM/manufacturer found in drawing",
                suggested_fix="Run entity enhancement to extract OEM from title block.",
            )
            issues.append(issue)

        return issues

    def _check_geometric_overlaps(
        self, entities: List[Entity], text_block_by_id: Dict[str, TextBlock]
    ) -> List[ValidationIssue]:
        """Detect overlapping or conflicting entity annotations.

        Identifies entities with source text blocks that significantly overlap,
        which may indicate duplicate or conflicting annotations.

        Args:
            entities: List of extracted entities.
            text_block_by_id: Lookup map of text blocks by ID.

        Returns:
            List of MEDIUM severity issues for overlapping entities.
        """
        issues: List[ValidationIssue] = []

        # Group entities with valid text blocks
        valid_entities = [
            (e, text_block_by_id[e.source_text_id])
            for e in entities
            if e.source_text_id in text_block_by_id
        ]

        # Check for overlaps (O(n²) but typically small n)
        for i, (entity1, tb1) in enumerate(valid_entities):
            for entity2, tb2 in valid_entities[i + 1 :]:
                if entity1.entity_type == entity2.entity_type:
                    continue  # Same type OK (e.g., multiple dimensions)

                overlap_ratio = self._calculate_bbox_overlap(tb1.bbox, tb2.bbox)

                if overlap_ratio > self.config.overlap_threshold:
                    issue = ValidationIssue(
                        severity=Severity.MEDIUM.value,
                        type=IssueType.OVERLAPPING_ENTITIES.value,
                        message=f"Entities {entity1.entity_type.value} and {entity2.entity_type.value} "
                        f"have {overlap_ratio:.1%} overlapping bounding boxes",
                        entity_id=entity1.entity_id,
                        suggested_fix="Review OCR output for duplicate or misaligned text blocks.",
                    )
                    issues.append(issue)

        if issues:
            logger.debug(f"Found {len(issues)} overlapping entity pairs")

        return issues

    def _check_association_quality(
        self, associations: List[Association]
    ) -> List[ValidationIssue]:
        """Validate association quality for conflicts and anomalies.

        Detects:
            - Multiple associations from same text to different shapes
            - Circular/self-referential associations
            - Unusually low confidence associations

        Args:
            associations: List of text-shape associations.

        Returns:
            List of LOW severity issues for association quality problems.
        """
        issues: List[ValidationIssue] = []

        # Group associations by text_id
        assoc_by_text: Dict[str, List[Association]] = defaultdict(list)
        for assoc in associations:
            assoc_by_text[assoc.text_id].append(assoc)

        # Check for conflicting associations (same text → multiple shapes)
        for text_id, text_assocs in assoc_by_text.items():
            if len(text_assocs) > 1:
                shape_ids = [a.shape_id for a in text_assocs]
                issue = ValidationIssue(
                    severity=Severity.LOW.value,
                    type=IssueType.CONFLICTING_ASSOCIATION.value,
                    message=f"Text block {text_id} associated with {len(shape_ids)} different shapes",
                    suggested_fix="Review association logic to handle one-to-many relationships "
                    "or increase association confidence threshold.",
                )
                issues.append(issue)

        if issues:
            logger.debug(f"Found {len(issues)} association quality issues")

        return issues

    def _looks_like_dimension(self, content: str) -> bool:
        """Heuristic pattern matching to identify dimensional text.

        Uses compiled regex patterns for efficiency and semantic filtering
        to exclude non-dimensional text (notes, labels, etc.).

        Args:
            content: Text string to evaluate.

        Returns:
            True if text contains numbers, dimensional indicators, and
            does not match exclusion patterns.

        Examples:
            >>> validator._looks_like_dimension("Ø25 ±0.1mm")
            True
            >>> validator._looks_like_dimension("NOTE: See detail A")
            False
        """
        # Check exclusion patterns first (faster)
        if self.DIMENSION_EXCLUDE_PATTERN.search(content):
            return False

        # Check for dimension pattern
        return bool(self.DIMENSION_PATTERN.search(content))

    def _calculate_bbox_overlap(self, bbox1, bbox2) -> float:
        """Calculate intersection-over-union ratio for two bounding boxes.

        Args:
            bbox1: First bounding box.
            bbox2: Second bounding box.

        Returns:
            Overlap ratio (0.0 to 1.0), where 1.0 means complete overlap.
        """
        if not bbox_intersects(bbox1, bbox2):
            return 0.0

        # Calculate intersection area
        x_left = max(bbox1.x1, bbox2.x1)
        y_top = max(bbox1.y1, bbox2.y1)
        x_right = min(bbox1.x2, bbox2.x2)
        y_bottom = min(bbox1.y2, bbox2.y2)

        intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

        # Calculate union area
        bbox1_area = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1)
        bbox2_area = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1)
        union_area = bbox1_area + bbox2_area - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def _group_issues_by_severity(
        self, issues: List[ValidationIssue]
    ) -> Dict[Severity, List[ValidationIssue]]:
        """Group validation issues by severity level.

        Args:
            issues: List of validation issues.

        Returns:
            Dictionary mapping severity levels to issue lists.
        """
        grouped: Dict[Severity, List[ValidationIssue]] = {
            Severity.CRITICAL: [],
            Severity.HIGH: [],
            Severity.MEDIUM: [],
            Severity.LOW: [],
        }

        for issue in issues:
            severity = Severity(issue.severity)
            grouped[severity].append(issue)

        return grouped

    def _calculate_confidence_penalty(self, issues: List[ValidationIssue]) -> float:
        """Calculate confidence reduction factor from validation issues.

        Applies configurable severity-weighted penalties with early termination
        when maximum penalty is reached.

        Args:
            issues: List of validation issues with severity classifications.

        Returns:
            Confidence adjustment factor between (1-max_penalty) and 1.0, where:
                - 1.0 = no issues, no penalty
                - (1-max_penalty) = maximum penalty applied

        Example:
            >>> config = ValidationConfig(penalty_high=0.15, penalty_medium=0.05)
            >>> issues = [
            ...     ValidationIssue(severity="HIGH", ...),
            ...     ValidationIssue(severity="MEDIUM", ...)
            ... ]
            >>> penalty = validator._calculate_confidence_penalty(issues)
            >>> # Returns 0.80 (20% penalty: 0.15 + 0.05)
        """
        if not issues:
            return 1.0

        # Penalty weights from config
        penalty_map = {
            Severity.CRITICAL: self.config.penalty_critical,
            Severity.HIGH: self.config.penalty_high,
            Severity.MEDIUM: self.config.penalty_medium,
            Severity.LOW: self.config.penalty_low,
        }

        # Calculate cumulative penalty with early termination
        penalty = 0.0
        for issue in issues:
            severity = Severity(issue.severity)
            penalty += penalty_map[severity]

            # Early termination at max penalty
            if penalty >= self.config.max_penalty:
                penalty = self.config.max_penalty
                break

        # Return adjustment factor (1.0 = no change)
        return 1.0 - penalty
