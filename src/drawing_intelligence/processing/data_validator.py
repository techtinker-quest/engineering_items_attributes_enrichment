"""
Data validation module for the Drawing Intelligence System.

Cross-validates extraction results for consistency and completeness.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from ..models.data_structures import (
    OCRResult,
    Detection,
    Entity,
    Association,
    ValidationReport,
    ValidationIssue,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """
    Configuration for data validation.

    Attributes:
        enforce_part_number: Require part number presence (default: True)
        warn_on_orphaned_entities: Flag orphaned entities (default: True)
        check_dimension_compatibility: Validate dimension-shape compatibility (default: True)
    """

    enforce_part_number: bool = True
    warn_on_orphaned_entities: bool = True
    check_dimension_compatibility: bool = True


class DataValidator:
    """
    Validate consistency across all extraction layers.

    Performs cross-validation checks:
    - Orphaned entities (no source text)
    - Unassociated text (dimensions near shapes but not linked)
    - Entity-shape compatibility
    - Data consistency
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize data validator.

        Args:
            config: Validation configuration (uses defaults if None)
        """
        self.config = config or ValidationConfig()
        logger.info("DataValidator initialized")

    def validate_associations(
        self,
        ocr_result: OCRResult,
        detections: List[Detection],
        entities: List[Entity],
        associations: List[Association],
    ) -> ValidationReport:
        """
        Cross-validate all extraction layers.

        Args:
            ocr_result: OCR extraction result
            detections: Shape detections
            entities: Extracted entities
            associations: Text-shape associations

        Returns:
            ValidationReport with issues and confidence adjustment
        """
        issues = []

        # Check 1: Orphaned entities
        if self.config.warn_on_orphaned_entities:
            orphaned_issues = self._check_orphaned_entities(
                entities, ocr_result.text_blocks
            )
            issues.extend(orphaned_issues)

        # Check 2: Unassociated critical text
        unassociated_issues = self._check_unassociated_text(
            ocr_result.text_blocks, detections, associations
        )
        issues.extend(unassociated_issues)

        # Check 3: Entity-shape consistency
        if self.config.check_dimension_compatibility:
            consistency_issues = self._check_entity_shape_consistency(
                associations, entities, detections
            )
            issues.extend(consistency_issues)

        # Check 4: Critical field presence
        if self.config.enforce_part_number:
            part_number_issues = self._check_critical_fields(entities)
            issues.extend(part_number_issues)

        # Calculate validation status
        high_severity = [i for i in issues if i.severity == "HIGH"]
        critical_severity = [i for i in issues if i.severity == "CRITICAL"]

        is_valid = len(critical_severity) == 0
        requires_review = len(high_severity) > 0 or len(critical_severity) > 0

        # Calculate confidence adjustment
        confidence_adjustment = self._calculate_confidence_penalty(issues)

        report = ValidationReport(
            is_valid=is_valid,
            issues=issues,
            confidence_adjustment=confidence_adjustment,
            requires_human_review=requires_review,
        )

        logger.info(
            f"Validation complete: {len(issues)} issues found "
            f"(HIGH: {len(high_severity)}, CRITICAL: {len(critical_severity)})"
        )

        return report

    def _check_orphaned_entities(
        self, entities: List[Entity], text_blocks: List
    ) -> List[ValidationIssue]:
        """
        Find entities without source text blocks.

        Args:
            entities: List of entities
            text_blocks: List of text blocks

        Returns:
            List of validation issues
        """
        issues = []

        text_block_ids = {tb.text_id for tb in text_blocks}

        for entity in entities:
            if entity.source_text_id not in text_block_ids:
                issue = ValidationIssue(
                    severity="HIGH",
                    type="ORPHANED_ENTITY",
                    message=f"Entity {entity.entity_type.value} has no source text block",
                    entity_id=entity.entity_id,
                    suggested_fix="Review entity extraction logic",
                )
                issues.append(issue)

        return issues

    def _check_unassociated_text(
        self,
        text_blocks: List,
        detections: List[Detection],
        associations: List[Association],
    ) -> List[ValidationIssue]:
        """
        Find dimensions near shapes that aren't linked.

        Args:
            text_blocks: List of text blocks
            detections: List of detections
            associations: List of associations

        Returns:
            List of validation issues
        """
        issues = []

        if not detections:
            return issues

        # Get associated text IDs
        associated_text_ids = {assoc.text_id for assoc in associations}

        for text_block in text_blocks:
            # Skip if already associated
            if text_block.text_id in associated_text_ids:
                continue

            # Check if looks like dimension
            if not self._looks_like_dimension(text_block.content):
                continue

            # Find nearest shape
            text_center = text_block.bbox.center()
            nearest_shape = None
            min_distance = float("inf")

            from ..utils.geometry_utils import calculate_distance

            for det in detections:
                shape_center = det.bbox.center()
                distance = calculate_distance(text_center, shape_center)

                if distance < min_distance:
                    min_distance = distance
                    nearest_shape = det

            # If dimension is near a shape but not linked
            if nearest_shape and min_distance < 500:
                issue = ValidationIssue(
                    severity="MEDIUM",
                    type="MISSING_ASSOCIATION",
                    message=f"Dimension '{text_block.content}' near "
                    f"{nearest_shape.class_name} but not linked",
                    shape_id=nearest_shape.detection_id,
                    suggested_fix="Review association distance thresholds",
                )
                issues.append(issue)

        return issues

    def _check_entity_shape_consistency(
        self,
        associations: List[Association],
        entities: List[Entity],
        detections: List[Detection],
    ) -> List[ValidationIssue]:
        """
        Check if dimension entities match associated shapes.

        Args:
            associations: Text-shape associations
            entities: Extracted entities
            detections: Shape detections

        Returns:
            List of validation issues
        """
        issues = []

        # Build lookup maps
        entity_by_text: dict = {}
        for entity in entities:
            if entity.source_text_id not in entity_by_text:
                entity_by_text[entity.source_text_id] = []
            entity_by_text[entity.source_text_id].append(entity)

        detection_by_id = {det.detection_id: det for det in detections}

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

            # Check compatibility (simplified)
            if not self._dimension_compatible_with_shape(dimension_entities, shape):
                issue = ValidationIssue(
                    severity="LOW",
                    type="INCOMPATIBLE_ASSOCIATION",
                    message=f"Dimension type may not match shape type {shape.class_name}",
                    entity_id=dimension_entities[0].entity_id,
                    shape_id=shape.detection_id,
                    suggested_fix="Review dimension-shape pairing",
                )
                issues.append(issue)

        return issues

    def _check_critical_fields(self, entities: List[Entity]) -> List[ValidationIssue]:
        """
        Check for presence of critical fields.

        Args:
            entities: List of entities

        Returns:
            List of validation issues
        """
        issues = []

        from ..models.data_structures import EntityType

        # Check for part number
        has_part_number = any(e.entity_type == EntityType.PART_NUMBER for e in entities)

        if not has_part_number:
            issue = ValidationIssue(
                severity="CRITICAL",
                type="MISSING_CRITICAL_FIELD",
                message="No part number found in drawing",
                suggested_fix="Run LLM enhancement or manual review",
            )
            issues.append(issue)

        # Check for OEM (warning only)
        has_oem = any(e.entity_type == EntityType.OEM for e in entities)

        if not has_oem:
            issue = ValidationIssue(
                severity="LOW",
                type="MISSING_FIELD",
                message="No OEM/manufacturer found in drawing",
                suggested_fix="Check title block or run entity enhancement",
            )
            issues.append(issue)

        return issues

    def _looks_like_dimension(self, content: str) -> bool:
        """
        Pattern check to identify dimensional text.

        Args:
            content: Text content

        Returns:
            True if looks like dimension
        """
        content_lower = content.lower()

        # Has number
        has_number = any(c.isdigit() for c in content)
        if not has_number:
            return False

        # Has dimension indicators
        indicators = ["mm", "cm", "inch", "in", '"', "ø", "±", "diameter", "m", "r"]

        return any(ind in content_lower for ind in indicators)

    def _dimension_compatible_with_shape(
        self, dimension_entities: List[Entity], shape: Detection
    ) -> bool:
        """
        Check if dimension type matches shape type.

        Simplified compatibility check.

        Args:
            dimension_entities: List of dimension entities
            shape: Shape detection

        Returns:
            True if compatible
        """
        # For now, all dimensions are considered potentially compatible
        # Full implementation would check:
        # - Diameter dimensions only for cylindrical shapes
        # - Length/width only for rectangular shapes
        # - Thread specs only for threaded components

        return True

    def _calculate_confidence_penalty(self, issues: List[ValidationIssue]) -> float:
        """
        Calculate confidence reduction based on issues.

        Args:
            issues: List of validation issues

        Returns:
            Confidence penalty (0.0 to 1.0, where 1.0 = no penalty)
        """
        if not issues:
            return 1.0

        # Calculate penalty based on severity
        penalty = 0.0

        for issue in issues:
            if issue.severity == "CRITICAL":
                penalty += 0.3
            elif issue.severity == "HIGH":
                penalty += 0.15
            elif issue.severity == "MEDIUM":
                penalty += 0.05
            elif issue.severity == "LOW":
                penalty += 0.02

        # Cap penalty at 0.5 (maximum 50% reduction)
        penalty = min(penalty, 0.5)

        # Return adjustment factor (1.0 = no change, 0.5 = 50% reduction)
        return 1.0 - penalty
