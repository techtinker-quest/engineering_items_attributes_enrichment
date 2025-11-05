"""
Quality scoring module for the Drawing Intelligence System.

Calculates confidence scores and generates review flags.
"""

import logging
from dataclasses import dataclass
from typing import List

from ..models.data_structures import (
    ProcessingResult,
    ReviewFlag,
    CompletenessScore,
    FlagType,
    Severity,
    EntityType,
)
from ..utils.file_utils import generate_unique_id

logger = logging.getLogger(__name__)


@dataclass
class QualityConfig:
    """
    Configuration for quality scoring.

    Attributes:
        review_threshold: Threshold for human review (default: 0.75)
        ocr_weight: Weight for OCR confidence (default: 0.30)
        detection_weight: Weight for detection confidence (default: 0.40)
        entity_weight: Weight for entity confidence (default: 0.30)
        flag_missing_critical_entities: Flag missing critical entities (default: True)
        critical_entities: List of critical entity types (default: ['PART_NUMBER'])
    """

    review_threshold: float = 0.75
    ocr_weight: float = 0.30
    detection_weight: float = 0.40
    entity_weight: float = 0.30
    flag_missing_critical_entities: bool = True
    critical_entities: List[str] = None

    def __post_init__(self):
        if self.critical_entities is None:
            self.critical_entities = ["PART_NUMBER"]


class QualityScorer:
    """
    Calculate confidence scores and generate review flags.

    Evaluates:
    - Overall confidence (weighted)
    - Completeness
    - Critical field presence
    - Data quality issues
    """

    def __init__(self, config: QualityConfig):
        """
        Initialize quality scorer.

        Args:
            config: Quality configuration
        """
        self.config = config
        logger.info("QualityScorer initialized")

    def calculate_drawing_confidence(
        self, processing_result: ProcessingResult
    ) -> float:
        """
        Calculate overall confidence score.

        Uses weighted formula:
        overall = (ocr_weight * ocr_conf) + (detection_weight * det_conf) +
                  (entity_weight * entity_conf)

        Args:
            processing_result: Complete processing result

        Returns:
            Overall confidence score (0.0-1.0)
        """
        # OCR confidence
        if processing_result.ocr_result and processing_result.ocr_result.text_blocks:
            ocr_conf = processing_result.ocr_result.average_confidence
        else:
            ocr_conf = 0.0

        # Detection confidence
        if processing_result.detections:
            det_conf = sum(d.confidence for d in processing_result.detections) / len(
                processing_result.detections
            )
        else:
            det_conf = 0.0

        # Entity confidence
        if processing_result.entities:
            entity_conf = sum(e.confidence for e in processing_result.entities) / len(
                processing_result.entities
            )
        else:
            entity_conf = 0.0

        # Weighted score
        overall = (
            self.config.ocr_weight * ocr_conf
            + self.config.detection_weight * det_conf
            + self.config.entity_weight * entity_conf
        )

        # Apply validation penalty if present
        if processing_result.validation_report:
            overall *= processing_result.validation_report.confidence_adjustment

        logger.debug(
            f"Confidence: OCR={ocr_conf:.2f}, Detection={det_conf:.2f}, "
            f"Entity={entity_conf:.2f}, Overall={overall:.2f}"
        )

        return overall

    def generate_review_flags(
        self, processing_result: ProcessingResult
    ) -> List[ReviewFlag]:
        """
        Generate flags for human review.

        Args:
            processing_result: Complete processing result

        Returns:
            List of ReviewFlags
        """
        flags = []

        # Flag 1: Low overall confidence
        overall_conf = processing_result.overall_confidence
        if overall_conf < self.config.review_threshold:
            flag = ReviewFlag(
                flag_id=generate_unique_id("FLAG"),
                flag_type=FlagType.LOW_CONFIDENCE,
                severity=self._classify_severity(overall_conf),
                reason=f"Overall confidence {overall_conf:.2f} below threshold "
                f"{self.config.review_threshold}",
                details={"confidence": overall_conf},
                suggested_action="Review extraction results and consider LLM enhancement",
            )
            flags.append(flag)

        # Flag 2: Missing entities
        if not processing_result.entities:
            flag = ReviewFlag(
                flag_id=generate_unique_id("FLAG"),
                flag_type=FlagType.MISSING_DATA,
                severity=Severity.HIGH,
                reason="No entities extracted from drawing",
                details={},
                suggested_action="Verify OCR quality and run entity extraction again",
            )
            flags.append(flag)

        # Flag 3: Missing detections
        if not processing_result.detections:
            flag = ReviewFlag(
                flag_id=generate_unique_id("FLAG"),
                flag_type=FlagType.MISSING_DATA,
                severity=Severity.HIGH,
                reason="No shapes detected in drawing",
                details={},
                suggested_action="Check image quality and detection model",
            )
            flags.append(flag)

        # Flag 4: Missing critical entities
        if self.config.flag_missing_critical_entities:
            missing_critical = self._check_critical_entities(processing_result)
            if missing_critical:
                flag = ReviewFlag(
                    flag_id=generate_unique_id("FLAG"),
                    flag_type=FlagType.CRITICAL_FIELD_MISSING,
                    severity=Severity.CRITICAL,
                    reason=f"Missing critical entities: {', '.join(missing_critical)}",
                    details={"missing_entities": missing_critical},
                    suggested_action="Run LLM entity enhancement or manual review",
                )
                flags.append(flag)

        # Flag 5: Validation issues
        if processing_result.validation_report:
            for issue in processing_result.validation_report.issues:
                if issue.severity in ["HIGH", "CRITICAL"]:
                    flag = ReviewFlag(
                        flag_id=generate_unique_id("FLAG"),
                        flag_type=FlagType.INCONSISTENCY,
                        severity=Severity[issue.severity],
                        reason=issue.message,
                        details={"issue_type": issue.type},
                        suggested_action=issue.suggested_fix
                        or "Manual review required",
                        affected_entities=[issue.entity_id] if issue.entity_id else [],
                        affected_shapes=[issue.shape_id] if issue.shape_id else [],
                    )
                    flags.append(flag)

        # Flag 6: Processing errors
        if processing_result.status == "failed":
            flag = ReviewFlag(
                flag_id=generate_unique_id("FLAG"),
                flag_type=FlagType.ERROR,
                severity=Severity.CRITICAL,
                reason="Processing failed or encountered errors",
                details={},
                suggested_action="Review error logs and retry processing",
            )
            flags.append(flag)

        logger.info(f"Generated {len(flags)} review flags")
        return flags

    def assess_completeness(
        self, processing_result: ProcessingResult
    ) -> CompletenessScore:
        """
        Assess data completeness.

        Args:
            processing_result: Complete processing result

        Returns:
            CompletenessScore
        """
        # Check critical fields
        has_part_number = any(
            e.entity_type == EntityType.PART_NUMBER for e in processing_result.entities
        )

        has_dimensions = any(
            e.entity_type == EntityType.DIMENSION for e in processing_result.entities
        )

        has_shapes = len(processing_result.detections) > 0

        has_title_block = processing_result.title_block is not None

        # Identify missing critical fields
        missing_critical = []
        if not has_part_number:
            missing_critical.append("PART_NUMBER")
        if not has_dimensions:
            missing_critical.append("DIMENSION")
        if not has_shapes:
            missing_critical.append("SHAPES")

        # Calculate completeness by category
        completeness_by_category = {
            "entities": self._calculate_entity_completeness(processing_result),
            "shapes": 1.0 if has_shapes else 0.0,
            "associations": self._calculate_association_completeness(processing_result),
            "title_block": 1.0 if has_title_block else 0.5,
        }

        # Overall score (weighted average)
        overall_score = (
            0.40 * completeness_by_category["entities"]
            + 0.30 * completeness_by_category["shapes"]
            + 0.20 * completeness_by_category["associations"]
            + 0.10 * completeness_by_category["title_block"]
        )

        completeness = CompletenessScore(
            overall_score=overall_score,
            has_part_number=has_part_number,
            has_dimensions=has_dimensions,
            has_shapes=has_shapes,
            has_title_block=has_title_block,
            missing_critical_fields=missing_critical,
            completeness_by_category=completeness_by_category,
        )

        logger.debug(
            f"Completeness: {overall_score:.2f} "
            f"(entities={completeness_by_category['entities']:.2f}, "
            f"shapes={completeness_by_category['shapes']:.2f})"
        )

        return completeness

    def _check_critical_entities(
        self, processing_result: ProcessingResult
    ) -> List[str]:
        """
        Check for missing critical entities.

        Args:
            processing_result: Processing result

        Returns:
            List of missing critical entity types
        """
        found_types = {e.entity_type.value for e in processing_result.entities}

        missing = [
            entity_type
            for entity_type in self.config.critical_entities
            if entity_type not in found_types
        ]

        return missing

    def _classify_severity(self, confidence: float) -> Severity:
        """
        Classify severity based on confidence level.

        Args:
            confidence: Confidence score

        Returns:
            Severity level
        """
        if confidence < 0.5:
            return Severity.CRITICAL
        elif confidence < 0.65:
            return Severity.HIGH
        elif confidence < 0.75:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _calculate_entity_completeness(
        self, processing_result: ProcessingResult
    ) -> float:
        """
        Calculate entity completeness score.

        Args:
            processing_result: Processing result

        Returns:
            Completeness score (0.0-1.0)
        """
        if not processing_result.entities:
            return 0.0

        # Expected entity types (not all required)
        expected_types = {
            EntityType.PART_NUMBER: 1.0,  # Critical
            EntityType.OEM: 0.5,
            EntityType.DIMENSION: 0.8,
            EntityType.MATERIAL: 0.3,
            EntityType.WEIGHT: 0.2,
        }

        # Calculate score based on presence and importance
        found_types = {e.entity_type for e in processing_result.entities}

        total_weight = sum(expected_types.values())
        achieved_weight = sum(
            weight
            for entity_type, weight in expected_types.items()
            if entity_type in found_types
        )

        return achieved_weight / total_weight

    def _calculate_association_completeness(
        self, processing_result: ProcessingResult
    ) -> float:
        """
        Calculate association completeness score.

        Args:
            processing_result: Processing result

        Returns:
            Completeness score (0.0-1.0)
        """
        # Simple heuristic: ratio of associated text blocks
        if (
            not processing_result.ocr_result
            or not processing_result.ocr_result.text_blocks
        ):
            return 0.0

        total_text_blocks = len(processing_result.ocr_result.text_blocks)
        associated_blocks = len(processing_result.associations)

        # Not all text needs to be associated (notes, etc.)
        # So we cap the expected association rate at 70%
        expected_rate = 0.7
        actual_rate = min(associated_blocks / total_text_blocks, expected_rate)

        return actual_rate / expected_rate
