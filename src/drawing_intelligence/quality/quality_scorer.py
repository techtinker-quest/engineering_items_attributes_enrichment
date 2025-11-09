"""
Quality scoring module for the Drawing Intelligence System.

This module provides functionality for calculating confidence scores,
assessing data completeness, and generating human review flags for
processed engineering drawings.

Classes:
    QualityConfig: Configuration parameters for quality scoring.
    QualityScorer: Main quality assessment engine.

Typical usage example:
    config = QualityConfig(
        review_threshold=0.75,
        critical_entities=[EntityType.PART_NUMBER, EntityType.DIMENSION]
    )
    scorer = QualityScorer(config)

    # Calculate confidence
    confidence = scorer.calculate_drawing_confidence(processing_result)

    # Generate review flags if needed
    if confidence < config.review_threshold:
        flags = scorer.generate_review_flags(processing_result)

    # Assess completeness
    completeness = scorer.assess_completeness(processing_result)
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, Final, List, Optional

from ..models.data_structures import (
    ProcessingResult,
    ReviewFlag,
    CompletenessScore,
    FlagType,
    Severity,
    EntityType,
)

logger = logging.getLogger(__name__)

# Processing status constants
STATUS_FAILED: Final[str] = "failed"
STATUS_COMPLETE: Final[str] = "complete"

# Default severity thresholds (confidence score ranges)
DEFAULT_SEVERITY_CRITICAL_THRESHOLD: Final[float] = 0.5
DEFAULT_SEVERITY_HIGH_THRESHOLD: Final[float] = 0.65
DEFAULT_SEVERITY_MEDIUM_THRESHOLD: Final[float] = 0.75

# Default association rate cap
DEFAULT_MAX_ASSOCIATION_RATE: Final[float] = 0.7

# Default title block partial credit
DEFAULT_TITLE_BLOCK_PARTIAL_SCORE: Final[float] = 0.5

# Weight validation tolerance
WEIGHT_SUM_TOLERANCE: Final[float] = 0.01


@dataclass
class QualityConfig:
    """Configuration parameters for quality scoring and review flag generation.

    This dataclass encapsulates all configurable thresholds and weights
    used by the QualityScorer to assess drawing processing quality.

    Attributes:
        review_threshold: Minimum confidence score required to avoid
            human review. Must be between 0.0 and 1.0. Default: 0.75.
        ocr_weight: Weight factor for OCR confidence in overall score.
            Must be between 0.0 and 1.0. Default: 0.30.
        detection_weight: Weight factor for shape detection confidence.
            Must be between 0.0 and 1.0. Default: 0.40.
        entity_weight: Weight factor for entity extraction confidence.
            Must be between 0.0 and 1.0. Default: 0.30.
        flag_missing_critical_entities: Whether to generate critical
            severity flags when essential entity types are missing.
            Default: True.
        critical_entities: List of EntityType enums that are considered
            essential. Defaults to [EntityType.PART_NUMBER].
        completeness_entity_weight: Weight for entity completeness in
            overall completeness score. Default: 0.40.
        completeness_shape_weight: Weight for shape presence in overall
            completeness score. Default: 0.30.
        completeness_association_weight: Weight for associations in
            overall completeness score. Default: 0.20.
        completeness_title_block_weight: Weight for title block in
            overall completeness score. Default: 0.10.
        severity_critical_threshold: Max confidence for CRITICAL severity.
            Default: 0.5.
        severity_high_threshold: Max confidence for HIGH severity.
            Default: 0.65.
        severity_medium_threshold: Max confidence for MEDIUM severity.
            Default: 0.75.
        max_expected_association_rate: Maximum expected ratio of text
            blocks that should be associated with shapes. Default: 0.7.
        title_block_partial_score: Completeness score when title block
            is missing (partial credit for other data). Default: 0.5.
        entity_importance_weights: Dict mapping EntityType to importance
            weight for completeness calculation. Higher weights indicate
            more critical entity types.

    Raises:
        ValueError: If any weight is outside [0.0, 1.0] range, or if
            confidence weights or completeness weights don't sum to 1.0
            (within tolerance).

    Note:
        All weight groups must sum to 1.0 (±0.01 tolerance):
            - ocr_weight + detection_weight + entity_weight = 1.0
            - completeness weights sum = 1.0
    """

    review_threshold: float = 0.75
    ocr_weight: float = 0.30
    detection_weight: float = 0.40
    entity_weight: float = 0.30
    flag_missing_critical_entities: bool = True
    critical_entities: Optional[List[EntityType]] = None
    completeness_entity_weight: float = 0.40
    completeness_shape_weight: float = 0.30
    completeness_association_weight: float = 0.20
    completeness_title_block_weight: float = 0.10
    severity_critical_threshold: float = DEFAULT_SEVERITY_CRITICAL_THRESHOLD
    severity_high_threshold: float = DEFAULT_SEVERITY_HIGH_THRESHOLD
    severity_medium_threshold: float = DEFAULT_SEVERITY_MEDIUM_THRESHOLD
    max_expected_association_rate: float = DEFAULT_MAX_ASSOCIATION_RATE
    title_block_partial_score: float = DEFAULT_TITLE_BLOCK_PARTIAL_SCORE
    entity_importance_weights: Optional[Dict[EntityType, float]] = None

    def __post_init__(self) -> None:
        """Initialize defaults and validate configuration.

        Raises:
            ValueError: If validation fails for any constraint.
        """
        # Set default critical entities
        if self.critical_entities is None:
            self.critical_entities = [EntityType.PART_NUMBER]

        # Set default entity importance weights
        if self.entity_importance_weights is None:
            self.entity_importance_weights = {
                EntityType.PART_NUMBER: 1.0,  # Critical
                EntityType.DIMENSION: 0.8,  # High importance
                EntityType.OEM: 0.5,  # Medium importance
                EntityType.MATERIAL: 0.3,  # Low importance
                EntityType.WEIGHT: 0.2,  # Low importance
            }

        # Validate all weights are in valid range
        self._validate_range("review_threshold", self.review_threshold)
        self._validate_range("ocr_weight", self.ocr_weight)
        self._validate_range("detection_weight", self.detection_weight)
        self._validate_range("entity_weight", self.entity_weight)
        self._validate_range(
            "completeness_entity_weight", self.completeness_entity_weight
        )
        self._validate_range(
            "completeness_shape_weight", self.completeness_shape_weight
        )
        self._validate_range(
            "completeness_association_weight", self.completeness_association_weight
        )
        self._validate_range(
            "completeness_title_block_weight", self.completeness_title_block_weight
        )
        self._validate_range(
            "severity_critical_threshold", self.severity_critical_threshold
        )
        self._validate_range("severity_high_threshold", self.severity_high_threshold)
        self._validate_range(
            "severity_medium_threshold", self.severity_medium_threshold
        )
        self._validate_range(
            "max_expected_association_rate", self.max_expected_association_rate
        )
        self._validate_range(
            "title_block_partial_score", self.title_block_partial_score
        )

        # Validate confidence weights sum to 1.0
        confidence_sum = self.ocr_weight + self.detection_weight + self.entity_weight
        if abs(confidence_sum - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"Confidence weights must sum to 1.0 (±{WEIGHT_SUM_TOLERANCE}), "
                f"got {confidence_sum:.4f}"
            )

        # Validate completeness weights sum to 1.0
        completeness_sum = (
            self.completeness_entity_weight
            + self.completeness_shape_weight
            + self.completeness_association_weight
            + self.completeness_title_block_weight
        )
        if abs(completeness_sum - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"Completeness weights must sum to 1.0 "
                f"(±{WEIGHT_SUM_TOLERANCE}), got {completeness_sum:.4f}"
            )

        # Validate severity thresholds are in ascending order
        if not (
            self.severity_critical_threshold
            < self.severity_high_threshold
            < self.severity_medium_threshold
        ):
            raise ValueError(
                "Severity thresholds must be in ascending order: "
                "critical < high < medium"
            )

        # Validate entity importance weights
        for entity_type, weight in self.entity_importance_weights.items():
            if not isinstance(entity_type, EntityType):
                raise ValueError(
                    f"Entity importance keys must be EntityType enums, "
                    f"got {type(entity_type)}"
                )
            if not 0.0 <= weight <= 1.0:
                raise ValueError(
                    f"Entity importance weight for {entity_type.value} "
                    f"must be in [0.0, 1.0], got {weight}"
                )

    def _validate_range(self, name: str, value: float) -> None:
        """Validate that a value is in the valid [0.0, 1.0] range.

        Args:
            name: Parameter name for error messages.
            value: Value to validate.

        Raises:
            ValueError: If value is outside [0.0, 1.0].
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")


class QualityScorer:
    """Calculate confidence scores and generate review flags for drawings.

    This class provides comprehensive quality assessment capabilities,
    including weighted confidence calculation, completeness analysis,
    and automated review flag generation based on configurable thresholds.

    The scorer evaluates multiple quality dimensions:
        - OCR text extraction confidence
        - Shape detection confidence
        - Entity extraction confidence
        - Data completeness (critical fields present)
        - Validation issues from data validators

    Attributes:
        config: QualityConfig instance containing scoring parameters.

    Example:
        >>> config = QualityConfig(review_threshold=0.70)
        >>> scorer = QualityScorer(config)
        >>> confidence = scorer.calculate_drawing_confidence(result)
        >>> if confidence < 0.70:
        ...     flags = scorer.generate_review_flags(result)
    """

    def __init__(self, config: QualityConfig) -> None:
        """Initialize the quality scorer with configuration.

        Args:
            config: QualityConfig instance specifying scoring parameters
                and thresholds.

        Raises:
            TypeError: If config is not a QualityConfig instance.
        """
        if not isinstance(config, QualityConfig):
            raise TypeError(f"config must be QualityConfig, got {type(config)}")
        self.config = config

    def calculate_drawing_confidence(
        self, processing_result: ProcessingResult
    ) -> float:
        """Calculate overall confidence score using weighted components.

        Computes a weighted average of OCR, detection, and entity
        extraction confidence scores. Applies any validation penalties
        from the processing result's validation report.

        The formula used is:
            overall = (ocr_weight × ocr_conf) +
                     (detection_weight × det_conf) +
                     (entity_weight × entity_conf)

        If a validation report exists with confidence_adjustment < 1.0,
        the overall score is multiplied by this adjustment factor.

        Args:
            processing_result: Complete ProcessingResult containing OCR
                results, detections, entities, and optional validation
                report.

        Returns:
            Overall confidence score normalized to range [0.0, 1.0].
            Returns 0.0 if no OCR results, detections, or entities exist.

        Note:
            Missing components (e.g., no detections) contribute 0.0 to
            their weighted portion, reducing the overall score.
        """
        drawing_id = processing_result.drawing_id

        # OCR confidence
        ocr_conf = 0.0
        if (
            processing_result.ocr_result is not None
            and processing_result.ocr_result.text_blocks
        ):
            ocr_conf = processing_result.ocr_result.average_confidence
        else:
            logger.debug(f"Drawing {drawing_id}: No OCR results, using 0.0 confidence")

        # Detection confidence
        det_conf = 0.0
        if processing_result.detections:
            total_conf = sum(d.confidence for d in processing_result.detections)
            det_conf = total_conf / len(processing_result.detections)
        else:
            logger.debug(f"Drawing {drawing_id}: No detections, using 0.0 confidence")

        # Entity confidence
        entity_conf = 0.0
        if processing_result.entities:
            total_conf = sum(e.confidence for e in processing_result.entities)
            entity_conf = total_conf / len(processing_result.entities)
        else:
            logger.debug(f"Drawing {drawing_id}: No entities, using 0.0 confidence")

        # Weighted score
        overall = (
            self.config.ocr_weight * ocr_conf
            + self.config.detection_weight * det_conf
            + self.config.entity_weight * entity_conf
        )

        # Apply validation penalty if present
        if (
            processing_result.validation_report is not None
            and processing_result.validation_report.confidence_adjustment < 1.0
        ):
            penalty = processing_result.validation_report.confidence_adjustment
            overall *= penalty
            logger.debug(
                f"Drawing {drawing_id}: Applied validation penalty " f"{penalty:.2f}"
            )

        logger.info(
            f"Drawing {drawing_id}: Confidence scores - "
            f"OCR={ocr_conf:.3f}, Detection={det_conf:.3f}, "
            f"Entity={entity_conf:.3f}, Overall={overall:.3f}"
        )

        return overall

    def generate_review_flags(
        self, processing_result: ProcessingResult
    ) -> List[ReviewFlag]:
        """Generate human review flags based on quality assessment.

        Analyzes the processing result and creates ReviewFlag instances
        for various quality issues, including:
            1. Low overall confidence (below review_threshold)
            2. Missing entities entirely
            3. Missing shape detections
            4. Missing critical entity types (e.g., PART_NUMBER)
            5. High/critical severity validation issues
            6. Processing failures

        Args:
            processing_result: Complete ProcessingResult to evaluate.
                The overall_confidence field will be recalculated if
                not already set.

        Returns:
            List of ReviewFlag instances. Empty list if no issues found.
            Flags are ordered by generation sequence (not severity).

        Raises:
            ValueError: If processing_result is malformed.
        """
        drawing_id = processing_result.drawing_id
        flags: List[ReviewFlag] = []

        # Ensure overall confidence is calculated
        overall_conf = processing_result.overall_confidence
        if overall_conf is None or overall_conf == 0.0:
            overall_conf = self.calculate_drawing_confidence(processing_result)
            logger.debug(
                f"Drawing {drawing_id}: Recalculated confidence " f"{overall_conf:.3f}"
            )

        # Flag 1: Low overall confidence
        if overall_conf < self.config.review_threshold:
            flags.append(self._flag_low_confidence(drawing_id, overall_conf))

        # Flag 2: Missing entities
        if not processing_result.entities:
            flags.append(self._flag_missing_entities(drawing_id))

        # Flag 3: Missing detections
        if not processing_result.detections:
            flags.append(self._flag_missing_detections(drawing_id))

        # Flag 4: Missing critical entities
        if self.config.flag_missing_critical_entities:
            missing_critical = self._check_critical_entities(processing_result)
            if missing_critical:
                flags.append(
                    self._flag_missing_critical_entities(drawing_id, missing_critical)
                )

        # Flag 5: Validation issues
        if processing_result.validation_report is not None:
            validation_flags = self._flag_validation_issues(
                drawing_id, processing_result.validation_report
            )
            flags.extend(validation_flags)

        # Flag 6: Processing errors
        if processing_result.status == STATUS_FAILED:
            flags.append(self._flag_processing_failure(drawing_id))

        logger.info(f"Drawing {drawing_id}: Generated {len(flags)} review flags")
        return flags

    def assess_completeness(
        self, processing_result: ProcessingResult
    ) -> CompletenessScore:
        """Assess data completeness across multiple categories.

        Evaluates the presence and quality of extracted data including:
            - Critical entities (part number, dimensions, shapes)
            - Entity type coverage (weighted by importance)
            - Text-to-shape associations
            - Title block information

        The overall completeness score is a weighted average using
        configurable weights from QualityConfig.

        Args:
            processing_result: Complete ProcessingResult to assess.

        Returns:
            CompletenessScore instance containing:
                - overall_score: Weighted average [0.0, 1.0]
                - Boolean flags for critical field presence
                - List of missing critical field names
                - Category-specific completeness scores

        Note:
            A score of 1.0 indicates all expected data present with
            high coverage. Scores below 0.5 typically indicate
            significant data gaps requiring review.
        """
        drawing_id = processing_result.drawing_id

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
        missing_critical: List[str] = []
        if not has_part_number:
            missing_critical.append(EntityType.PART_NUMBER.value)
        if not has_dimensions:
            missing_critical.append(EntityType.DIMENSION.value)
        if not has_shapes:
            missing_critical.append("SHAPES")

        # Calculate completeness by category
        entity_completeness = self._calculate_entity_completeness(processing_result)
        shape_completeness = 1.0 if has_shapes else 0.0
        association_completeness = self._calculate_association_completeness(
            processing_result
        )
        # Title block: full credit if present, partial credit if missing
        title_block_completeness = (
            1.0 if has_title_block else self.config.title_block_partial_score
        )

        completeness_by_category = {
            "entities": entity_completeness,
            "shapes": shape_completeness,
            "associations": association_completeness,
            "title_block": title_block_completeness,
        }

        # Overall score (weighted average using config)
        overall_score = (
            self.config.completeness_entity_weight * entity_completeness
            + self.config.completeness_shape_weight * shape_completeness
            + self.config.completeness_association_weight * association_completeness
            + self.config.completeness_title_block_weight * title_block_completeness
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

        logger.info(
            f"Drawing {drawing_id}: Completeness={overall_score:.3f} "
            f"(entities={entity_completeness:.3f}, "
            f"shapes={shape_completeness:.3f}, "
            f"associations={association_completeness:.3f}, "
            f"title_block={title_block_completeness:.3f})"
        )

        return completeness

    # ===================================================================
    # Private Helper Methods: Critical Entity Checking
    # ===================================================================

    def _check_critical_entities(
        self, processing_result: ProcessingResult
    ) -> List[str]:
        """Identify missing critical entity types.

        Compares extracted entity types against the configured list
        of critical entity types. Returns entity type names that
        were expected but not found.

        Args:
            processing_result: ProcessingResult containing extracted
                entities.

        Returns:
            List of missing critical entity type names (strings).
            Empty list if all critical entities are present.
        """
        found_types = {e.entity_type for e in processing_result.entities}

        missing = [
            entity_type.value
            for entity_type in self.config.critical_entities
            if entity_type not in found_types
        ]

        return missing

    # ===================================================================
    # Private Helper Methods: Severity Classification
    # ===================================================================

    def _classify_severity(self, confidence_score: float) -> Severity:
        """Classify severity level based on confidence score.

        Maps confidence scores to severity levels using configurable
        thresholds from QualityConfig.

        Args:
            confidence_score: Confidence value in range [0.0, 1.0].

        Returns:
            Severity enum value corresponding to the confidence level.

        Note:
            Lower confidence produces higher severity. This is used
            primarily for low-confidence review flags.
        """
        if confidence_score < self.config.severity_critical_threshold:
            return Severity.CRITICAL
        elif confidence_score < self.config.severity_high_threshold:
            return Severity.HIGH
        elif confidence_score < self.config.severity_medium_threshold:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    # ===================================================================
    # Private Helper Methods: Completeness Calculations
    # ===================================================================

    def _calculate_entity_completeness(
        self, processing_result: ProcessingResult
    ) -> float:
        """Calculate entity extraction completeness score.

        Evaluates the presence of expected entity types, weighted by
        their importance. The score reflects both coverage (which types
        were found) and importance (critical types weighted higher).

        Uses entity importance weights from QualityConfig. The score is
        calculated as: sum(found_weights) / sum(all_expected_weights).

        Args:
            processing_result: ProcessingResult containing extracted
                entities.

        Returns:
            Completeness score in range [0.0, 1.0]. Returns 0.0 if no
            entities were extracted. Score of 1.0 indicates all expected
            entity types were found.

        Note:
            This is a subset assessment - not all entity types are
            included in the expected set. Additional entities don't
            increase the score beyond 1.0.
        """
        if not processing_result.entities:
            return 0.0

        expected_types = self.config.entity_importance_weights

        # Calculate score based on presence and importance
        found_types = {e.entity_type for e in processing_result.entities}

        total_weight = sum(expected_types.values())
        achieved_weight = sum(
            weight
            for entity_type, weight in expected_types.items()
            if entity_type in found_types
        )

        return achieved_weight / total_weight if total_weight > 0 else 0.0

    def _calculate_association_completeness(
        self, processing_result: ProcessingResult
    ) -> float:
        """Calculate text-to-shape association completeness score.

        Uses a heuristic based on the ratio of text blocks that have
        been associated with shapes. Since not all text requires
        association (e.g., notes, disclaimers), the expected association
        rate is capped at a configurable threshold.

        Args:
            processing_result: ProcessingResult containing OCR text blocks
                and associations.

        Returns:
            Completeness score in range [0.0, 1.0]. Returns 0.0 if no
            OCR text blocks exist. A score of 1.0 indicates the expected
            percentage of text blocks have been associated with shapes.

        Note:
            The expected rate threshold is configurable in QualityConfig
            and is based on typical engineering drawing text distribution.
        """
        if (
            processing_result.ocr_result is None
            or not processing_result.ocr_result.text_blocks
        ):
            return 0.0

        total_text_blocks = len(processing_result.ocr_result.text_blocks)
        associated_blocks = len(processing_result.associations)

        # Cap at expected rate
        expected_rate = self.config.max_expected_association_rate
        actual_rate = min(associated_blocks / total_text_blocks, expected_rate)

        return actual_rate / expected_rate if expected_rate > 0 else 0.0

    # ===================================================================
    # Private Helper Methods: Flag Generation
    # ===================================================================

    def _flag_low_confidence(self, drawing_id: str, confidence: float) -> ReviewFlag:
        """Generate flag for low overall confidence.

        Args:
            drawing_id: Drawing identifier for context.
            confidence: Overall confidence score.

        Returns:
            ReviewFlag for low confidence issue.
        """
        return ReviewFlag(
            flag_id=str(uuid.uuid4()),
            flag_type=FlagType.LOW_CONFIDENCE,
            severity=self._classify_severity(confidence),
            reason=(
                f"Overall confidence {confidence:.3f} below threshold "
                f"{self.config.review_threshold:.3f}"
            ),
            details={"confidence": confidence, "drawing_id": drawing_id},
            suggested_action=("Review extraction results and consider LLM enhancement"),
            affected_entities=[],
            affected_shapes=[],
        )

    def _flag_missing_entities(self, drawing_id: str) -> ReviewFlag:
        """Generate flag for missing entity extractions.

        Args:
            drawing_id: Drawing identifier for context.

        Returns:
            ReviewFlag for missing entities issue.
        """
        return ReviewFlag(
            flag_id=str(uuid.uuid4()),
            flag_type=FlagType.MISSING_DATA,
            severity=Severity.HIGH,
            reason="No entities extracted from drawing",
            details={"drawing_id": drawing_id},
            suggested_action=("Verify OCR quality and run entity extraction again"),
            affected_entities=[],
            affected_shapes=[],
        )

    def _flag_missing_detections(self, drawing_id: str) -> ReviewFlag:
        """Generate flag for missing shape detections.

        Args:
            drawing_id: Drawing identifier for context.

        Returns:
            ReviewFlag for missing detections issue.
        """
        return ReviewFlag(
            flag_id=str(uuid.uuid4()),
            flag_type=FlagType.MISSING_DATA,
            severity=Severity.HIGH,
            reason="No shapes detected in drawing",
            details={"drawing_id": drawing_id},
            suggested_action="Check image quality and detection model",
            affected_entities=[],
            affected_shapes=[],
        )

    def _flag_missing_critical_entities(
        self, drawing_id: str, missing: List[str]
    ) -> ReviewFlag:
        """Generate flag for missing critical entity types.

        Args:
            drawing_id: Drawing identifier for context.
            missing: List of missing critical entity type names.

        Returns:
            ReviewFlag for missing critical entities issue.
        """
        return ReviewFlag(
            flag_id=str(uuid.uuid4()),
            flag_type=FlagType.CRITICAL_FIELD_MISSING,
            severity=Severity.CRITICAL,
            reason=f"Missing critical entities: {', '.join(missing)}",
            details={
                "missing_entities": missing,
                "drawing_id": drawing_id,
            },
            suggested_action="Run LLM entity enhancement or manual review",
            affected_entities=[],
            affected_shapes=[],
        )

    def _flag_validation_issues(
        self, drawing_id: str, validation_report
    ) -> List[ReviewFlag]:
        """Generate flags from validation report issues.

        Args:
            drawing_id: Drawing identifier for context.
            validation_report: ValidationReport containing issues.

        Returns:
            List of ReviewFlags for high/critical validation issues.
        """
        flags: List[ReviewFlag] = []

        for issue in validation_report.issues:
            if issue.severity not in ["HIGH", "CRITICAL"]:
                continue

            try:
                severity_enum = Severity[issue.severity]
            except KeyError:
                logger.warning(
                    f"Drawing {drawing_id}: Invalid severity "
                    f"'{issue.severity}' in validation issue, "
                    f"defaulting to MEDIUM"
                )
                severity_enum = Severity.MEDIUM

            flag = ReviewFlag(
                flag_id=str(uuid.uuid4()),
                flag_type=FlagType.INCONSISTENCY,
                severity=severity_enum,
                reason=issue.message,
                details={
                    "issue_type": issue.type,
                    "drawing_id": drawing_id,
                },
                suggested_action=(
                    issue.suggested_fix
                    if issue.suggested_fix
                    else "Manual review required"
                ),
                affected_entities=([issue.entity_id] if issue.entity_id else []),
                affected_shapes=([issue.shape_id] if issue.shape_id else []),
            )
            flags.append(flag)

        return flags

    def _flag_processing_failure(self, drawing_id: str) -> ReviewFlag:
        """Generate flag for processing failures.

        Args:
            drawing_id: Drawing identifier for context.

        Returns:
            ReviewFlag for processing failure.
        """
        return ReviewFlag(
            flag_id=str(uuid.uuid4()),
            flag_type=FlagType.ERROR,
            severity=Severity.CRITICAL,
            reason="Processing failed or encountered errors",
            details={"drawing_id": drawing_id},
            suggested_action="Review error logs and retry processing",
            affected_entities=[],
            affected_shapes=[],
        )
