# orchestration/routing_engine.py
"""Routing engine with multi-dimensional confidence assessment.

This module implements intelligent pipeline routing between baseline (open-source only),
hybrid (selective LLM), and full LLM-enhanced processing paths based on multi-dimensional
confidence scoring with critical PART_NUMBER field validation.
"""

import logging
from dataclasses import dataclass, field, InitVar
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import canonical classes from data_structures
from ..models.data_structures import (
    PipelineType,
    ProcessingStage,
    Priority,
    EntityType,
    Entity,
    Detection,
    ProcessingResult,
    DrawingRecord,
)
from ..utils.config_loader import SystemConfig
from ..llm.budget_controller import BudgetController

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfidenceScores:
    """Multi-dimensional confidence assessment for routing decisions.

    All scores must be in range [0.0, 1.0]. Weights for overall_weighted()
    are validated to sum to 1.0 with floating-point tolerance.

    Attributes:
        ocr_quality: OCR confidence score (0.0-1.0).
        entity_completeness: Entity extraction completeness weighted by importance (0.0-1.0).
        shape_detection_quality: Shape detection confidence score (0.0-1.0).
        critical_field_presence: PART_NUMBER presence and confidence (0.0-1.0).
        data_consistency: Cross-validation consistency score (0.0-1.0).
        weights: Dimension weights for overall score calculation (must sum to 1.0).
    """

    ocr_quality: float
    entity_completeness: float
    shape_detection_quality: float
    critical_field_presence: float
    data_consistency: float
    weights: InitVar[Optional[Dict[str, float]]] = None

    # Store weights as hidden field
    _weights: Dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self, weights: Optional[Dict[str, float]]) -> None:
        """Validate score ranges and weight sum.

        Args:
            weights: Optional custom weights dict. If None, uses default weights.

        Raises:
            ValueError: If any score is outside [0.0, 1.0] or weights don't sum to 1.0.
        """
        # Validate score ranges
        scores = {
            "ocr_quality": self.ocr_quality,
            "entity_completeness": self.entity_completeness,
            "shape_detection_quality": self.shape_detection_quality,
            "critical_field_presence": self.critical_field_presence,
            "data_consistency": self.data_consistency,
        }

        for name, score in scores.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Confidence score '{name}' must be in [0.0, 1.0], got {score}"
                )

        # Set weights (use default if not provided)
        default_weights = {
            "ocr_quality": 0.20,
            "entity_completeness": 0.35,
            "shape_detection_quality": 0.25,
            "critical_field_presence": 0.15,
            "data_consistency": 0.05,
        }

        final_weights = weights if weights is not None else default_weights

        # Validate weights sum to 1.0 (with tolerance for floating-point)
        weight_sum = sum(final_weights.values())
        if not abs(weight_sum - 1.0) < 1e-6:
            raise ValueError(
                f"Confidence weights must sum to 1.0, got {weight_sum:.6f}"
            )

        # Store weights (workaround for frozen dataclass)
        object.__setattr__(self, "_weights", final_weights)

    def overall_weighted(self) -> float:
        """Calculate weighted overall score for routing decisions.

        Default weights:
            - Entity completeness: 35% (highest priority)
            - Shape detection: 25%
            - OCR quality: 20%
            - Critical field: 15%
            - Data consistency: 5%

        Returns:
            Weighted confidence score from 0.0 to 1.0.
        """
        return (
            self._weights["ocr_quality"] * self.ocr_quality
            + self._weights["entity_completeness"] * self.entity_completeness
            + self._weights["shape_detection_quality"] * self.shape_detection_quality
            + self._weights["critical_field_presence"] * self.critical_field_presence
            + self._weights["data_consistency"] * self.data_consistency
        )


@dataclass(frozen=True)
class ProcessingRoute:
    """Pipeline routing decision with cost estimation.

    Attributes:
        pipeline: Selected pipeline type (baseline/hybrid/LLM-enhanced).
        llm_stages: List of stages requiring LLM enhancement.
        reason: Standardized explanation code for routing decision.
        reason_detail: Human-readable explanation with context.
        estimated_cost: Estimated USD cost for LLM calls.
        confidence_scores: Optional multi-dimensional confidence breakdown.
        forced_by_critical_field: Flag indicating critical field gate triggered decision.
    """

    pipeline: PipelineType
    llm_stages: List[ProcessingStage]
    reason: str  # Standardized code: "MISSING_PART_NUMBER", "LOW_CONFIDENCE_PN", etc.
    reason_detail: str  # Human-readable explanation
    estimated_cost: float
    confidence_scores: Optional[ConfidenceScores] = None
    forced_by_critical_field: bool = False


class RoutingEngine:
    """Intelligent routing engine for pipeline selection based on confidence assessment.

    Determines optimal processing path (baseline/hybrid/LLM-enhanced) using
    multi-dimensional confidence scoring with critical field validation. Enforces
    PART_NUMBER gate and budget constraints per project requirements.

    Attributes:
        config: System configuration containing routing thresholds and sampling rates.
        budget: Budget controller for cost tracking and LLM call authorization.
        rng: NumPy random generator for reproducible sampling.
        entity_weights: Importance weights for entity completeness calculation.
        confidence_weights: Weights for overall confidence calculation.
        part_number_min_confidence: Minimum confidence threshold for PART_NUMBER.
        overall_threshold_high: Confidence threshold for baseline-only processing.
        overall_threshold_medium: Confidence threshold for hybrid processing.
        missing_entity_partial_credit: Credit factor for missing non-critical entities.
        ocr_text_block_norm: Normalization factor for text block count.
        ocr_confidence_weight: Weight for OCR confidence in quality assessment.
        shape_count_norm: Normalization factor for shape detection count.
        shape_confidence_weight: Weight for shape confidence in quality assessment.

    Example:
        >>> config = SystemConfig.load()
        >>> budget = BudgetController(config)
        >>> engine = RoutingEngine(config, budget, random_seed=42)
        >>> route = engine.determine_route(drawing, processing_result)
        >>> print(f"Pipeline: {route.pipeline.value}, Reason: {route.reason}")
    """

    def __init__(
        self,
        config: SystemConfig,
        budget_controller: BudgetController,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize routing engine with configuration and budget controller.

        Args:
            config: System configuration object containing routing parameters.
            budget_controller: Budget controller for LLM cost management.
            random_seed: Optional random seed for reproducible sampling (for testing).

        Raises:
            ValueError: If required config sections are missing or invalid.
        """
        self.config = config
        self.budget = budget_controller
        self.rng = np.random.default_rng(random_seed)

        # Load routing configuration
        try:
            routing_config = config.get("routing", {})
        except Exception as e:
            logger.error("Failed to load routing config: %s", e)
            routing_config = {}

        # Load entity importance weights (can be overridden in config)
        default_entity_weights = {
            "PART_NUMBER": 0.40,  # CRITICAL
            "OEM": 0.15,
            "MATERIAL": 0.10,
            "DIMENSION": 0.15,
            "WEIGHT": 0.05,
            "THREAD_SPEC": 0.05,
            "TOLERANCE": 0.05,
            "SURFACE_FINISH": 0.05,
        }
        self.entity_weights = routing_config.get(
            "entity_weights", default_entity_weights
        )

        # Validate entity weights sum to 1.0
        weight_sum = sum(self.entity_weights.values())
        if not abs(weight_sum - 1.0) < 1e-6:
            logger.warning(
                "Entity weights sum to %.6f (expected 1.0), normalizing", weight_sum
            )
            # Normalize weights
            self.entity_weights = {
                k: v / weight_sum for k, v in self.entity_weights.items()
            }

        # Load confidence dimension weights
        default_confidence_weights = {
            "ocr_quality": 0.20,
            "entity_completeness": 0.35,
            "shape_detection_quality": 0.25,
            "critical_field_presence": 0.15,
            "data_consistency": 0.05,
        }
        self.confidence_weights = routing_config.get(
            "confidence_weights", default_confidence_weights
        )

        # Load thresholds
        self.part_number_min_confidence = routing_config.get(
            "part_number_min_confidence", 0.70
        )
        self.overall_threshold_high = routing_config.get("overall_threshold_high", 0.85)
        self.overall_threshold_medium = routing_config.get(
            "overall_threshold_medium", 0.70
        )

        # Load scoring parameters
        self.missing_entity_partial_credit = routing_config.get(
            "missing_entity_partial_credit", 0.3
        )
        self.ocr_text_block_norm = routing_config.get("ocr_text_block_norm", 20)
        self.ocr_confidence_weight = routing_config.get("ocr_confidence_weight", 0.7)
        self.shape_count_norm = routing_config.get("shape_count_norm", 5)
        self.shape_confidence_weight = routing_config.get(
            "shape_confidence_weight", 0.8
        )

        # Sampling rate
        self.default_sampling_rate = routing_config.get("default_sampling_rate", 0.08)

    def determine_route(
        self,
        drawing: DrawingRecord,
        initial_result: ProcessingResult,
    ) -> ProcessingRoute:
        """Evaluate processing result and determine optimal pipeline routing.

        Workflow:
        1. Validate inputs
        2. Calculate all confidence dimensions
        3. Apply routing rules (including critical field gate)
        4. Reserve budget if LLM path selected

        Args:
            drawing: Drawing record containing ID and priority information.
            initial_result: Baseline processing result with entities and detections.

        Returns:
            ProcessingRoute object specifying pipeline type, LLM stages, reasoning,
            and cost estimate.

        Raises:
            ValueError: If inputs are invalid.
            RuntimeError: If budget controller fails.

        Example:
            >>> result = ProcessingResult(entities=[...], detections=[...], ...)
            >>> route = engine.determine_route(drawing, result)
            >>> if route.forced_by_critical_field:
            ...     logger.warning("Critical field issue: %s", route.reason_detail)
        """
        # === STEP 1: INPUT VALIDATION ===
        self._validate_inputs(drawing, initial_result)

        # === STEP 2: CALCULATE ALL CONFIDENCE DIMENSIONS ===
        try:
            part_number_check = self._check_part_number(initial_result.entities)

            confidence_scores = ConfidenceScores(
                ocr_quality=self._assess_ocr_quality(initial_result),
                entity_completeness=self._assess_entity_completeness(
                    initial_result.entities
                ),
                shape_detection_quality=self._assess_shape_quality(
                    initial_result.detections
                ),
                critical_field_presence=part_number_check["confidence"],
                data_consistency=self._validate_data_consistency(initial_result),
                weights=self.confidence_weights,
            )
        except Exception as e:
            logger.error(
                "Failed to calculate confidence scores for drawing %s: %s",
                drawing.drawing_id,
                e,
            )
            raise

        overall_score = confidence_scores.overall_weighted()

        # Log detailed confidence breakdown
        logger.info(
            "Drawing %s confidence assessment:\n"
            "  OCR Quality:          %.3f\n"
            "  Entity Completeness:  %.3f\n"
            "  Shape Detection:      %.3f\n"
            "  Critical Fields:      %.3f\n"
            "  Data Consistency:     %.3f\n"
            "  Overall Weighted:     %.3f\n"
            "  PART_NUMBER Status:   exists=%s, confidence=%.3f, value='%s'",
            drawing.drawing_id,
            confidence_scores.ocr_quality,
            confidence_scores.entity_completeness,
            confidence_scores.shape_detection_quality,
            confidence_scores.critical_field_presence,
            confidence_scores.data_consistency,
            overall_score,
            part_number_check["exists"],
            part_number_check["confidence"],
            part_number_check["value"],
        )

        # === STEP 3: APPLY ROUTING RULES ===
        try:
            route = self._apply_routing_rules(
                confidence_scores,
                overall_score,
                drawing,
                part_number_check,
            )
            route = dataclass_replace(route, confidence_scores=confidence_scores)
        except Exception as e:
            logger.error(
                "Failed to determine route for drawing %s: %s",
                drawing.drawing_id,
                e,
            )
            raise

        # === STEP 4: BUDGET RESERVATION (if LLM path selected) ===
        if route.estimated_cost > 0:
            try:
                self._reserve_budget(route.estimated_cost, drawing.drawing_id)
            except Exception as e:
                logger.warning(
                    "Budget reservation failed for drawing %s: %s. "
                    "Falling back to baseline.",
                    drawing.drawing_id,
                    e,
                )
                # Fallback to baseline if budget reservation fails
                route = ProcessingRoute(
                    pipeline=PipelineType.BASELINE_ONLY,
                    llm_stages=[],
                    reason="BUDGET_RESERVATION_FAILED",
                    reason_detail=f"Budget reservation failed: {e}",
                    estimated_cost=0.0,
                    confidence_scores=confidence_scores,
                )

        logger.info(
            "Drawing %s routing decision: pipeline=%s, reason=%s, cost=$%.4f",
            drawing.drawing_id,
            route.pipeline.value,
            route.reason,
            route.estimated_cost,
        )

        return route

    def _validate_inputs(
        self,
        drawing: DrawingRecord,
        result: ProcessingResult,
    ) -> None:
        """Validate input parameters for routing decision.

        Args:
            drawing: Drawing record to validate.
            result: Processing result to validate.

        Raises:
            ValueError: If inputs are missing required attributes or invalid.
        """
        # Validate drawing
        if not hasattr(drawing, "drawing_id") or not drawing.drawing_id:
            raise ValueError("Drawing must have non-empty drawing_id")
        if not hasattr(drawing, "priority"):
            raise ValueError("Drawing must have priority attribute")

        # Validate result
        if not hasattr(result, "entities"):
            raise ValueError("ProcessingResult must have entities attribute")
        if not hasattr(result, "detections"):
            raise ValueError("ProcessingResult must have detections attribute")
        if not hasattr(result, "ocr_avg_confidence"):
            raise ValueError("ProcessingResult must have ocr_avg_confidence")
        if not hasattr(result, "text_block_count"):
            raise ValueError("ProcessingResult must have text_block_count")

        # Validate confidence ranges
        if not 0.0 <= result.ocr_avg_confidence <= 1.0:
            raise ValueError(
                f"OCR confidence must be in [0.0, 1.0], got {result.ocr_avg_confidence}"
            )

    def _check_part_number(self, entities: List[Entity]) -> Dict[str, Any]:
        """Check PART_NUMBER entity presence and confidence level.

        Args:
            entities: List of extracted entities to search.

        Returns:
            Dictionary containing:
                - 'exists': bool indicating if PART_NUMBER was found
                - 'confidence': float score (0.0 if not found)
                - 'value': str value of part number or None

        Example:
            >>> result = self._check_part_number(entities)
            >>> if result['exists'] and result['confidence'] >= 0.70:
            ...     print(f"Valid part number: {result['value']}")
        """
        part_numbers = [
            e
            for e in entities
            if e.entity_type == EntityType.PART_NUMBER
            or e.entity_type == "PART_NUMBER"  # Support string fallback
        ]

        if not part_numbers:
            return {"exists": False, "confidence": 0.0, "value": None}

        # If multiple part numbers, use highest confidence (deterministic tie-break by value)
        best_pn = max(
            part_numbers,
            key=lambda e: (e.confidence, e.value),  # Tie-break by value
        )

        return {
            "exists": True,
            "confidence": best_pn.confidence,
            "value": best_pn.value,
        }

    def _assess_entity_completeness(self, entities: List[Entity]) -> float:
        """Calculate entity completeness score using weighted importance.

        Evaluates presence and confidence of expected entity types using
        predefined importance weights (PART_NUMBER: 0.40, OEM: 0.15, etc.).
        Missing critical entities receive zero points; missing optional entities
        receive partial credit (configurable, default 0.3 * weight).

        Args:
            entities: List of extracted entities.

        Returns:
            Float score from 0.0 (no entities) to 1.0 (all critical entities
            present with high confidence).
        """
        score = 0.0
        entity_types_found: Dict[str, List[Entity]] = {}

        for e in entities:
            # Support both enum and string entity types
            entity_type_str = (
                e.entity_type.name
                if isinstance(e.entity_type, EntityType)
                else e.entity_type
            )
            if entity_type_str not in entity_types_found:
                entity_types_found[entity_type_str] = []
            entity_types_found[entity_type_str].append(e)

        # Check presence and confidence of each weighted entity type
        for entity_type, weight in self.entity_weights.items():
            if entity_type in entity_types_found and entity_types_found[entity_type]:
                # Average confidence of entities of this type (deterministic tie-break)
                avg_confidence = float(
                    np.mean([e.confidence for e in entity_types_found[entity_type]])
                )
                score += weight * avg_confidence
            else:
                # Penalty for missing entity types
                if entity_type == "PART_NUMBER":
                    score += 0.0  # Critical miss: 0 points
                else:
                    score += weight * self.missing_entity_partial_credit

        # Log unexpected entity types (not in weights)
        unexpected_types = set(entity_types_found.keys()) - set(
            self.entity_weights.keys()
        )
        if unexpected_types:
            logger.debug(
                "Found entity types not in weight configuration: %s",
                unexpected_types,
            )

        return min(score, 1.0)  # Cap at 1.0

    def _assess_ocr_quality(self, result: ProcessingResult) -> float:
        """Assess OCR quality based on confidence and text block density.

        Args:
            result: Processing result containing OCR metrics.

        Returns:
            Float score from 0.0 to 1.0 combining average confidence (default 70%)
            and text block count normalized (default 30%).
        """
        if result.text_block_count == 0:
            return 0.0

        # Normalize text block count
        text_score = min(result.text_block_count / self.ocr_text_block_norm, 1.0)

        return (
            self.ocr_confidence_weight * result.ocr_avg_confidence
            + (1 - self.ocr_confidence_weight) * text_score
        )

    def _assess_shape_quality(self, detections: List[Detection]) -> float:
        """Assess shape detection quality from detection results.

        Args:
            detections: List of shape detections.

        Returns:
            Float score from 0.0 to 1.0 combining average confidence (default 80%)
            and detection count normalized (default 20%).
        """
        if not detections:
            return 0.0

        avg_conf = float(np.mean([d.confidence for d in detections]))
        count_score = min(len(detections) / self.shape_count_norm, 1.0)

        return (
            self.shape_confidence_weight * avg_conf
            + (1 - self.shape_confidence_weight) * count_score
        )

    def _validate_data_consistency(self, result: ProcessingResult) -> float:
        """Cross-validate data consistency across processing results.

        Simplified implementation checking for presence of complementary data types.
        Full implementation would validate entity-shape associations, dimension
        compatibility, and orphaned entities.

        Args:
            result: Processing result to validate.

        Returns:
            Float score: 0.9 (all data types present), 0.7 (text + entities),
            or 0.5 (incomplete data).

        Note:
            This is a placeholder. Production implementation should perform
            actual cross-validation of entity-shape associations.
        """
        has_text = result.text_block_count > 0
        has_shapes = len(result.detections) > 0
        has_entities = len(result.entities) > 0

        if has_text and has_shapes and has_entities:
            return 0.9
        elif has_text and has_entities:
            return 0.7
        else:
            return 0.5

    def _apply_routing_rules(
        self,
        scores: ConfidenceScores,
        overall_score: float,
        drawing: DrawingRecord,
        part_number_check: Dict[str, Any],
    ) -> ProcessingRoute:
        """Apply decision tree for pipeline routing based on confidence dimensions.

        Routing logic (priority order):
        1. Missing PART_NUMBER → Force LLM (if budget + not LOW priority)
        2. Low PART_NUMBER confidence → Hybrid LLM verification
        3. High overall confidence (≥0.85) + high critical field (≥0.9) → Baseline
        4. Low entity completeness (<0.75) → Hybrid entity extraction
        5. Poor OCR (<0.85) + poor entities (<0.80) → Hybrid OCR + entity
        6. Medium confidence (0.70-0.85) → Hybrid on weakest dimension
        7. Fallback → Baseline (budget exhausted or low sampling)

        Args:
            scores: Multi-dimensional confidence scores.
            overall_score: Weighted overall confidence score.
            drawing: Drawing record for priority/sampling checks.
            part_number_check: PART_NUMBER validation results.

        Returns:
            ProcessingRoute with pipeline decision and rationale.
        """
        # === RULE 1: MISSING PART_NUMBER (Critical Field Gate) ===
        if not part_number_check["exists"]:
            logger.debug(
                "Drawing %s: PART_NUMBER missing, checking budget/priority",
                drawing.drawing_id,
            )

            if self._budget_available() and drawing.priority != Priority.LOW:
                return ProcessingRoute(
                    pipeline=PipelineType.LLM_ENHANCED,
                    llm_stages=[ProcessingStage.ENTITY_EXTRACTION],
                    reason="MISSING_PART_NUMBER",
                    reason_detail=(
                        "CRITICAL: Missing PART_NUMBER → force LLM enhancement"
                    ),
                    estimated_cost=self._estimate_cost(
                        [ProcessingStage.ENTITY_EXTRACTION]
                    ),
                    forced_by_critical_field=True,
                )
            else:
                return ProcessingRoute(
                    pipeline=PipelineType.BASELINE_ONLY,
                    llm_stages=[],
                    reason="MISSING_PART_NUMBER_NO_BUDGET",
                    reason_detail=(
                        "Missing PART_NUMBER but budget exhausted or LOW priority "
                        "→ flag for manual review"
                    ),
                    estimated_cost=0.0,
                    forced_by_critical_field=True,
                )

        # === RULE 2: LOW PART_NUMBER CONFIDENCE ===
        if part_number_check["confidence"] < self.part_number_min_confidence:
            logger.debug(
                "Drawing %s: PART_NUMBER confidence %.3f below threshold %.3f",
                drawing.drawing_id,
                part_number_check["confidence"],
                self.part_number_min_confidence,
            )

            if self._budget_available():
                return ProcessingRoute(
                    pipeline=PipelineType.HYBRID,
                    llm_stages=[ProcessingStage.ENTITY_EXTRACTION],
                    reason="LOW_PART_NUMBER_CONFIDENCE",
                    reason_detail=(
                        f"Low PART_NUMBER confidence "
                        f"({part_number_check['confidence']:.2f}) → LLM verification"
                    ),
                    estimated_cost=self._estimate_cost(
                        [ProcessingStage.ENTITY_EXTRACTION]
                    ),
                    forced_by_critical_field=True,
                )

        # === RULE 3: HIGH CONFIDENCE → Baseline sufficient ===
        if (
            overall_score >= self.overall_threshold_high
            and scores.critical_field_presence >= 0.9
        ):
            logger.debug(
                "Drawing %s: High confidence (%.3f), using baseline",
                drawing.drawing_id,
                overall_score,
            )
            return ProcessingRoute(
                pipeline=PipelineType.BASELINE_ONLY,
                llm_stages=[],
                reason="HIGH_CONFIDENCE",
                reason_detail=(
                    f"High confidence across all dimensions: {overall_score:.2f}"
                ),
                estimated_cost=0.0,
            )

        # === RULE 4: LOW ENTITY COMPLETENESS ===
        if scores.entity_completeness < 0.75:
            logger.debug(
                "Drawing %s: Low entity completeness %.3f",
                drawing.drawing_id,
                scores.entity_completeness,
            )

            if self._budget_available() and self._sampling_criteria_met(drawing):
                return ProcessingRoute(
                    pipeline=PipelineType.HYBRID,
                    llm_stages=[ProcessingStage.ENTITY_EXTRACTION],
                    reason="LOW_ENTITY_COMPLETENESS",
                    reason_detail=(
                        f"Low entity completeness: " f"{scores.entity_completeness:.2f}"
                    ),
                    estimated_cost=self._estimate_cost(
                        [ProcessingStage.ENTITY_EXTRACTION]
                    ),
                )

        # === RULE 5: POOR OCR + ENTITY QUALITY ===
        if scores.ocr_quality < 0.85 and scores.entity_completeness < 0.80:
            logger.debug(
                "Drawing %s: Poor OCR (%.3f) and entities (%.3f)",
                drawing.drawing_id,
                scores.ocr_quality,
                scores.entity_completeness,
            )

            if self._budget_available():
                return ProcessingRoute(
                    pipeline=PipelineType.HYBRID,
                    llm_stages=[
                        ProcessingStage.OCR_VERIFICATION,
                        ProcessingStage.ENTITY_EXTRACTION,
                    ],
                    reason="POOR_OCR_AND_ENTITIES",
                    reason_detail="Poor OCR quality impacting entity extraction",
                    estimated_cost=self._estimate_cost(
                        [
                            ProcessingStage.OCR_VERIFICATION,
                            ProcessingStage.ENTITY_EXTRACTION,
                        ]
                    ),
                )

        # === RULE 6: MEDIUM CONFIDENCE → Target weakest dimension ===
        if self.overall_threshold_medium <= overall_score < self.overall_threshold_high:
            logger.debug(
                "Drawing %s: Medium confidence %.3f, targeting weakest dimension",
                drawing.drawing_id,
                overall_score,
            )

            if self._budget_available() and self._sampling_criteria_met(drawing):
                weakest_stage = self._identify_weakest_stage(scores)
                return ProcessingRoute(
                    pipeline=PipelineType.HYBRID,
                    llm_stages=[weakest_stage],
                    reason="MEDIUM_CONFIDENCE_WEAK_DIMENSION",
                    reason_detail=(
                        f"Medium confidence: {overall_score:.2f}, "
                        f"targeting weak dimension: {weakest_stage.value}"
                    ),
                    estimated_cost=self._estimate_cost([weakest_stage]),
                )

        # === RULE 7: FALLBACK → Baseline ===
        logger.debug(
            "Drawing %s: Fallback to baseline (budget/sampling criteria not met)",
            drawing.drawing_id,
        )
        return ProcessingRoute(
            pipeline=PipelineType.BASELINE_ONLY,
            llm_stages=[],
            reason="BASELINE_FALLBACK",
            reason_detail="Budget exhausted or sampling criteria not met",
            estimated_cost=0.0,
        )

    def _identify_weakest_stage(self, scores: ConfidenceScores) -> ProcessingStage:
        """Identify processing stage that would benefit most from LLM enhancement.

        Considers all five confidence dimensions and maps to appropriate
        processing stage. Uses deterministic tie-breaking (alphabetical by dimension name).

        Args:
            scores: Multi-dimensional confidence scores.

        Returns:
            ProcessingStage enum corresponding to weakest confidence dimension.
        """
        dimension_to_stage = {
            "critical_field_presence": ProcessingStage.ENTITY_EXTRACTION,
            "data_consistency": ProcessingStage.SHAPE_VALIDATION,
            "entity_completeness": ProcessingStage.ENTITY_EXTRACTION,
            "ocr_quality": ProcessingStage.OCR_VERIFICATION,
            "shape_detection_quality": ProcessingStage.SHAPE_VALIDATION,
        }

        dimensions = [
            ("critical_field_presence", scores.critical_field_presence),
            ("data_consistency", scores.data_consistency),
            ("entity_completeness", scores.entity_completeness),
            ("ocr_quality", scores.ocr_quality),
            ("shape_detection_quality", scores.shape_detection_quality),
        ]

        # Find weakest dimension (deterministic tie-break by name)
        weakest = min(dimensions, key=lambda x: (x[1], x[0]))

        logger.debug(
            "Weakest dimension: %s (score=%.3f) → stage=%s",
            weakest[0],
            weakest[1],
            dimension_to_stage[weakest[0]].value,
        )

        return dimension_to_stage[weakest[0]]

    def _budget_available(self) -> bool:
        """Check if budget allows additional LLM calls.

        Returns:
            True if remaining budget > 0, False otherwise.

        Raises:
            RuntimeError: If budget controller query fails.
        """
        try:
            summary = self.budget.get_usage_summary()
            remaining = summary.get("remaining_budget", 0.0)

            logger.debug("Budget check: remaining=$%.4f", remaining)
            return remaining > 0

        except Exception as e:
            logger.error("Budget availability check failed: %s", e)
            raise RuntimeError(f"Budget controller error: {e}") from e

    def _sampling_criteria_met(self, drawing: DrawingRecord) -> bool:
        """Determine if drawing meets sampling criteria for LLM enhancement.

        Uses numpy random generator for reproducible sampling (testing).

        Args:
            drawing: Drawing record (used for future priority-based sampling).

        Returns:
            True if random sample falls within sampling rate threshold.

        Note:
            Currently implements simple random sampling. Future enhancement:
            stratified sampling by drawing priority/complexity.
        """
        sample = self.rng.random()
        meets_criteria = sample < self.default_sampling_rate

        logger.debug(
            "Sampling check for drawing %s: sample=%.4f, rate=%.4f, result=%s",
            drawing.drawing_id,
            sample,
            self.default_sampling_rate,
            meets_criteria,
        )

        return meets_criteria

    def _estimate_cost(self, stages: List[ProcessingStage]) -> float:
        """Estimate USD cost for given LLM processing stages.

        Queries BudgetController for dynamic, up-to-date pricing based on
        configured models and current provider rates.

        Args:
            stages: List of processing stages requiring LLM calls.

        Returns:
            Estimated total cost in USD.

        Raises:
            RuntimeError: If cost estimation fails.
        """
        try:
            total_cost = 0.0

            for stage in stages:
                # Query budget controller for stage-specific cost
                # Assumes BudgetController has estimate_stage_cost method
                if hasattr(self.budget, "estimate_stage_cost"):
                    stage_cost = self.budget.estimate_stage_cost(stage)
                else:
                    # Fallback to conservative estimates if method unavailable
                    fallback_costs = {
                        ProcessingStage.OCR_VERIFICATION: 0.02,
                        ProcessingStage.ENTITY_EXTRACTION: 0.08,
                        ProcessingStage.SHAPE_VALIDATION: 0.04,
                    }
                    stage_cost = fallback_costs.get(stage, 0.05)
                    logger.debug(
                        "Using fallback cost estimation for stage %s: $%.4f",
                        stage.value,
                        stage_cost,
                    )

                total_cost += stage_cost

            logger.debug(
                "Estimated cost for stages %s: $%.4f",
                [s.value for s in stages],
                total_cost,
            )

            return total_cost

        except Exception as e:
            logger.error("Cost estimation failed: %s", e)
            raise RuntimeError(f"Cost estimation error: {e}") from e

    def _reserve_budget(self, cost: float, drawing_id: str) -> None:
        """Reserve budget for upcoming LLM calls.

        Args:
            cost: Amount to reserve in USD.
            drawing_id: Drawing identifier for audit logging.

        Raises:
            RuntimeError: If budget reservation fails or insufficient funds.
        """
        try:
            if hasattr(self.budget, "reserve_budget"):
                self.budget.reserve_budget(cost, drawing_id)
                logger.debug("Reserved $%.4f for drawing %s", cost, drawing_id)
            else:
                logger.debug(
                    "BudgetController does not support reservation, "
                    "skipping for drawing %s",
                    drawing_id,
                )
        except Exception as e:
            logger.error(
                "Budget reservation failed for drawing %s: %s",
                drawing_id,
                e,
            )
            raise RuntimeError(f"Budget reservation failed: {e}") from e


def dataclass_replace(obj, **changes):
    """Helper function to replace fields in frozen dataclass.

    Args:
        obj: Frozen dataclass instance.
        changes: Fields to update.

    Returns:
        New dataclass instance with updated fields.
    """
    return type(obj)(**{**obj.__dict__, **changes})


# Example usage and testing
if __name__ == "__main__":
    from dataclasses import dataclass
    from typing import Dict, Any

    # Mock objects for testing
    @dataclass
    class MockBudget:
        """Mock budget controller for testing."""

        remaining: float = 25.0

        def get_usage_summary(self) -> Dict[str, float]:
            """Return mock usage summary."""
            return {"remaining_budget": self.remaining}

        def estimate_stage_cost(self, stage: ProcessingStage) -> float:
            """Return mock stage cost."""
            costs = {
                ProcessingStage.OCR_VERIFICATION: 0.02,
                ProcessingStage.ENTITY_EXTRACTION: 0.08,
                ProcessingStage.SHAPE_VALIDATION: 0.04,
            }
            return costs.get(stage, 0.05)

        def reserve_budget(self, cost: float, drawing_id: str) -> None:
            """Mock budget reservation."""
            if cost > self.remaining:
                raise RuntimeError("Insufficient budget")
            self.remaining -= cost

    @dataclass
    class MockConfig:
        """Mock config for testing."""

        def get(self, key: str, default: Any = None) -> Any:
            """Get config value with default."""
            config_dict = {
                "routing": {
                    "part_number_min_confidence": 0.70,
                    "overall_threshold_high": 0.85,
                    "overall_threshold_medium": 0.70,
                    "default_sampling_rate": 0.08,
                },
                "default_sampling_rate": 0.08,
            }

            # Support nested access
            keys = key.split(".")
            value = config_dict
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k, default)
                else:
                    return default
            return value if value is not None else default

    # Create routing engine with fixed seed for reproducibility
    config = MockConfig()
    budget = MockBudget()
    engine = RoutingEngine(config, budget, random_seed=42)

    print("=" * 70)
    print("ROUTING ENGINE TEST SUITE")
    print("=" * 70)

    # Test Case 1: Missing PART_NUMBER
    print("\n=== Test Case 1: Missing PART_NUMBER ===")
    result1 = ProcessingResult(
        entities=[
            Entity(entity_type=EntityType.OEM, value="SKF", confidence=0.92),
            Entity(entity_type=EntityType.MATERIAL, value="Steel", confidence=0.88),
        ],
        detections=[Detection(class_name="bearing", confidence=0.95)],
        ocr_avg_confidence=0.90,
        text_block_count=25,
    )

    route1 = engine.determine_route(
        DrawingRecord(drawing_id="DWG-001", priority=Priority.HIGH), result1
    )
    print(f"Pipeline: {route1.pipeline.value}")
    print(f"Reason: {route1.reason}")
    print(f"Detail: {route1.reason_detail}")
    print(f"Cost: ${route1.estimated_cost:.4f}")
    print(f"Forced by critical field: {route1.forced_by_critical_field}")

    # Test Case 2: Low confidence PART_NUMBER
    print("\n=== Test Case 2: Low Confidence PART_NUMBER ===")
    result2 = ProcessingResult(
        entities=[
            Entity(
                entity_type=EntityType.PART_NUMBER,
                value="ABC-123",
                confidence=0.65,
            ),
            Entity(entity_type=EntityType.OEM, value="SKF", confidence=0.92),
        ],
        detections=[Detection(class_name="bearing", confidence=0.95)],
        ocr_avg_confidence=0.90,
        text_block_count=25,
    )

    route2 = engine.determine_route(
        DrawingRecord(drawing_id="DWG-002", priority=Priority.HIGH), result2
    )
    print(f"Pipeline: {route2.pipeline.value}")
    print(f"Reason: {route2.reason}")
    print(f"Detail: {route2.reason_detail}")
    print(f"Cost: ${route2.estimated_cost:.4f}")
    print(f"Forced by critical field: {route2.forced_by_critical_field}")

    # Test Case 3: High confidence - baseline sufficient
    print("\n=== Test Case 3: High Confidence (Baseline Sufficient) ===")
    result3 = ProcessingResult(
        entities=[
            Entity(
                entity_type=EntityType.PART_NUMBER,
                value="ABC-123",
                confidence=0.95,
            ),
            Entity(entity_type=EntityType.OEM, value="SKF", confidence=0.92),
            Entity(entity_type=EntityType.MATERIAL, value="Steel", confidence=0.88),
            Entity(entity_type=EntityType.DIMENSION, value="10mm", confidence=0.90),
        ],
        detections=[
            Detection(class_name="bearing", confidence=0.95),
            Detection(class_name="shaft", confidence=0.92),
        ],
        ocr_avg_confidence=0.92,
        text_block_count=30,
    )

    route3 = engine.determine_route(
        DrawingRecord(drawing_id="DWG-003", priority=Priority.HIGH), result3
    )
    print(f"Pipeline: {route3.pipeline.value}")
    print(f"Reason: {route3.reason}")
    print(f"Detail: {route3.reason_detail}")
    print(f"Cost: ${route3.estimated_cost:.4f}")
    if route3.confidence_scores:
        print(f"Overall score: {route3.confidence_scores.overall_weighted():.3f}")

    # Test Case 4: Budget exhausted
    print("\n=== Test Case 4: Budget Exhausted ===")
    budget_exhausted = MockBudget(remaining=0.0)
    engine_no_budget = RoutingEngine(config, budget_exhausted, random_seed=42)

    route4 = engine_no_budget.determine_route(
        DrawingRecord(drawing_id="DWG-004", priority=Priority.HIGH), result1
    )
    print(f"Pipeline: {route4.pipeline.value}")
    print(f"Reason: {route4.reason}")
    print(f"Detail: {route4.reason_detail}")
    print(f"Cost: ${route4.estimated_cost:.4f}")

    print("\n" + "=" * 70)
    print("TEST SUITE COMPLETE")
    print("=" * 70)
