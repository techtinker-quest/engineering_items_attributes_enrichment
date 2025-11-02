# orchestration/routing_engine.py
"""
Routing engine with multi-dimensional confidence assessment
and critical field (PART_NUMBER) gate
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
import logging
import numpy as np

logger = logging.getLogger(__name__)


class PipelineType(Enum):
    BASELINE_ONLY = "baseline"
    HYBRID = "hybrid"
    LLM_ENHANCED = "llm_enhanced"


class ProcessingStage(Enum):
    PDF_PROCESSING = "pdf_processing"
    OCR_VERIFICATION = "ocr_verification"
    ENTITY_EXTRACTION = "entity_extraction"
    SHAPE_VALIDATION = "shape_validation"


class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ConfidenceScores:
    """Multi-dimensional confidence assessment"""

    ocr_quality: float  # 0.0-1.0
    entity_completeness: float  # 0.0-1.0 (weighted by entity importance)
    shape_detection_quality: float  # 0.0-1.0
    critical_field_presence: float  # 0.0-1.0 (PART_NUMBER presence/confidence)
    data_consistency: float  # 0.0-1.0 (cross-validation score)

    def overall_weighted(self) -> float:
        """Calculate weighted overall score for routing decisions"""
        return (
            0.20 * self.ocr_quality
            + 0.35 * self.entity_completeness  # Highest weight
            + 0.25 * self.shape_detection_quality
            + 0.15 * self.critical_field_presence
            + 0.05 * self.data_consistency
        )


@dataclass
class ProcessingRoute:
    """Decision on which pipeline to use"""

    pipeline: PipelineType
    llm_stages: List[ProcessingStage]
    reason: str
    estimated_cost: float
    confidence_scores: Optional[ConfidenceScores] = None

    # Flag if this was forced due to critical field
    forced_by_critical_field: bool = False


@dataclass
class Entity:
    """Simplified entity for routing decision"""

    entity_type: str
    value: str
    confidence: float


@dataclass
class Detection:
    """Simplified detection for routing decision"""

    class_name: str
    confidence: float


@dataclass
class ProcessingResult:
    """Baseline processing result for routing evaluation"""

    entities: List[Entity]
    detections: List[Detection]
    ocr_avg_confidence: float
    text_block_count: int


class RoutingEngine:
    """
    Intelligent routing between baseline and LLM-enhanced pipelines
    with critical field (PART_NUMBER) gate
    """

    # Entity importance weights for completeness calculation
    ENTITY_WEIGHTS = {
        "PART_NUMBER": 0.40,  # CRITICAL
        "OEM": 0.15,
        "MATERIAL": 0.10,
        "DIMENSION": 0.15,
        "WEIGHT": 0.05,
        "THREAD_SPEC": 0.05,
        "TOLERANCE": 0.05,
        "SURFACE_FINISH": 0.05,
    }

    def __init__(self, config, budget_controller):
        self.config = config
        self.budget = budget_controller

        # Thresholds from config
        self.part_number_min_confidence = config.get("part_number_min_confidence", 0.70)
        self.overall_threshold_high = 0.85
        self.overall_threshold_medium = 0.70

    def determine_route(
        self, drawing, initial_result: ProcessingResult
    ) -> ProcessingRoute:
        """
        Evaluate processing result and determine optimal routing.

        CRITICAL FIELD GATE: If PART_NUMBER missing or low confidence,
        force LLM enhancement (budget permitting).
        """

        # === STEP 1: CRITICAL FIELD GATE (PART_NUMBER) ===
        part_number_check = self._check_part_number(initial_result.entities)

        if not part_number_check["exists"]:
            # PART_NUMBER completely missing
            logger.warning(
                f"Drawing {drawing.drawing_id}: PART_NUMBER not found - "
                f"triggering LLM enhancement"
            )

            if self._budget_available() and drawing.priority != Priority.LOW:
                return ProcessingRoute(
                    pipeline=PipelineType.LLM_ENHANCED,
                    llm_stages=[ProcessingStage.ENTITY_EXTRACTION],
                    reason="CRITICAL: Missing PART_NUMBER → force LLM enhancement",
                    estimated_cost=self._estimate_cost(
                        [ProcessingStage.ENTITY_EXTRACTION]
                    ),
                    forced_by_critical_field=True,
                )
            else:
                # Budget exhausted or low priority - flag for manual review
                return ProcessingRoute(
                    pipeline=PipelineType.BASELINE_ONLY,
                    llm_stages=[],
                    reason="Missing PART_NUMBER but budget exhausted → flag for review",
                    estimated_cost=0.0,
                    forced_by_critical_field=True,  # Still note it was checked
                )

        elif part_number_check["confidence"] < self.part_number_min_confidence:
            # PART_NUMBER exists but low confidence
            logger.warning(
                f"Drawing {drawing.drawing_id}: PART_NUMBER confidence "
                f"({part_number_check['confidence']:.2f}) below threshold "
                f"({self.part_number_min_confidence}) - triggering LLM verification"
            )

            if self._budget_available():
                return ProcessingRoute(
                    pipeline=PipelineType.HYBRID,
                    llm_stages=[ProcessingStage.ENTITY_EXTRACTION],
                    reason=f"Low PART_NUMBER confidence ({part_number_check['confidence']:.2f}) → LLM verify",
                    estimated_cost=self._estimate_cost(
                        [ProcessingStage.ENTITY_EXTRACTION]
                    ),
                    forced_by_critical_field=True,
                )

        # === STEP 2: CALCULATE MULTI-DIMENSIONAL CONFIDENCE ===
        confidence_scores = ConfidenceScores(
            ocr_quality=self._assess_ocr_quality(initial_result),
            entity_completeness=self._assess_entity_completeness(
                initial_result.entities
            ),
            shape_detection_quality=self._assess_shape_quality(
                initial_result.detections
            ),
            critical_field_presence=part_number_check[
                "confidence"
            ],  # Use PN confidence
            data_consistency=self._validate_data_consistency(initial_result),
        )

        overall_score = confidence_scores.overall_weighted()

        logger.info(
            f"Drawing {drawing.drawing_id} confidence breakdown:\n"
            f"  OCR: {confidence_scores.ocr_quality:.2f}\n"
            f"  Entity: {confidence_scores.entity_completeness:.2f}\n"
            f"  Shape: {confidence_scores.shape_detection_quality:.2f}\n"
            f"  Critical Fields: {confidence_scores.critical_field_presence:.2f}\n"
            f"  Consistency: {confidence_scores.data_consistency:.2f}\n"
            f"  Overall: {overall_score:.2f}"
        )

        # === STEP 3: APPLY ROUTING RULES ===
        route = self._apply_routing_rules(confidence_scores, overall_score, drawing)
        route.confidence_scores = confidence_scores

        return route

    def _check_part_number(self, entities: List[Entity]) -> dict:
        """
        Check PART_NUMBER presence and confidence.

        Returns:
            {
                'exists': bool,
                'confidence': float (0.0 if not found),
                'value': str or None
            }
        """
        part_numbers = [e for e in entities if e.entity_type == "PART_NUMBER"]

        if not part_numbers:
            return {"exists": False, "confidence": 0.0, "value": None}

        # If multiple part numbers, use highest confidence
        best_pn = max(part_numbers, key=lambda e: e.confidence)

        return {
            "exists": True,
            "confidence": best_pn.confidence,
            "value": best_pn.value,
        }

    def _assess_entity_completeness(self, entities: List[Entity]) -> float:
        """
        Entity-specific confidence assessment.

        Returns confidence from 0.0-1.0 based on:
        - Presence of critical entities (PART_NUMBER: weight 0.4)
        - Presence of expected entities (weighted)
        - Entity confidence scores
        """
        score = 0.0
        entity_types_found = {e.entity_type: [] for e in entities}
        for e in entities:
            entity_types_found[e.entity_type].append(e)

        # Check presence and confidence of each entity type
        for entity_type, weight in self.ENTITY_WEIGHTS.items():
            if entity_type in entity_types_found and entity_types_found[entity_type]:
                # Average confidence of entities of this type
                avg_confidence = np.mean(
                    [e.confidence for e in entity_types_found[entity_type]]
                )
                score += weight * avg_confidence
            else:
                # Penalty for missing entity types
                if entity_type == "PART_NUMBER":
                    score += 0  # Critical miss: 0 points
                else:
                    score += weight * 0.3  # Partial credit for less critical

        return score

    def _assess_ocr_quality(self, result: ProcessingResult) -> float:
        """Assess OCR quality based on average confidence and text block count"""
        if result.text_block_count == 0:
            return 0.0

        # Weight by text block count (more text = more reliable)
        text_score = min(result.text_block_count / 20, 1.0)  # Normalize to 20+ blocks

        return 0.7 * result.ocr_avg_confidence + 0.3 * text_score

    def _assess_shape_quality(self, detections: List[Detection]) -> float:
        """Assess shape detection quality"""
        if not detections:
            return 0.0

        avg_conf = np.mean([d.confidence for d in detections])
        count_score = min(len(detections) / 5, 1.0)  # Normalize to 5+ shapes

        return 0.8 * avg_conf + 0.2 * count_score

    def _validate_data_consistency(self, result: ProcessingResult) -> float:
        """
        Cross-validate data consistency.
        This is a simplified version - full implementation would check:
        - Entity-shape associations
        - Dimension compatibility
        - No orphaned entities
        """
        # Placeholder: return high confidence if we have both text and shapes
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
        self, scores: ConfidenceScores, overall_score: float, drawing
    ) -> ProcessingRoute:
        """
        Decision tree based on multiple confidence dimensions.
        """

        # === HIGH CONFIDENCE → Baseline sufficient ===
        if (
            overall_score >= self.overall_threshold_high
            and scores.critical_field_presence >= 0.9
        ):
            return ProcessingRoute(
                pipeline=PipelineType.BASELINE_ONLY,
                llm_stages=[],
                reason=f"High confidence across all dimensions: {overall_score:.2f}",
                estimated_cost=0.0,
            )

        # === ENTITY-SPECIFIC ROUTING ===
        if scores.entity_completeness < 0.75:
            if self._budget_available() and self._sampling_criteria_met(drawing):
                return ProcessingRoute(
                    pipeline=PipelineType.HYBRID,
                    llm_stages=[ProcessingStage.ENTITY_EXTRACTION],
                    reason=f"Low entity completeness: {scores.entity_completeness:.2f}",
                    estimated_cost=self._estimate_cost(
                        [ProcessingStage.ENTITY_EXTRACTION]
                    ),
                )

        # === OCR-SPECIFIC ROUTING ===
        if scores.ocr_quality < 0.85 and scores.entity_completeness < 0.80:
            if self._budget_available():
                return ProcessingRoute(
                    pipeline=PipelineType.HYBRID,
                    llm_stages=[
                        ProcessingStage.OCR_VERIFICATION,
                        ProcessingStage.ENTITY_EXTRACTION,
                    ],
                    reason="Poor OCR quality impacting entity extraction",
                    estimated_cost=self._estimate_cost(
                        [
                            ProcessingStage.OCR_VERIFICATION,
                            ProcessingStage.ENTITY_EXTRACTION,
                        ]
                    ),
                )

        # === MEDIUM CONFIDENCE → Selective LLM enhancement ===
        if self.overall_threshold_medium <= overall_score < self.overall_threshold_high:
            if self._budget_available() and self._sampling_criteria_met(drawing):
                # Target LLM at weakest dimension
                weakest_stage = self._identify_weakest_stage(scores)
                return ProcessingRoute(
                    pipeline=PipelineType.HYBRID,
                    llm_stages=[weakest_stage],
                    reason=f"Medium confidence: {overall_score:.2f}, "
                    f"weak dimension: {weakest_stage.value}",
                    estimated_cost=self._estimate_cost([weakest_stage]),
                )

        # === FALLBACK → Baseline (budget exhausted or low priority) ===
        return ProcessingRoute(
            pipeline=PipelineType.BASELINE_ONLY,
            llm_stages=[],
            reason="Budget exhausted or sampling criteria not met",
            estimated_cost=0.0,
        )

    def _identify_weakest_stage(self, scores: ConfidenceScores) -> ProcessingStage:
        """Identify which stage would benefit most from LLM enhancement"""
        dimension_to_stage = {
            "ocr_quality": ProcessingStage.OCR_VERIFICATION,
            "entity_completeness": ProcessingStage.ENTITY_EXTRACTION,
            "shape_detection_quality": ProcessingStage.SHAPE_VALIDATION,
        }

        weakest = min(
            [
                ("ocr_quality", scores.ocr_quality),
                ("entity_completeness", scores.entity_completeness),
                ("shape_detection_quality", scores.shape_detection_quality),
            ],
            key=lambda x: x[1],
        )

        return dimension_to_stage[weakest[0]]

    def _budget_available(self) -> bool:
        """Check if budget allows LLM calls"""
        summary = self.budget.get_usage_summary()
        return summary["remaining_budget"] > 0

    def _sampling_criteria_met(self, drawing) -> bool:
        """Check if drawing meets sampling criteria"""
        # Implement sampling strategy (random, stratified, etc.)
        # For now, simple random sampling
        import random

        sampling_rate = self.config.get("default_sampling_rate", 0.08)
        return random.random() < sampling_rate

    def _estimate_cost(self, stages: List[ProcessingStage]) -> float:
        """Estimate cost for given LLM stages"""
        # Simplified estimation - real implementation would use
        # BudgetController.pre_call_check()
        cost_per_stage = {
            ProcessingStage.OCR_VERIFICATION: 0.02,
            ProcessingStage.ENTITY_EXTRACTION: 0.08,
            ProcessingStage.SHAPE_VALIDATION: 0.04,
        }
        return sum(cost_per_stage.get(stage, 0.05) for stage in stages)


# Example usage:
if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class MockDrawing:
        drawing_id: str
        priority: Priority

    @dataclass
    class MockBudget:
        def get_usage_summary(self):
            return {"remaining_budget": 25.0}

    # Create routing engine
    config = {"part_number_min_confidence": 0.70, "default_sampling_rate": 0.08}
    engine = RoutingEngine(config, MockBudget())

    # Test Case 1: Missing PART_NUMBER
    print("=== Test Case 1: Missing PART_NUMBER ===")
    result = ProcessingResult(
        entities=[Entity("OEM", "SKF", 0.92), Entity("MATERIAL", "Steel", 0.88)],
        detections=[Detection("bearing", 0.95)],
        ocr_avg_confidence=0.90,
        text_block_count=25,
    )

    route = engine.determine_route(MockDrawing("DWG-001", Priority.HIGH), result)
    print(f"Pipeline: {route.pipeline.value}")
    print(f"Reason: {route.reason}")
    print(f"Forced by critical field: {route.forced_by_critical_field}\n")

    # Test Case 2: Low confidence PART_NUMBER
    print("=== Test Case 2: Low Confidence PART_NUMBER ===")
    result = ProcessingResult(
        entities=[
            Entity("PART_NUMBER", "ABC-123", 0.65),  # Below threshold
            Entity("OEM", "SKF", 0.92),
        ],
        detections=[Detection("bearing", 0.95)],
        ocr_avg_confidence=0.90,
        text_block_count=25,
    )

    route = engine.determine_route(MockDrawing("DWG-002", Priority.HIGH), result)
    print(f"Pipeline: {route.pipeline.value}")
    print(f"Reason: {route.reason}")
    print(f"Forced by critical field: {route.forced_by_critical_field}\n")

    # Test Case 3: High confidence - baseline sufficient
    print("=== Test Case 3: High Confidence ===")
    result = ProcessingResult(
        entities=[
            Entity("PART_NUMBER", "ABC-123", 0.95),
            Entity("OEM", "SKF", 0.92),
            Entity("MATERIAL", "Steel", 0.88),
        ],
        detections=[Detection("bearing", 0.95)],
        ocr_avg_confidence=0.92,
        text_block_count=30,
    )

    route = engine.determine_route(MockDrawing("DWG-003", Priority.HIGH), result)
    print(f"Pipeline: {route.pipeline.value}")
    print(f"Reason: {route.reason}")
    if route.confidence_scores:
        print(f"Overall score: {route.confidence_scores.overall_weighted():.2f}")
