## Classes inventory — engineering_items_attributes_enrichment

Generated: 2025-11-02

This file lists all Python class definitions found in the repository (under `src/drawing_intelligence`) and the methods defined on each class. Enums and dataclasses are included.

If you want this regenerated or exported in another format (JSON, CSV), tell me and I can add a script to do it.

---

### `src/drawing_intelligence/utils/config_loader.py`

- class SystemConfig
  - __init__(self, **config_dict)
    - Initializes configuration holder. Fields used: `paths`, `database`, `shape_detection`, `entity_extraction`, `batch_processing`, `logging`.

- class Config
  - load(config_path: str = "config/system_config.yaml") -> SystemConfig (staticmethod)
    - Loads YAML config, resolves repository-relative paths (data, models, output, temp, DB, dictionaries, logging).
  - validate(config: SystemConfig) -> list (staticmethod)
    - Validates that critical file/directory paths exist; returns list of error messages (empty if OK).

### `src/drawing_intelligence/orchestration/workflow_state.py`

- class WorkflowState (dataclass)
  - update_stage(self, stage: str) -> None
    - Set `current_stage` and update `timestamp` to now.
  - save_checkpoint(self, data: Dict[str, any]) -> None
    - Merge/update `checkpoint_data` with provided data.

### `src/drawing_intelligence/llm/budget_controller.py`

- class UseCaseType (Enum)
  - Enum members: DRAWING_ASSESSMENT, OCR_VERIFICATION, ENTITY_EXTRACTION, SHAPE_VALIDATION, COMPLEX_REASONING

- class UseCaseModelConfig (dataclass)
  - Fields: use_case, preferred_model, tier_1_fallback, tier_0_fallback, max_tokens, temperature, requires_vision

- class BudgetController
  - __init__(self, daily_budget_usd, per_drawing_limit_usd, alert_threshold_pct: float = 0.80, db_manager=None)
    - Tracks daily budget, per-drawing limit, tier state and optional DB manager.
  - get_model_for_use_case(self, use_case: UseCaseType, override_model: Optional[str] = None) -> Tuple[ModelSpec, str]
    - Returns a ModelSpec to use for the given use case and a textual reason. Considers current tier and vision requirements.
  - pre_call_check(self, estimated_input_tokens: int, estimated_output_tokens: int, image_count: int, use_case: UseCaseType, drawing_id: str) -> Tuple[bool, str, Optional[ModelSpec]]
    ## classes.md — full signatures and docstrings

    Generated: 2025-11-02

    This document lists every Python class found under `src/drawing_intelligence` with expanded details: fields (for dataclasses), full method signatures, and method docstrings (where present in source).

    If you want this exported as JSON/CSV or regenerated automatically, tell me and I'll add a small script to the repo.

    ---

    ### src/drawing_intelligence/utils/config_loader.py

    class SystemConfig
      Signature: __init__(self, **config_dict)
      Docstring: (none on class)
      Notes:
        - Initializes configuration holder.
        - Fields used by code: `paths`, `database`, `shape_detection`, `entity_extraction`, `batch_processing`, `logging`.

    class Config
      Signature: load(config_path: str = "config/system_config.yaml") -> SystemConfig  (staticmethod)
      Docstring:
        - Resolves repository-relative paths and creates a SystemConfig instance.
        - Walks and resolves entries:
          - `paths.data_dir`, `paths.models_dir`, `paths.output_dir`, `paths.temp_dir`
          - `database.path`
          - `entity_extraction.oem_dictionary_path`
          - `shape_detection.model_path`
          - `batch_processing.checkpointing.batch_checkpoint_dir`
          - `logging.log_dir`

      Signature: validate(config: SystemConfig) -> list  (staticmethod)
      Docstring:
        """Validate configuration and return list of errors"""
      Description:
        - Iterates a set of critical path keys and tests os.path.exists(path_value).
        - Returns a list of error strings for missing paths.

    ---

    ### src/drawing_intelligence/orchestration/workflow_state.py

    class WorkflowState (dataclass)
      Fields:
        - drawing_id: str
        - current_stage: str
        - metadata: Dict[str, str]
        - checkpoint_data: Dict[str, any]
        - timestamp: datetime
        - confidence_scores: Optional[Dict[str, float]] = None

      Signature: update_stage(self, stage: str) -> None
      Docstring: """Update the current processing stage."""
      Behavior:
        - Sets self.current_stage = stage
        - Updates self.timestamp = datetime.now()

      Signature: save_checkpoint(self, data: Dict[str, any]) -> None
      Docstring: """Save checkpoint data for the current stage."""
      Behavior:
        - Merges the provided dict into self.checkpoint_data via self.checkpoint_data.update(data)

    ---

    ### src/drawing_intelligence/llm/budget_controller.py

    class UseCaseType (Enum)
      Members:
        - DRAWING_ASSESSMENT = "drawing_assessment"
        - OCR_VERIFICATION = "ocr_verification"
        - ENTITY_EXTRACTION = "entity_extraction"
        - SHAPE_VALIDATION = "shape_validation"
        - COMPLEX_REASONING = "complex_reasoning"

    class UseCaseModelConfig (dataclass)
      Fields:
        - use_case: UseCaseType
        - preferred_model: str
        - tier_1_fallback: str
        - tier_0_fallback: str
        - max_tokens: int
        - temperature: float = 0.0
        - requires_vision: bool = False

    class BudgetController
      Signature: __init__(
        self,
        daily_budget_usd: float,
        per_drawing_limit_usd: float,
        alert_threshold_pct: float = 0.80,
        db_manager=None,
      )
      Docstring: (none on __init__)
      Behavior:
        - Sets daily budget, per-drawing limit, and alert threshold.
        - Initializes current_tier to ModelTier.TIER_2_PREMIUM and other internal trackers.

      Signature: get_model_for_use_case(self, use_case: UseCaseType, override_model: Optional[str] = None) -> Tuple[ModelSpec, str]
      Docstring:
        """
        Get the appropriate model for a use case, considering budget constraints.

        Returns:
            Tuple of (ModelSpec, reason)
        """
      Behavior:
        - If override_model provided, returns ModelRegistry.get_model(override_model) and reason.
        - Otherwise selects based on USE_CASE_CONFIGS and self.current_tier.
        - Validates vision requirement: raises ValueError if required but model doesn't support vision.

      Signature: pre_call_check(
        self,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
        image_count: int,
        use_case: UseCaseType,
        drawing_id: str,
      ) -> Tuple[bool, str, Optional[ModelSpec]]
      Docstring:
        """
        Pre-flight check before making LLM call.

        Returns:
            Tuple of (allowed, reason, model_to_use)
            - allowed: Whether the call should proceed
            - reason: Explanation of the decision
            - model_to_use: The model to use (None if call not allowed)
        """
      Behavior:
        - Gets current spend and drawing spend via internal helpers.
        - Selects a model for the use case, estimates cost, checks per-drawing limit.
        - Steps down tier if alert threshold exceeded.
        - Returns False with reason if daily budget would be exceeded.

      Signature: track_call(
        self,
        provider: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        image_count: int,
        drawing_id: str,
        use_case: UseCaseType,
      ) -> float
      Docstring:
        """
        Track an LLM call and return actual cost.
        """
      Behavior:
        - Uses ModelRegistry.get_model(model_id) to compute actual cost via ModelSpec.calculate_cost.
        - If self.db exists, stores usage record with timestamp.
        - Updates current_drawing_spend and logs the call.
        - Returns actual_cost.

      Signature: _step_down_tier(self)
      Docstring: """Step down to next cheaper tier"""
      Behavior:
        - Moves current_tier down from TIER_2_PREMIUM -> TIER_1_BALANCED -> TIER_0_CHEAP and flags tier_stepped_down.

      Signature: _get_daily_spend(self) -> float
      Docstring: """Get today's total LLM spend"""
      Behavior: Queries db if present; otherwise returns 0.0.

      Signature: _get_drawing_spend(self, drawing_id: str) -> float
      Docstring: """Get total spend for a specific drawing"""
      Behavior: Query DB or return self.current_drawing_spend.

      Signature: _get_budget_used_pct(self) -> float
      Docstring: """Get percentage of daily budget used"""

      Signature: reset_drawing_tracker(self)
      Docstring: """Reset per-drawing spend tracker (call when starting new drawing)"""

      Signature: get_usage_summary(self) -> Dict
      Docstring: """Get current usage summary"""
      Returns:
        - dict with daily_budget, daily_spend, remaining_budget, budget_used_pct, current_tier, tier_stepped_down, alert_threshold, alert_triggered

    ---

    ### src/drawing_intelligence/orchestration/checkpoint_manager.py

    class ProcessingStage (Enum)
      Members:
        - PDF_PROCESSING
        - IMAGE_PREPROCESSING
        - OCR_EXTRACTION
        - ENTITY_EXTRACTION
        - SHAPE_DETECTION
        - DATA_ASSOCIATION
        - QUALITY_SCORING
        - COMPLETE

    class IntraDrawingCheckpoint (dataclass)
      Fields:
        - drawing_id: str
        - source_file: str
        - current_stage: ProcessingStage
        - completed_stages: List[ProcessingStage]
        - intermediate_results: Dict[str, Any]
        - timestamp: str
        - checkpoint_id: str

      Signature: to_dict(self) -> Dict
      Docstring: (none)
      Behavior:
        - Returns asdict(self) with enums converted to their string values for JSON serialization.

      Signature: from_dict(cls, data: Dict) -> "IntraDrawingCheckpoint"  (classmethod)
      Docstring: (none)
      Behavior:
        - Converts string stage values back into ProcessingStage enums and returns the dataclass.

    class BatchCheckpoint (dataclass)
      Fields:
        - batch_id: str
        - total_drawings: int
        - completed_count: int
        - failed_count: int
        - completed_drawing_ids: List[str]
        - failed_drawing_ids: List[str]
        - pending_drawing_ids: List[str]
        - current_drawing_id: Optional[str]
        - timestamp: str
        - checkpoint_id: str
        - total_llm_cost: float
        - average_processing_time_seconds: float

      Signature: to_dict(self) -> Dict
      Signature: from_dict(cls, data: Dict) -> "BatchCheckpoint"  (classmethod)

    class CheckpointManager
      Signature: __init__(self, checkpoint_dir: str = "/tmp/checkpoints")
      Docstring: (none)
      Behavior:
        - Creates directories for intra_drawing and batch checkpoints under checkpoint_dir.

      Signature: save_intra_drawing_checkpoint(
        self,
        drawing_id: str,
        source_file: str,
        current_stage: ProcessingStage,
        completed_stages: List[ProcessingStage],
        intermediate_results: Dict[str, Any],
      ) -> str
      Docstring:
        """
        Save checkpoint after completing a stage within a drawing.

        Returns:
            checkpoint_id
        """
      Behavior:
        - Builds an `IntraDrawingCheckpoint`, writes to JSON file, returns checkpoint_id.

      Signature: load_intra_drawing_checkpoint(self, drawing_id: str) -> Optional[IntraDrawingCheckpoint]
      Docstring:
        """
        Load the most recent checkpoint for a drawing.
        Returns None if no checkpoint exists.
        """

      Signature: delete_intra_drawing_checkpoints(self, drawing_id: str)
      Docstring: """Delete all checkpoints for a drawing (after successful completion)"""

      Signature: save_batch_checkpoint(
        self,
        batch_id: str,
        total_drawings: int,
        completed_drawing_ids: List[str],
        failed_drawing_ids: List[str],
        pending_drawing_ids: List[str],
        current_drawing_id: Optional[str],
        total_llm_cost: float,
        average_processing_time: float,
      ) -> str
      Docstring:
        """
        Save batch processing state every N drawings.

        Returns:
            checkpoint_id
        """

      Signature: load_batch_checkpoint(self, batch_id: str) -> Optional[BatchCheckpoint]
      Signature: delete_batch_checkpoints(self, batch_id: str)
      Signature: list_available_checkpoints(self) -> Dict[str, List[str]]
      Docstring: """List all available checkpoints"""
      Returns:
        - {'intra_drawing': [...], 'batch': [...]} listing checkpoint filename stems

    ---

    ### src/drawing_intelligence/orchestration/routing_engine.py

    class PipelineType (Enum)
      Members: BASELINE_ONLY, HYBRID, LLM_ENHANCED

    class ProcessingStage (Enum)
      Members used in routing: PDF_PROCESSING, OCR_VERIFICATION, ENTITY_EXTRACTION, SHAPE_VALIDATION

    class Priority (Enum)
      Members: LOW, MEDIUM, HIGH, CRITICAL

    class ConfidenceScores (dataclass)
      Fields:
        - ocr_quality: float
        - entity_completeness: float
        - shape_detection_quality: float
        - critical_field_presence: float
        - data_consistency: float

      Signature: overall_weighted(self) -> float
      Docstring: """Calculate weighted overall score for routing decisions"""
      Behavior:
        - Returns weighted sum: 0.20*ocr + 0.35*entity + 0.25*shape + 0.15*critical + 0.05*consistency

    class ProcessingRoute (dataclass)
      Fields: pipeline, llm_stages, reason, estimated_cost, confidence_scores (Optional[ConfidenceScores]), forced_by_critical_field (bool)

    class Entity (dataclass)
      Fields: entity_type, value, confidence

    class Detection (dataclass)
      Fields: class_name, confidence

    class ProcessingResult (dataclass)
      Fields: entities: List[Entity], detections: List[Detection], ocr_avg_confidence: float, text_block_count: int

    class RoutingEngine
      Signature: __init__(self, config, budget_controller)
      Docstring: (none on class)
      Behavior:
        - Saves config and budget controller reference.
        - Sets thresholds: part_number_min_confidence, overall_threshold_high, overall_threshold_medium.

      Signature: determine_route(self, drawing, initial_result: ProcessingResult) -> ProcessingRoute
      Docstring:
        """
        Evaluate processing result and determine optimal routing.

        CRITICAL FIELD GATE: If PART_NUMBER missing or low confidence,
        force LLM enhancement (budget permitting).
        """
      Behavior:
        - Uses _check_part_number; forces LLM paths if missing/low PN and budget allows.
        - Computes ConfidenceScores with helper methods and calls _apply_routing_rules.

      Signature: _check_part_number(self, entities: List[Entity]) -> dict
      Docstring:
        """
        Check PART_NUMBER presence and confidence.

        Returns:
            {
                'exists': bool,
                'confidence': float (0.0 if not found),
                'value': str or None
            }
        """

      Signature: _assess_entity_completeness(self, entities: List[Entity]) -> float
      Docstring:
        """
        Entity-specific confidence assessment.

        Returns confidence from 0.0-1.0 based on:
        - Presence of critical entities (PART_NUMBER: weight 0.4)
        - Presence of expected entities (weighted)
        - Entity confidence scores
        """
      Behavior:
        - Computes weighted score using ENTITY_WEIGHTS; gives partial credit for non-critical missing types.

      Signature: _assess_ocr_quality(self, result: ProcessingResult) -> float
      Docstring: """Assess OCR quality based on average confidence and text block count"""

      Signature: _assess_shape_quality(self, detections: List[Detection]) -> float
      Docstring: """Assess shape detection quality"""

      Signature: _validate_data_consistency(self, result: ProcessingResult) -> float
      Docstring:
        """
        Cross-validate data consistency.

        This is a simplified version - full implementation would check:
        - Entity-shape associations
        - Dimension compatibility
        - No orphaned entities
        """

      Signature: _apply_routing_rules(self, scores: ConfidenceScores, overall_score: float, drawing) -> ProcessingRoute
      Docstring: """Decision tree based on multiple confidence dimensions."""
      Behavior:
        - High-confidence -> baseline
        - Low entity completeness or OCR -> hybrid if budget allows
        - Medium confidence -> targeted LLM on weakest stage
        - Fallback -> baseline

      Signature: _identify_weakest_stage(self, scores: ConfidenceScores) -> ProcessingStage
      Docstring: """Identify which stage would benefit most from LLM enhancement"""

      Signature: _budget_available(self) -> bool
      Docstring: """Check if budget allows LLM calls"""
      Behavior: returns summary['remaining_budget'] > 0 from budget controller

      Signature: _sampling_criteria_met(self, drawing) -> bool
      Docstring: """Check if drawing meets sampling criteria"""
      Note: current implementation uses random sampling with config default_sampling_rate

      Signature: _estimate_cost(self, stages: List[ProcessingStage]) -> float
      Docstring: """Estimate cost for given LLM stages"""
      Behavior: uses a simple per-stage cost mapping and sums costs

      Test/example-only classes in module `__main__` (used by tests / examples):
        - MockDrawing (dataclass): fields drawing_id, priority
        - MockBudget (dataclass): get_usage_summary(self) -> dict

    ---

    ### src/drawing_intelligence/models/model_registry.py

    class ModelProvider (Enum)
      Members: OPENAI, ANTHROPIC, GOOGLE

    class ModelTier (Enum)
      Members: TIER_0_CHEAP, TIER_1_BALANCED, TIER_2_PREMIUM

    class ModelSpec (dataclass)
      Fields:
        - provider: ModelProvider
        - model_id: str
        - canonical_name: str
        - tier: ModelTier
        - supports_vision: bool
        - max_tokens: int
        - input_cost_per_1m: float
        - output_cost_per_1m: float
        - image_cost: Optional[float] = None

      Signature: calculate_cost(self, input_tokens: int, output_tokens: int, image_count: int = 0) -> float
      Docstring: """Calculate total cost for this model"""
      Behavior:
        - Computes input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
        - Computes output_cost similarly
        - Handles image costs: explicit image_cost or Anthropic-specific approximation

    class ModelRegistry
      Signature: get_model(cls, name: str) -> ModelSpec  (classmethod)
      Docstring: """Get model by canonical name or full model_id"""
      Behavior:
        - Returns a ModelSpec if name matches canonical key or model.model_id
        - Raises ValueError if unknown

      Signature: get_models_by_tier(cls, tier: ModelTier, supports_vision: Optional[bool] = None) -> list[ModelSpec]  (classmethod)
      Docstring: """Get all models in a specific tier, optionally filtered by vision support"""

      Signature: validate_model_exists(cls, name: str) -> bool  (classmethod)
      Docstring: """Check if a model exists in the registry"""

    ---

    Notes
    - This file was regenerated to include full method signatures and the method docstrings (or method descriptions when docstrings were not present) pulled from the source files.
    - If you want the docstrings copied verbatim (including triple quotes and full multi-line content) into the file, or prefer a machine-readable JSON export, I can produce that next.

    How to re-run locally
    - Use a small Python script that parses AST nodes (ast.ClassDef, ast.FunctionDef) and extracts signatures and docstrings. I can add that script to the repo if you want.

    ---
