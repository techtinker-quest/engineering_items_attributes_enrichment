"""
Pipeline Orchestrator Module

Manages end-to-end workflow execution, error handling, and checkpointing.
"""

import logging
from pathlib import Path
from typing import List, Optional, Any, Dict, Union, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ..utils.config_loader import Config
from ..database.database_manager import DatabaseManager
from .checkpoint_manager import CheckpointManager
from .routing_engine import RoutingEngine, ProcessingRoute, Drawing
from .workflow_state import WorkflowState
from ..models.data_structures import (
    ProcessingStage,
    ProcessingResult,
    PipelineType,
    Priority,
    EntityType,
    FlagType,
    Severity,
    OCRResult,
    DetectionResult,
)

# Import processing modules
from ..processing.pdf_processor import PDFProcessor
from ..processing.image_preprocessor import ImagePreprocessor
from ..processing.ocr_pipeline import OCRPipeline
from ..processing.entity_extractor import EntityExtractor
from ..processing.shape_detector import ShapeDetector
from ..processing.data_associator import DataAssociator
from ..processing.data_validator import DataValidator
from ..processing.hierarchy_builder import HierarchyBuilder

# Import quality modules
from ..quality.quality_scorer import QualityScorer

# Import LLM gateway
from ..llm.llm_gateway import LLMGateway
from ..llm.budget_controller import BudgetController

# Import error handlers
from ..utils.error_handlers import (
    DrawingProcessingError,
    PDFProcessingError,
    OCRError,
    ShapeDetectionError,
    BudgetExceededException,
    DatabaseError,
)


logger: logging.Logger = logging.getLogger(__name__)


# Configuration Constants
class OrchestratorConfig:
    """Configuration constants for pipeline orchestrator."""

    # Retry configuration
    MAX_RETRIES: int = 3
    RETRY_BACKOFF_BASE: float = 2.0
    RETRY_INITIAL_DELAY: float = 1.0

    # LLM thresholds
    LLM_OCR_CONFIDENCE_THRESHOLD: float = 0.85

    # Batch processing
    DEFAULT_PARALLEL_WORKERS: int = 4
    DEFAULT_CHECKPOINT_FREQUENCY: int = 10


@dataclass
class StageResult:
    """Result from a single processing stage.

    Attributes:
        stage: The processing stage that was executed.
        success: Whether the stage completed successfully.
        data: Output data from the stage (type varies by stage).
        confidence: Confidence score for this stage (0.0-1.0).
        duration_seconds: Wall-clock execution time in seconds.
        error_message: Error details if the stage failed.
    """

    stage: ProcessingStage
    success: bool
    data: Any
    confidence: float
    duration_seconds: float
    error_message: Optional[str] = None


@dataclass
class InitialRoutingResult:
    """Initial processing results for routing decisions.

    Attributes:
        entities: List of extracted entities.
        detections: List of detected shapes.
        ocr_avg_confidence: Average OCR confidence score.
        text_block_count: Number of text blocks extracted.
    """

    entities: List[Any]
    detections: List[Any]
    ocr_avg_confidence: float
    text_block_count: int


@dataclass
class DrawingResultSummary:
    """Lightweight summary of drawing processing results.

    Used to reduce memory footprint in batch processing.

    Attributes:
        drawing_id: Unique drawing identifier.
        source_file: Path to source PDF.
        status: Processing status (complete/failed).
        overall_confidence: Overall confidence score.
        needs_review: Whether human review is required.
        llm_cost: LLM API cost for this drawing.
        processing_time: Total processing time in seconds.
        error_message: Error message if failed.
    """

    drawing_id: str
    source_file: str
    status: str
    overall_confidence: float
    needs_review: bool
    llm_cost: float
    processing_time: float
    error_message: Optional[str] = None


@dataclass
class BatchResult:
    """Result from batch processing.

    Attributes:
        batch_id: Unique identifier for this batch.
        total_drawings: Total number of drawings in the batch.
        successful: Number of successfully processed drawings.
        failed: Number of failed drawings.
        needs_review: Number of drawings flagged for human review.
        success_rate: Percentage of successful drawings (0.0-1.0).
        review_rate: Percentage requiring review (0.0-1.0).
        total_llm_cost: Total cost of LLM API calls in USD.
        average_processing_time: Average time per drawing in seconds.
        drawing_summaries: List of lightweight result summaries.
    """

    batch_id: str
    total_drawings: int
    successful: int
    failed: int
    needs_review: int
    success_rate: float
    review_rate: float
    total_llm_cost: float
    average_processing_time: float
    drawing_summaries: List[DrawingResultSummary]


class PipelineOrchestrator:
    """Manages end-to-end workflow execution.

    Coordinates all processing stages, error handling, and checkpointing
    for both single drawing and batch processing operations.

    Attributes:
        config: System configuration object.
        db: Database manager for persistence.
        checkpoint_manager: Manages processing checkpoints.
        routing_engine: Determines pipeline routing decisions.
    """

    def __init__(
        self,
        config: Config,
        db: DatabaseManager,
        checkpoint_manager: CheckpointManager,
        routing_engine: RoutingEngine,
        pdf_processor: Optional[PDFProcessor] = None,
        image_preprocessor: Optional[ImagePreprocessor] = None,
        ocr_pipeline: Optional[OCRPipeline] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        shape_detector: Optional[ShapeDetector] = None,
        data_associator: Optional[DataAssociator] = None,
        data_validator: Optional[DataValidator] = None,
        hierarchy_builder: Optional[HierarchyBuilder] = None,
        quality_scorer: Optional[QualityScorer] = None,
        llm_gateway: Optional[LLMGateway] = None,
    ) -> None:
        """Initialize pipeline orchestrator with dependency injection.

        Args:
            config: System configuration object.
            db: Database manager instance.
            checkpoint_manager: Checkpoint manager instance.
            routing_engine: Routing engine instance.
            pdf_processor: Optional PDF processor (created if None).
            image_preprocessor: Optional image preprocessor.
            ocr_pipeline: Optional OCR pipeline.
            entity_extractor: Optional entity extractor.
            shape_detector: Optional shape detector.
            data_associator: Optional data associator.
            data_validator: Optional data validator.
            hierarchy_builder: Optional hierarchy builder.
            quality_scorer: Optional quality scorer.
            llm_gateway: Optional LLM gateway.
        """
        self.config = config
        self.db = db
        self.checkpoint_manager = checkpoint_manager
        self.routing_engine = routing_engine

        # Initialize or inject processing components
        self.pdf_processor = pdf_processor or PDFProcessor(
            self.config.get("pdf_processing", {})
        )
        self.image_preprocessor = image_preprocessor or ImagePreprocessor(
            self.config.get("image_preprocessing", {})
        )
        self.ocr_pipeline = ocr_pipeline or OCRPipeline(self.config.get("ocr", {}))
        self.shape_detector = shape_detector or ShapeDetector(
            self.config.get("shape_detection", {})
        )
        self.data_associator = data_associator or DataAssociator(
            self.config.get("data_association", {})
        )
        self.data_validator = data_validator or DataValidator(
            self.config.get("validation", {})
        )
        self.hierarchy_builder = hierarchy_builder or HierarchyBuilder()
        self.quality_scorer = quality_scorer or QualityScorer(
            self.config.get("quality_scoring", {})
        )

        # LLM Gateway (conditional)
        if llm_gateway:
            self.llm_gateway = llm_gateway
        else:
            llm_config = self.config.get("llm_integration", {})
            if llm_config.get("enabled", False):
                cost_controls = llm_config.get("cost_controls", {})
                budget_controller = BudgetController(
                    daily_budget_usd=cost_controls.get("daily_budget_usd", 100.0),
                    per_drawing_limit_usd=cost_controls.get(
                        "per_drawing_limit_usd", 5.0
                    ),
                    db_manager=self.db,
                )
                self.llm_gateway = LLMGateway(llm_config, budget_controller)
            else:
                self.llm_gateway = None

        # Entity Extractor (needs LLM gateway)
        self.entity_extractor = entity_extractor or EntityExtractor(
            self.config.get("entity_extraction", {}), llm_gateway=self.llm_gateway
        )

        # Build stage processor map (Strategy Pattern)
        self._stage_processors: Dict[
            ProcessingStage, Callable[[Any, WorkflowState], Tuple[Any, float]]
        ] = self._build_stage_processor_map()

        # Get configurable stage flow
        self._pipeline_stages = self._get_pipeline_stages()

        logger.info(
            "PipelineOrchestrator initialized", extra={"component": "orchestrator"}
        )

    def _build_stage_processor_map(
        self,
    ) -> Dict[ProcessingStage, Callable[[Any, WorkflowState], Tuple[Any, float]]]:
        """Build mapping of stages to processor functions.

        Returns:
            Dictionary mapping ProcessingStage to processor callables.
        """
        return {
            ProcessingStage.PDF_EXTRACTION: self._process_pdf_extraction,
            ProcessingStage.IMAGE_PREPROCESSING: self._process_image_preprocessing,
            ProcessingStage.OCR_EXTRACTION: self._process_ocr_extraction,
            ProcessingStage.ENTITY_EXTRACTION: self._process_entity_extraction,
            ProcessingStage.SHAPE_DETECTION: self._process_shape_detection,
            ProcessingStage.DATA_ASSOCIATION: self._process_data_association,
            ProcessingStage.DATA_VALIDATION: self._process_data_validation,
            ProcessingStage.HIERARCHY_BUILDING: self._process_hierarchy_building,
        }

    def _get_pipeline_stages(self) -> List[ProcessingStage]:
        """Get configurable pipeline stage sequence.

        Returns:
            Ordered list of processing stages.
        """
        # Check for custom stage configuration
        custom_stages = self.config.get("pipeline", {}).get("stages")

        if custom_stages:
            return [ProcessingStage(stage) for stage in custom_stages]

        # Default stage sequence
        return [
            ProcessingStage.PDF_EXTRACTION,
            ProcessingStage.IMAGE_PREPROCESSING,
            ProcessingStage.OCR_EXTRACTION,
            ProcessingStage.ENTITY_EXTRACTION,
            ProcessingStage.SHAPE_DETECTION,
            ProcessingStage.DATA_ASSOCIATION,
            ProcessingStage.DATA_VALIDATION,
            ProcessingStage.HIERARCHY_BUILDING,
        ]

    def process_drawing(
        self, pdf_path: str, force_llm: bool = False
    ) -> ProcessingResult:
        """Process a single drawing through the complete pipeline.

        Args:
            pdf_path: Path to PDF file.
            force_llm: Force LLM enhancement regardless of routing decision.

        Returns:
            ProcessingResult with all extracted data and metadata.

        Raises:
            DrawingProcessingError: If processing fails.
            PDFProcessingError: If PDF extraction fails.
            OCRError: If OCR extraction fails.
            ShapeDetectionError: If shape detection fails.
            BudgetExceededException: If LLM budget is exceeded.
            FileNotFoundError: If PDF file doesn't exist.
        """
        # Validate input file
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if not pdf_file.is_file():
            raise ValueError(f"Path is not a file: {pdf_path}")

        start_time = time.time()

        # Generate drawing ID
        from ..utils.file_utils import generate_unique_id

        drawing_id = generate_unique_id("DWG")

        logger.info(
            f"Processing drawing: {pdf_path}",
            extra={"drawing_id": drawing_id, "force_llm": force_llm},
        )

        # Initialize workflow state
        state = WorkflowState(
            drawing_id=drawing_id,
            source_file=pdf_path,
            current_stage=ProcessingStage.PDF_EXTRACTION,
            overall_confidence=0.0,
        )

        try:
            # Execute pipeline stages
            for stage in self._pipeline_stages:
                stage_input = self._prepare_stage_input(stage, state)
                stage_result = self._process_stage_with_retry(stage, stage_input, state)

                if not stage_result.success:
                    self._handle_stage_failure(stage, stage_result, state)

                # Update state with results
                self._update_state_from_stage(stage, stage_result, state)

            # Routing Decision
            route = self._determine_route(state, force_llm, drawing_id, pdf_path)

            logger.info(
                f"Routing decision: {route.pipeline.value} - {route.reason}",
                extra={"drawing_id": drawing_id},
            )

            # LLM Enhancement (if needed and budget allows)
            if route.pipeline != PipelineType.BASELINE_ONLY and self.llm_gateway:
                self._apply_llm_enhancement(state, route)

            # Create processing result
            processing_result = self._create_processing_result(state)

            # Calculate quality metrics BEFORE storing
            processing_result.overall_confidence = (
                self.quality_scorer.calculate_drawing_confidence(processing_result)
            )
            processing_result.review_flags = self.quality_scorer.generate_review_flags(
                processing_result
            )
            processing_result.completeness_score = (
                self.quality_scorer.assess_completeness(processing_result)
            )

            # Store in database with complete metrics
            self.db.store_drawing(processing_result)

            # Store audit trail
            total_time = time.time() - start_time
            processing_result.processing_times["total"] = total_time

            self.db.store_audit_entry(
                drawing_id=drawing_id,
                stage="complete",
                status="success",
                duration=total_time,
            )

            logger.info(
                f"Successfully processed drawing in {total_time:.2f}s",
                extra={
                    "drawing_id": drawing_id,
                    "confidence": processing_result.overall_confidence,
                    "needs_review": processing_result.needs_human_review(),
                },
            )

            return processing_result

        except BudgetExceededException as e:
            logger.error(f"Budget exceeded: {e}", extra={"drawing_id": drawing_id})
            self.db.store_audit_entry(
                drawing_id=drawing_id,
                stage=state.current_stage.value,
                status="failed",
                duration=time.time() - start_time,
                error_message=str(e),
            )
            raise

        except Exception as e:
            logger.error(
                f"Processing failed: {e}",
                extra={"drawing_id": drawing_id},
                exc_info=True,
            )
            self._handle_stage_error(state.current_stage, e, state)

            # Create minimal result for failed drawing
            processing_result = self._create_processing_result(state)
            processing_result.status = "failed"
            processing_result.error_message = str(e)

            # Store in database
            self.db.store_drawing(processing_result)

            raise DrawingProcessingError(
                f"Drawing processing failed: {e}",
                drawing_id=drawing_id,
                stage=state.current_stage.value,
            )

    def process_batch(
        self,
        pdf_paths: List[str],
        batch_id: Optional[str] = None,
        parallel_workers: int = OrchestratorConfig.DEFAULT_PARALLEL_WORKERS,
        checkpoint_every_n: int = OrchestratorConfig.DEFAULT_CHECKPOINT_FREQUENCY,
    ) -> BatchResult:
        """Process multiple drawings with parallelization and checkpointing.

        Args:
            pdf_paths: List of PDF file paths to process.
            batch_id: Optional batch identifier. Generated if not provided.
            parallel_workers: Number of parallel worker threads.
            checkpoint_every_n: Save checkpoint every N drawings.

        Returns:
            BatchResult with aggregated statistics and summaries.

        Raises:
            ValueError: If pdf_paths is empty.
        """
        if not pdf_paths:
            raise ValueError("pdf_paths cannot be empty")

        start_time = time.time()

        # Generate batch ID if not provided
        if batch_id is None:
            from ..utils.file_utils import generate_unique_id

            batch_id = generate_unique_id("BATCH")

        logger.info(
            f"Processing batch with {len(pdf_paths)} drawings",
            extra={"batch_id": batch_id, "workers": parallel_workers},
        )

        # Create checkpoint
        self.checkpoint_manager.create_checkpoint(batch_id, pdf_paths)

        # Track results
        successful = 0
        failed = 0
        needs_review = 0
        total_llm_cost = 0.0
        processing_times: List[float] = []
        drawing_summaries: List[DrawingResultSummary] = []
        completed_count = 0

        # Process drawings in parallel
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(
                    self._process_drawing_safe, pdf_path, batch_id
                ): pdf_path
                for pdf_path in pdf_paths
            }

            # Process completed tasks
            for future in as_completed(future_to_path):
                pdf_path = future_to_path[future]

                try:
                    result = future.result()

                    # Create lightweight summary
                    summary = DrawingResultSummary(
                        drawing_id=result.drawing_id,
                        source_file=result.source_file,
                        status=result.status,
                        overall_confidence=result.overall_confidence,
                        needs_review=result.needs_human_review(),
                        llm_cost=(
                            sum(u.cost_usd for u in result.llm_usage)
                            if result.llm_usage
                            else 0.0
                        ),
                        processing_time=result.processing_times.get("total", 0.0),
                        error_message=getattr(result, "error_message", None),
                    )

                    drawing_summaries.append(summary)

                    if result.status == "complete":
                        successful += 1
                    elif result.status == "failed":
                        failed += 1

                    if summary.needs_review:
                        needs_review += 1

                    total_llm_cost += summary.llm_cost
                    if summary.processing_time > 0:
                        processing_times.append(summary.processing_time)

                    completed_count += 1

                    # Periodic checkpoint (reduce I/O overhead)
                    if completed_count % checkpoint_every_n == 0:
                        self.checkpoint_manager.update_checkpoint(
                            batch_id, pdf_path, "complete", result.drawing_id
                        )

                    # Log progress
                    logger.info(
                        f"Progress: {completed_count}/{len(pdf_paths)} "
                        f"({successful} success, {failed} failed)",
                        extra={"batch_id": batch_id},
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to process {pdf_path}: {e}",
                        extra={"batch_id": batch_id},
                    )
                    failed += 1
                    completed_count += 1

                    # Update checkpoint
                    if completed_count % checkpoint_every_n == 0:
                        self.checkpoint_manager.update_checkpoint(
                            batch_id, pdf_path, "failed", error_message=str(e)
                        )

        # Final checkpoint update
        self.checkpoint_manager.finalize_checkpoint(batch_id)

        # Calculate statistics
        total_time = time.time() - start_time
        success_rate = successful / len(pdf_paths) if pdf_paths else 0.0
        review_rate = needs_review / len(pdf_paths) if pdf_paths else 0.0
        avg_processing_time = (
            sum(processing_times) / len(processing_times) if processing_times else 0.0
        )

        # Create batch result
        batch_result = BatchResult(
            batch_id=batch_id,
            total_drawings=len(pdf_paths),
            successful=successful,
            failed=failed,
            needs_review=needs_review,
            success_rate=success_rate,
            review_rate=review_rate,
            total_llm_cost=total_llm_cost,
            average_processing_time=avg_processing_time,
            drawing_summaries=drawing_summaries,
        )

        logger.info(
            f"Batch complete: {successful}/{len(pdf_paths)} successful, "
            f"${total_llm_cost:.2f} LLM cost, {total_time:.2f}s total time",
            extra={"batch_id": batch_id},
        )

        return batch_result

    def resume_batch(self, batch_id: str) -> BatchResult:
        """Resume a previously interrupted batch from checkpoint.

        Args:
            batch_id: Batch identifier for the checkpoint to resume.

        Returns:
            BatchResult for the resumed batch processing.

        Raises:
            ValueError: If no checkpoint exists for the batch_id.
        """
        logger.info("Resuming batch", extra={"batch_id": batch_id})

        # Get checkpoint state
        state = self.checkpoint_manager.get_checkpoint_state(batch_id)

        if state is None:
            raise ValueError(f"No checkpoint found for batch {batch_id}")

        # Get pending files
        pending_files = [
            file_path
            for file_path, status in state.file_status.items()
            if status == "pending"
        ]

        logger.info(
            f"Resuming {len(pending_files)} pending files", extra={"batch_id": batch_id}
        )

        # Process pending files
        return self.process_batch(pdf_paths=pending_files, batch_id=batch_id)

    def _process_stage_with_retry(
        self,
        stage: ProcessingStage,
        input_data: Union[str, Any, Dict[str, Any]],
        state: WorkflowState,
    ) -> StageResult:
        """Execute stage with retry logic and exponential backoff.

        Args:
            stage: Processing stage to execute.
            input_data: Input data for the stage.
            state: Current workflow state.

        Returns:
            StageResult from successful execution or final retry attempt.
        """
        attempt = 0
        last_error = None

        while attempt < OrchestratorConfig.MAX_RETRIES:
            try:
                return self._process_stage(stage, input_data, state)
            except Exception as e:
                last_error = e
                attempt += 1

                if not self._should_retry(e, attempt):
                    break

                # Exponential backoff
                delay = OrchestratorConfig.RETRY_INITIAL_DELAY * (
                    OrchestratorConfig.RETRY_BACKOFF_BASE ** (attempt - 1)
                )

                logger.warning(
                    f"Stage {stage.value} failed (attempt {attempt}), "
                    f"retrying in {delay:.1f}s: {e}",
                    extra={"drawing_id": state.drawing_id},
                )

                time.sleep(delay)

        # All retries exhausted
        logger.error(
            f"Stage {stage.value} failed after {attempt} attempts",
            extra={"drawing_id": state.drawing_id},
        )

        return StageResult(
            stage=stage,
            success=False,
            data=None,
            confidence=0.0,
            duration_seconds=0.0,
            error_message=str(last_error),
        )

    def _process_stage(
        self,
        stage: ProcessingStage,
        input_data: Union[str, Any, Dict[str, Any]],
        state: WorkflowState,
    ) -> StageResult:
        """Execute a single processing stage using processor map.

        Args:
            stage: Processing stage to execute.
            input_data: Input data for the stage.
            state: Current workflow state.

        Returns:
            StageResult with stage output.

        Raises:
            ValueError: If stage is not in processor map.
        """
        start_time = time.time()
        state.current_stage = stage

        logger.info(
            f"Processing stage: {stage.value}",
            extra={"drawing_id": state.drawing_id, "stage": stage.value},
        )

        try:
            # Get processor from map
            processor = self._stage_processors.get(stage)

            if processor is None:
                raise ValueError(f"No processor found for stage: {stage}")

            # Execute processor
            result_data, confidence = processor(input_data, state)

            duration = time.time() - start_time

            # Store audit entry
            self.db.store_audit_entry(
                drawing_id=state.drawing_id,
                stage=stage.value,
                status="success",
                duration=duration,
            )

            logger.info(
                f"Stage {stage.value} completed in {duration:.2f}s",
                extra={"drawing_id": state.drawing_id, "confidence": confidence},
            )

            return StageResult(
                stage=stage,
                success=True,
                data=result_data,
                confidence=confidence,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time

            logger.error(
                f"Stage {stage.value} failed: {e}",
                extra={"drawing_id": state.drawing_id},
                exc_info=True,
            )

            # Store audit entry
            self.db.store_audit_entry(
                drawing_id=state.drawing_id,
                stage=stage.value,
                status="failed",
                duration=duration,
                error_message=str(e),
            )

            raise

    # Stage Processor Methods (Strategy Pattern)

    def _process_pdf_extraction(
        self, input_data: str, state: WorkflowState
    ) -> Tuple[Any, float]:
        """Process PDF extraction stage."""
        pdf_pages = self.pdf_processor.extract_pages(input_data)
        return pdf_pages, 1.0

    def _process_image_preprocessing(
        self, input_data: Any, state: WorkflowState
    ) -> Tuple[Dict[str, Any], float]:
        """Process image preprocessing for multiple pages."""
        # Handle multi-page PDFs
        preprocessed_pages = []

        for page in state.pdf_pages:
            ocr_image = self.image_preprocessor.preprocess_for_ocr(page.image)
            detection_image = self.image_preprocessor.preprocess_for_detection(
                page.image
            )
            preprocessed_pages.append(
                {
                    "page_number": page.page_number,
                    "ocr": ocr_image,
                    "detection": detection_image,
                }
            )

        # For now, use first page (can be extended for multi-page support)
        result = {
            "ocr": preprocessed_pages[0]["ocr"],
            "detection": preprocessed_pages[0]["detection"],
            "all_pages": preprocessed_pages,
        }

        return result, 1.0

    def _process_ocr_extraction(
        self, input_data: Any, state: WorkflowState
    ) -> Tuple[OCRResult, float]:
        """Process OCR extraction stage."""
        ocr_result = self.ocr_pipeline.extract_text(input_data)
        return ocr_result, ocr_result.average_confidence

    def _process_entity_extraction(
        self, input_data: OCRResult, state: WorkflowState
    ) -> Tuple[Any, float]:
        """Process entity extraction stage."""
        entity_result = self.entity_extractor.extract_entities(input_data)
        confidence = entity_result.extraction_statistics.average_confidence
        return entity_result, confidence

    def _process_shape_detection(
        self, input_data: Any, state: WorkflowState
    ) -> Tuple[DetectionResult, float]:
        """Process shape detection stage."""
        detection_result = self.shape_detector.detect_shapes(input_data)
        return detection_result, detection_result.summary.average_confidence

    def _process_data_association(
        self, input_data: Dict[str, Any], state: WorkflowState
    ) -> Tuple[List[Any], float]:
        """Process data association stage."""
        associations = self.data_associator.associate_text_to_shapes(
            input_data["text_blocks"], input_data["detections"]
        )
        return associations, 1.0

    def _process_data_validation(
        self, input_data: Dict[str, Any], state: WorkflowState
    ) -> Tuple[Any, float]:
        """Process data validation stage."""
        validation_report = self.data_validator.validate_associations(
            input_data["ocr_result"],
            input_data["detections"],
            input_data["entities"],
            input_data["associations"],
        )
        return validation_report, validation_report.confidence_adjustment

    def _process_hierarchy_building(
        self, input_data: Dict[str, Any], state: WorkflowState
    ) -> Tuple[Any, float]:
        """Process hierarchy building stage."""
        hierarchy = self.hierarchy_builder.build_hierarchy(
            input_data["detections"], input_data["associations"]
        )
        return hierarchy, 1.0

    def _prepare_stage_input(self, stage: ProcessingStage, state: WorkflowState) -> Any:
        """Prepare input data for a given stage.

        Args:
            stage: Stage to prepare input for.
            state: Current workflow state.

        Returns:
            Appropriate input data for the stage.
        """
        if stage == ProcessingStage.PDF_EXTRACTION:
            return state.source_file
        elif stage == ProcessingStage.IMAGE_PREPROCESSING:
            return state.pdf_pages[0].image if state.pdf_pages else None
        elif stage == ProcessingStage.OCR_EXTRACTION:
            return state.ocr_image
        elif stage == ProcessingStage.ENTITY_EXTRACTION:
            return state.ocr_result
        elif stage == ProcessingStage.SHAPE_DETECTION:
            return state.detection_image
        elif stage == ProcessingStage.DATA_ASSOCIATION:
            return {
                "text_blocks": state.ocr_result.text_blocks if state.ocr_result else [],
                "detections": state.detections or [],
            }
        elif stage == ProcessingStage.DATA_VALIDATION:
            return {
                "ocr_result": state.ocr_result,
                "detections": state.detections or [],
                "entities": state.entities or [],
                "associations": state.associations or [],
            }
        elif stage == ProcessingStage.HIERARCHY_BUILDING:
            return {
                "detections": state.detections or [],
                "associations": state.associations or [],
            }
        else:
            return None

    def _update_state_from_stage(
        self, stage: ProcessingStage, stage_result: StageResult, state: WorkflowState
    ) -> None:
        """Update workflow state with stage results.

        Args:
            stage: Completed stage.
            stage_result: Result from the stage.
            state: Workflow state to update.
        """
        if stage == ProcessingStage.PDF_EXTRACTION:
            state.pdf_pages = stage_result.data
        elif stage == ProcessingStage.IMAGE_PREPROCESSING:
            state.ocr_image = stage_result.data["ocr"]
            state.detection_image = stage_result.data["detection"]
        elif stage == ProcessingStage.OCR_EXTRACTION:
            state.ocr_result = stage_result.data
            state.overall_confidence = stage_result.data.average_confidence
        elif stage == ProcessingStage.ENTITY_EXTRACTION:
            state.entities = stage_result.data.entities
            state.title_block = stage_result.data.title_block
        elif stage == ProcessingStage.SHAPE_DETECTION:
            state.detections = stage_result.data.detections
        elif stage == ProcessingStage.DATA_ASSOCIATION:
            state.associations = stage_result.data
        elif stage == ProcessingStage.DATA_VALIDATION:
            state.validation_report = stage_result.data
        elif stage == ProcessingStage.HIERARCHY_BUILDING:
            state.hierarchy = stage_result.data

    def _determine_route(
        self, state: WorkflowState, force_llm: bool, drawing_id: str, pdf_path: str
    ) -> ProcessingRoute:
        """Determine routing decision for LLM enhancement.

        Args:
            state: Current workflow state.
            force_llm: Whether to force LLM enhancement.
            drawing_id: Drawing identifier.
            pdf_path: Path to PDF file.

        Returns:
            ProcessingRoute with pipeline decision.
        """
        if force_llm:
            return ProcessingRoute(
                pipeline=PipelineType.LLM_ENHANCED,
                llm_stages=[ProcessingStage.ENTITY_EXTRACTION],
                reason="Force LLM requested by user",
            )

        return self.routing_engine.determine_route(
            drawing=self._create_mock_drawing(drawing_id, pdf_path),
            initial_result=self._create_initial_result(state),
        )

    def _handle_stage_failure(
        self, stage: ProcessingStage, stage_result: StageResult, state: WorkflowState
    ) -> None:
        """Handle stage failure by raising appropriate exception.

        Args:
            stage: Failed stage.
            stage_result: Stage result with error.
            state: Current workflow state.

        Raises:
            PDFProcessingError: For PDF extraction failures.
            OCRError: For OCR extraction failures.
            ShapeDetectionError: For shape detection failures.
            DrawingProcessingError: For other stage failures.
        """
        if stage == ProcessingStage.PDF_EXTRACTION:
            raise PDFProcessingError(
                f"PDF extraction failed: {stage_result.error_message}",
                drawing_id=state.drawing_id,
            )
        elif stage == ProcessingStage.OCR_EXTRACTION:
            raise OCRError(
                f"OCR extraction failed: {stage_result.error_message}",
                drawing_id=state.drawing_id,
            )
        elif stage == ProcessingStage.SHAPE_DETECTION:
            raise ShapeDetectionError(
                f"Shape detection failed: {stage_result.error_message}",
                drawing_id=state.drawing_id,
            )
        else:
            raise DrawingProcessingError(
                f"{stage.value} failed: {stage_result.error_message}",
                drawing_id=state.drawing_id,
                stage=stage.value,
            )

    def _apply_llm_enhancement(
        self, state: WorkflowState, route: ProcessingRoute
    ) -> None:
        """Apply LLM enhancement to low-confidence results.

        Processes specified stages with LLM assistance based on the routing
        decision. Supports OCR verification, entity extraction, and shape
        labeling.

        Args:
            state: Workflow state containing current results. Modified
                in-place with LLM-enhanced data.
            route: Routing decision specifying which stages need LLM
                enhancement.

        Note:
            - Checks budget before each LLM call
            - Catches BudgetExceededException and continues with baseline
            - Marks drawing for review if LLM enhancement fails
            - Only processes text blocks below confidence threshold
        """
        logger.info(
            f"Applying LLM enhancement: {route.llm_stages}",
            extra={"drawing_id": state.drawing_id},
        )

        try:
            # Check budget before any LLM operations
            if not self.llm_gateway.budget_controller.can_proceed(state.drawing_id):
                logger.warning(
                    "Budget limit reached, skipping LLM enhancement",
                    extra={"drawing_id": state.drawing_id},
                )
                state.llm_enhancement_skipped = True
                return

            # OCR Verification (if requested)
            if ProcessingStage.OCR_EXTRACTION in route.llm_stages:
                self._apply_llm_ocr_verification(state)

            # Entity Extraction Enhancement
            if ProcessingStage.ENTITY_EXTRACTION in route.llm_stages:
                self._apply_llm_entity_extraction(state)

            # Shape Labeling/Validation (if requested)
            if ProcessingStage.SHAPE_DETECTION in route.llm_stages:
                self._apply_llm_shape_labeling(state)

        except BudgetExceededException as e:
            logger.warning(
                f"Budget exceeded during LLM enhancement: {e}",
                extra={"drawing_id": state.drawing_id},
            )
            state.llm_enhancement_skipped = True

        except Exception as e:
            logger.error(
                f"LLM enhancement failed: {e}",
                extra={"drawing_id": state.drawing_id},
                exc_info=True,
            )
            state.llm_enhancement_skipped = True

    def _apply_llm_ocr_verification(self, state: WorkflowState) -> None:
        """Apply LLM OCR verification to low-confidence text blocks.

        Args:
            state: Workflow state with OCR results to verify.
        """
        logger.info(
            "Applying LLM OCR verification", extra={"drawing_id": state.drawing_id}
        )

        verified_count = 0
        for text_block in state.ocr_result.text_blocks:
            if text_block.confidence < OrchestratorConfig.LLM_OCR_CONFIDENCE_THRESHOLD:
                # Check budget before each verification
                if not self.llm_gateway.budget_controller.can_proceed(state.drawing_id):
                    logger.warning("Budget limit reached during OCR verification")
                    break

                try:
                    # Crop region
                    from ..utils.image_utils import crop_image

                    crop = crop_image(state.ocr_image, text_block.bbox)

                    # Verify with LLM
                    verification = self.llm_gateway.verify_ocr(
                        image_crop=crop,
                        ocr_text=text_block.content,
                        drawing_id=state.drawing_id,
                        region_type=text_block.region_type,
                    )

                    # Update text block
                    text_block.content = verification.corrected_text
                    text_block.confidence = verification.confidence
                    verified_count += 1

                except Exception as e:
                    logger.warning(
                        f"OCR verification failed for text block: {e}",
                        extra={"drawing_id": state.drawing_id},
                    )

        logger.info(
            f"LLM verified {verified_count} text blocks",
            extra={"drawing_id": state.drawing_id},
        )

    def _apply_llm_entity_extraction(self, state: WorkflowState) -> None:
        """Apply LLM entity extraction for missing critical fields.

        Args:
            state: Workflow state with entity extraction results.
        """
        logger.info(
            "Applying LLM entity extraction", extra={"drawing_id": state.drawing_id}
        )

        # Check budget before extraction
        if not self.llm_gateway.budget_controller.can_proceed(state.drawing_id):
            logger.warning("Budget limit reached, skipping entity extraction")
            return

        # Find missing critical entities
        missing_types = []
        has_part_number = any(
            e.entity_type == EntityType.PART_NUMBER for e in state.entities
        )

        if not has_part_number:
            missing_types.append("PART_NUMBER")

        if missing_types:
            try:
                # Extract with LLM
                llm_entities = self.llm_gateway.extract_entities_llm(
                    text=state.ocr_result.full_text,
                    context="Technical engineering drawing",
                    entity_types=missing_types,
                    drawing_id=state.drawing_id,
                )

                state.entities.extend(llm_entities)
                logger.info(
                    f"LLM extracted {len(llm_entities)} additional entities",
                    extra={"drawing_id": state.drawing_id},
                )
            except Exception as e:
                logger.warning(
                    f"LLM entity extraction failed: {e}",
                    extra={"drawing_id": state.drawing_id},
                )

    def _apply_llm_shape_labeling(self, state: WorkflowState) -> None:
        """Apply LLM shape labeling/validation.

        Args:
            state: Workflow state with shape detection results.
        """
        logger.info(
            "Applying LLM shape labeling", extra={"drawing_id": state.drawing_id}
        )

        # Check budget
        if not self.llm_gateway.budget_controller.can_proceed(state.drawing_id):
            logger.warning("Budget limit reached, skipping shape labeling")
            return

        # Find unlabeled or low-confidence shapes
        unlabeled_shapes = [
            d for d in state.detections if d.confidence < 0.7 or not hasattr(d, "label")
        ]

        if unlabeled_shapes:
            try:
                # Use LLM for shape classification/labeling
                labeled_count = 0
                for detection in unlabeled_shapes[:5]:  # Limit to 5 for budget
                    if not self.llm_gateway.budget_controller.can_proceed(
                        state.drawing_id
                    ):
                        break

                    # Crop shape region
                    from ..utils.image_utils import crop_image

                    crop = crop_image(state.detection_image, detection.bbox)

                    # Classify with LLM
                    label = self.llm_gateway.classify_shape(
                        image_crop=crop,
                        drawing_id=state.drawing_id,
                    )

                    detection.class_name = label
                    labeled_count += 1

                logger.info(
                    f"LLM labeled {labeled_count} shapes",
                    extra={"drawing_id": state.drawing_id},
                )
            except Exception as e:
                logger.warning(
                    f"LLM shape labeling failed: {e}",
                    extra={"drawing_id": state.drawing_id},
                )

    def _should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if a failed stage should be retried.

        Evaluates the error type and current attempt count to decide
        whether to retry the failed operation. Only retriable errors
        (e.g., transient network issues, database locks) are retried.

        Args:
            error: Exception that occurred during processing.
            attempt: Current attempt number (1-indexed).

        Returns:
            True if the operation should be retried, False otherwise.

        Note:
            Always returns False if attempt >= max_retries or if the error
            is not classified as retriable.
        """
        if attempt >= OrchestratorConfig.MAX_RETRIES:
            return False

        # Check if error is retriable
        retriable_errors = (
            ConnectionError,
            TimeoutError,
            DatabaseError,
            # Add other transient errors
        )
        return isinstance(error, retriable_errors)

    def _handle_stage_error(
        self, stage: ProcessingStage, error: Exception, state: WorkflowState
    ) -> None:
        """Log and handle stage processing errors.

        Creates a structured error log entry with context information about
        the failed stage, drawing, and error details.

        Args:
            stage: Processing stage that failed.
            error: Exception that was raised during processing.
            state: Current workflow state containing drawing context.

        Note:
            Does not raise exceptions; only logs for audit purposes.
        """
        logger.error(
            f"Stage error: {error}",
            extra={
                "drawing_id": state.drawing_id,
                "stage": stage.value,
                "source_file": state.source_file,
            },
            exc_info=True,
        )

    def _create_processing_result(self, state: WorkflowState) -> ProcessingResult:
        """Create ProcessingResult from accumulated workflow state.

        Assembles all processing artifacts from the workflow state into a
        complete ProcessingResult object for storage and analysis.

        Args:
            state: Workflow state containing all processing results.

        Returns:
            ProcessingResult with all available data. Some fields
            (confidence_scores, review_flags, completeness_score) are
            initially None and populated by the quality scorer.

        Note:
            Sets pipeline_type to BASELINE_ONLY initially; updated by
            routing logic. Status defaults to "complete" unless explicitly
            changed by error handling.
        """
        return ProcessingResult(
            drawing_id=state.drawing_id,
            source_file=state.source_file,
            processing_timestamp=datetime.now(),
            pipeline_type=PipelineType.BASELINE_ONLY,
            pipeline_version="1.0.0",
            pdf_pages=state.pdf_pages or [],
            ocr_result=state.ocr_result,
            entities=state.entities or [],
            title_block=state.title_block,
            detections=state.detections or [],
            associations=state.associations or [],
            hierarchy=state.hierarchy,
            validation_report=state.validation_report,
            overall_confidence=state.overall_confidence,
            confidence_scores=None,  # Set by quality scorer
            review_flags=[],  # Set by quality scorer
            completeness_score=None,  # Set by quality scorer
            llm_usage=[],
            processing_times={},
            status="complete",
        )

    def _create_mock_drawing(self, drawing_id: str, pdf_path: str) -> Drawing:
        """Create a minimal Drawing object for routing decisions.

        Constructs a lightweight Drawing object with default values, used
        by the routing engine to make pipeline selection decisions before
        full processing completes.

        Args:
            drawing_id: Unique identifier for the drawing.
            pdf_path: File path to the source PDF.

        Returns:
            Drawing object with medium priority and empty metadata.
        """
        return Drawing(
            drawing_id=drawing_id,
            source_file=pdf_path,
            priority=Priority.MEDIUM,
            metadata={},
        )

    def _create_initial_result(self, state: WorkflowState) -> InitialRoutingResult:
        """Create initial result object for routing decisions.

        Builds a simple result object containing early-stage metrics used
        by the routing engine to determine whether LLM enhancement is
        needed.

        Args:
            state: Current workflow state with OCR and entity extraction
                results.

        Returns:
            InitialRoutingResult with attributes: entities, detections,
            ocr_avg_confidence, text_block_count. Returns default values
            (empty lists, 0.0) for missing data.
        """
        return InitialRoutingResult(
            entities=state.entities or [],
            detections=state.detections or [],
            ocr_avg_confidence=(
                state.ocr_result.average_confidence if state.ocr_result else 0.0
            ),
            text_block_count=(
                len(state.ocr_result.text_blocks) if state.ocr_result else 0
            ),
        )

    def _process_drawing_safe(self, pdf_path: str, batch_id: str) -> ProcessingResult:
        """Process drawing with error handling for batch processing.

        Wraps process_drawing() with exception handling to ensure batch
        processing continues even if individual drawings fail. Creates a
        minimal failed result instead of propagating exceptions.

        Args:
            pdf_path: Path to the PDF file to process.
            batch_id: Identifier for the current batch (for logging
                context).

        Returns:
            ProcessingResult with status='complete' on success or
            status='failed' with error_message on failure.

        Note:
            Never raises exceptions; always returns a ProcessingResult
            object.
        """
        try:
            return self.process_drawing(pdf_path)
        except Exception as e:
            logger.error(
                f"Failed to process {pdf_path}",
                extra={"batch_id": batch_id, "pdf_path": pdf_path},
                exc_info=True,
            )

            # Create minimal failed result
            from ..utils.file_utils import generate_unique_id

            drawing_id = generate_unique_id("DWG")

            return ProcessingResult(
                drawing_id=drawing_id,
                source_file=pdf_path,
                processing_timestamp=datetime.now(),
                pipeline_type=PipelineType.BASELINE_ONLY,
                pipeline_version="1.0.0",
                pdf_pages=[],
                ocr_result=None,
                entities=[],
                title_block=None,
                detections=[],
                associations=[],
                hierarchy=None,
                validation_report=None,
                overall_confidence=0.0,
                confidence_scores=None,
                review_flags=[],
                completeness_score=None,
                llm_usage=[],
                processing_times={},
                status="failed",
                error_message=str(e),
            )
