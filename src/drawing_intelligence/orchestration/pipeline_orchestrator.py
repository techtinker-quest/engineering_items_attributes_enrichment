"""
Pipeline Orchestrator Module

Manages end-to-end workflow execution, error handling, and checkpointing.
"""

import logging
from pathlib import Path
from typing import List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils.config_loader import Config
from ..database.database_manager import DatabaseManager
from .checkpoint_manager import CheckpointManager
from .routing_engine import RoutingEngine, ProcessingRoute
from .workflow_state import WorkflowState
from ..models.data_structures import (
    ProcessingStage,
    ProcessingResult,
    PipelineType,
    Priority,
    EntityType,
    FlagType,
    Severity,
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

# Import error handlers
from ..utils.error_handlers import (
    DrawingProcessingError,
    PDFProcessingError,
    OCRError,
    ShapeDetectionError,
    BudgetExceededException,
)


logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Result from a single processing stage."""

    stage: ProcessingStage
    success: bool
    data: Any
    confidence: float
    duration_seconds: float
    error_message: Optional[str] = None


@dataclass
class BatchResult:
    """Result from batch processing."""

    batch_id: str
    total_drawings: int
    successful: int
    failed: int
    needs_review: int
    success_rate: float
    review_rate: float
    total_llm_cost: float
    average_processing_time: float
    drawing_results: List[ProcessingResult]


class PipelineOrchestrator:
    """
    Manages end-to-end workflow execution.

    Coordinates all processing stages, error handling, and checkpointing.
    """

    def __init__(
        self,
        config: Config,
        db: DatabaseManager,
        checkpoint_manager: CheckpointManager,
        routing_engine: RoutingEngine,
    ):
        """
        Initialize pipeline orchestrator.

        Args:
            config: System configuration
            db: Database manager
            checkpoint_manager: Checkpoint manager
            routing_engine: Routing engine
        """
        self.config = config
        self.db = db
        self.checkpoint_manager = checkpoint_manager
        self.routing_engine = routing_engine

        # Initialize processing components
        self._initialize_processors()

        logger.info("PipelineOrchestrator initialized")

    def _initialize_processors(self):
        """Initialize all processing components."""
        logger.info("Initializing processing components...")

        # PDF Processing
        self.pdf_processor = PDFProcessor(self.config.pdf_processing)

        # Image Preprocessing
        self.image_preprocessor = ImagePreprocessor(self.config.image_preprocessing)

        # OCR Pipeline
        self.ocr_pipeline = OCRPipeline(self.config.ocr)

        # Entity Extraction
        llm_gateway = None
        if self.config.llm_integration.enabled:
            from ..llm.llm_gateway import LLMGateway
            from ..llm.budget_controller import BudgetController

            budget_controller = BudgetController(
                daily_budget_usd=self.config.llm_integration.cost_controls.daily_budget_usd,
                per_drawing_limit_usd=self.config.llm_integration.cost_controls.per_drawing_limit_usd,
                db_manager=self.db,
            )
            llm_gateway = LLMGateway(self.config.llm_integration, budget_controller)

        self.entity_extractor = EntityExtractor(
            self.config.entity_extraction, llm_gateway=llm_gateway
        )

        # Shape Detection
        self.shape_detector = ShapeDetector(self.config.shape_detection)

        # Data Association
        self.data_associator = DataAssociator(self.config.data_association)

        # Data Validation
        self.data_validator = DataValidator(self.config.validation)

        # Hierarchy Builder
        self.hierarchy_builder = HierarchyBuilder()

        # Quality Scorer
        self.quality_scorer = QualityScorer(self.config.quality_scoring)

        # LLM Gateway
        self.llm_gateway = llm_gateway

        logger.info("Processing components initialized")

    def process_drawing(
        self, pdf_path: str, force_llm: bool = False
    ) -> ProcessingResult:
        """
        Process a single drawing through the complete pipeline.

        Args:
            pdf_path: Path to PDF file
            force_llm: Force LLM enhancement regardless of routing

        Returns:
            ProcessingResult with all extracted data and metadata

        Raises:
            DrawingProcessingError: If processing fails
        """
        start_time = time.time()

        logger.info(f"Processing drawing: {pdf_path}")

        # Generate drawing ID
        from ..utils.file_utils import generate_unique_id

        drawing_id = generate_unique_id("DWG")

        # Initialize workflow state
        state = WorkflowState(
            drawing_id=drawing_id,
            source_file=pdf_path,
            current_stage=ProcessingStage.PDF_EXTRACTION,
            overall_confidence=0.0,
        )

        try:
            # Stage 1: PDF Extraction
            stage_result = self._process_stage(
                ProcessingStage.PDF_EXTRACTION, pdf_path, state
            )

            if not stage_result.success:
                raise PDFProcessingError(
                    f"PDF extraction failed: {stage_result.error_message}",
                    drawing_id=drawing_id,
                )

            pdf_pages = stage_result.data
            state.pdf_pages = pdf_pages

            # Stage 2: Image Preprocessing
            stage_result = self._process_stage(
                ProcessingStage.IMAGE_PREPROCESSING, pdf_pages[0].image, state
            )

            if not stage_result.success:
                raise DrawingProcessingError(
                    f"Image preprocessing failed: {stage_result.error_message}",
                    drawing_id=drawing_id,
                )

            preprocessed_images = stage_result.data
            state.ocr_image = preprocessed_images["ocr"]
            state.detection_image = preprocessed_images["detection"]

            # Stage 3: OCR Extraction
            stage_result = self._process_stage(
                ProcessingStage.OCR_EXTRACTION, state.ocr_image, state
            )

            if not stage_result.success:
                raise OCRError(
                    f"OCR extraction failed: {stage_result.error_message}",
                    drawing_id=drawing_id,
                )

            ocr_result = stage_result.data
            state.ocr_result = ocr_result
            state.overall_confidence = ocr_result.average_confidence

            # Stage 4: Entity Extraction
            stage_result = self._process_stage(
                ProcessingStage.ENTITY_EXTRACTION, ocr_result, state
            )

            entity_result = stage_result.data
            state.entities = entity_result.entities
            state.title_block = entity_result.title_block

            # Stage 5: Shape Detection
            stage_result = self._process_stage(
                ProcessingStage.SHAPE_DETECTION, state.detection_image, state
            )

            if not stage_result.success:
                raise ShapeDetectionError(
                    f"Shape detection failed: {stage_result.error_message}",
                    drawing_id=drawing_id,
                )

            detection_result = stage_result.data
            state.detections = detection_result.detections

            # Routing Decision
            if not force_llm:
                route = self.routing_engine.determine_route(
                    drawing=self._create_mock_drawing(drawing_id, pdf_path),
                    initial_result=self._create_initial_result(state),
                )
            else:
                # Force LLM enhancement
                from ..orchestration.routing_engine import ProcessingRoute

                route = ProcessingRoute(
                    pipeline=PipelineType.LLM_ENHANCED,
                    llm_stages=[
                        ProcessingStage.OCR_VERIFICATION,
                        ProcessingStage.ENTITY_EXTRACTION,
                    ],
                    reason="Force LLM requested by user",
                )

            logger.info(f"Routing decision: {route.pipeline.value} - {route.reason}")

            # LLM Enhancement (if needed)
            if route.pipeline != PipelineType.BASELINE_ONLY and self.llm_gateway:
                self._apply_llm_enhancement(state, route)

            # Stage 6: Data Association
            stage_result = self._process_stage(
                ProcessingStage.DATA_ASSOCIATION,
                {"text_blocks": ocr_result.text_blocks, "detections": state.detections},
                state,
            )

            state.associations = stage_result.data

            # Stage 7: Data Validation
            stage_result = self._process_stage(
                ProcessingStage.DATA_VALIDATION,
                {
                    "ocr_result": ocr_result,
                    "detections": state.detections,
                    "entities": state.entities,
                    "associations": state.associations,
                },
                state,
            )

            state.validation_report = stage_result.data

            # Stage 8: Hierarchy Building
            stage_result = self._process_stage(
                ProcessingStage.HIERARCHY_BUILDING,
                {"detections": state.detections, "associations": state.associations},
                state,
            )

            state.hierarchy = stage_result.data

            # Create processing result
            processing_result = self._create_processing_result(state)

            # Calculate quality metrics
            processing_result.overall_confidence = (
                self.quality_scorer.calculate_drawing_confidence(processing_result)
            )
            processing_result.review_flags = self.quality_scorer.generate_review_flags(
                processing_result
            )
            processing_result.completeness_score = (
                self.quality_scorer.assess_completeness(processing_result)
            )

            # Store in database
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
                f"Successfully processed drawing {drawing_id} in {total_time:.2f}s"
            )

            return processing_result

        except BudgetExceededException as e:
            logger.error(f"Budget exceeded for {drawing_id}: {e}")
            self.db.store_audit_entry(
                drawing_id=drawing_id,
                stage=state.current_stage.value,
                status="failed",
                duration=time.time() - start_time,
                error_message=str(e),
            )
            raise

        except Exception as e:
            logger.error(f"Processing failed for {drawing_id}: {e}", exc_info=True)
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
        parallel_workers: int = 4,
    ) -> BatchResult:
        """
        Process multiple drawings with parallelization and checkpointing.

        Args:
            pdf_paths: List of PDF file paths
            batch_id: Optional batch identifier
            parallel_workers: Number of parallel workers

        Returns:
            BatchResult with aggregated statistics
        """
        start_time = time.time()

        # Generate batch ID if not provided
        if batch_id is None:
            from ..utils.file_utils import generate_unique_id

            batch_id = generate_unique_id("BATCH")

        logger.info(f"Processing batch {batch_id} with {len(pdf_paths)} drawings")

        # Create checkpoint
        self.checkpoint_manager.create_checkpoint(batch_id, pdf_paths)

        # Track results
        successful = 0
        failed = 0
        needs_review = 0
        total_llm_cost = 0.0
        processing_times = []
        drawing_results = []

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

                    if result.status == "complete":
                        successful += 1
                    elif result.status == "failed":
                        failed += 1

                    if result.needs_human_review():
                        needs_review += 1

                    # Track LLM cost
                    if result.llm_usage:
                        for usage in result.llm_usage:
                            total_llm_cost += usage.cost_usd

                    # Track processing time
                    if "total" in result.processing_times:
                        processing_times.append(result.processing_times["total"])

                    drawing_results.append(result)

                    # Update checkpoint
                    self.checkpoint_manager.update_checkpoint(
                        batch_id, pdf_path, "complete", result.drawing_id
                    )

                    # Log progress
                    completed = successful + failed
                    logger.info(
                        f"Progress: {completed}/{len(pdf_paths)} "
                        f"({successful} success, {failed} failed)"
                    )

                except Exception as e:
                    logger.error(f"Failed to process {pdf_path}: {e}")
                    failed += 1

                    # Update checkpoint
                    self.checkpoint_manager.update_checkpoint(
                        batch_id, pdf_path, "failed", error_message=str(e)
                    )

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
            drawing_results=drawing_results,
        )

        logger.info(
            f"Batch {batch_id} complete: "
            f"{successful}/{len(pdf_paths)} successful, "
            f"${total_llm_cost:.2f} LLM cost, "
            f"{total_time:.2f}s total time"
        )

        return batch_result

    def resume_batch(self, batch_id: str) -> BatchResult:
        """
        Resume a previously interrupted batch from checkpoint.

        Args:
            batch_id: Batch identifier

        Returns:
            BatchResult for the resumed batch
        """
        logger.info(f"Resuming batch {batch_id}")

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

        logger.info(f"Resuming {len(pending_files)} pending files")

        # Process pending files
        return self.process_batch(pdf_paths=pending_files, batch_id=batch_id)

    def _process_stage(
        self, stage: ProcessingStage, input_data: Any, state: WorkflowState
    ) -> StageResult:
        """
        Execute a single processing stage.

        Args:
            stage: Processing stage to execute
            input_data: Input data for stage
            state: Current workflow state

        Returns:
            StageResult with stage output
        """
        start_time = time.time()
        state.current_stage = stage

        logger.info(f"Processing stage: {stage.value}")

        try:
            # Route to appropriate processor
            if stage == ProcessingStage.PDF_EXTRACTION:
                pdf_pages = self.pdf_processor.extract_pages(input_data)
                result_data = pdf_pages
                confidence = 1.0

            elif stage == ProcessingStage.IMAGE_PREPROCESSING:
                ocr_image = self.image_preprocessor.preprocess_for_ocr(input_data)
                detection_image = self.image_preprocessor.preprocess_for_detection(
                    input_data
                )
                result_data = {"ocr": ocr_image, "detection": detection_image}
                confidence = 1.0

            elif stage == ProcessingStage.OCR_EXTRACTION:
                ocr_result = self.ocr_pipeline.extract_text(input_data)
                result_data = ocr_result
                confidence = ocr_result.average_confidence

            elif stage == ProcessingStage.ENTITY_EXTRACTION:
                entity_result = self.entity_extractor.extract_entities(input_data)
                result_data = entity_result
                confidence = entity_result.extraction_statistics.average_confidence

            elif stage == ProcessingStage.SHAPE_DETECTION:
                detection_result = self.shape_detector.detect_shapes(input_data)
                result_data = detection_result
                confidence = detection_result.summary.average_confidence

            elif stage == ProcessingStage.DATA_ASSOCIATION:
                associations = self.data_associator.associate_text_to_shapes(
                    input_data["text_blocks"], input_data["detections"]
                )
                result_data = associations
                confidence = 1.0

            elif stage == ProcessingStage.DATA_VALIDATION:
                validation_report = self.data_validator.validate_associations(
                    input_data["ocr_result"],
                    input_data["detections"],
                    input_data["entities"],
                    input_data["associations"],
                )
                result_data = validation_report
                confidence = validation_report.confidence_adjustment

            elif stage == ProcessingStage.HIERARCHY_BUILDING:
                hierarchy = self.hierarchy_builder.build_hierarchy(
                    input_data["detections"], input_data["associations"]
                )
                result_data = hierarchy
                confidence = 1.0

            else:
                raise ValueError(f"Unknown stage: {stage}")

            duration = time.time() - start_time

            # Store audit entry
            self.db.store_audit_entry(
                drawing_id=state.drawing_id,
                stage=stage.value,
                status="success",
                duration=duration,
            )

            logger.info(f"Stage {stage.value} completed in {duration:.2f}s")

            return StageResult(
                stage=stage,
                success=True,
                data=result_data,
                confidence=confidence,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time

            logger.error(f"Stage {stage.value} failed: {e}", exc_info=True)

            # Store audit entry
            self.db.store_audit_entry(
                drawing_id=state.drawing_id,
                stage=stage.value,
                status="failed",
                duration=duration,
                error_message=str(e),
            )

            return StageResult(
                stage=stage,
                success=False,
                data=None,
                confidence=0.0,
                duration_seconds=duration,
                error_message=str(e),
            )

    def _apply_llm_enhancement(self, state: WorkflowState, route: ProcessingRoute):
        """Apply LLM enhancement based on routing decision."""
        logger.info(f"Applying LLM enhancement: {route.llm_stages}")

        try:
            # OCR Verification
            if ProcessingStage.OCR_VERIFICATION in route.llm_stages:
                logger.info("Applying LLM OCR verification")

                for text_block in state.ocr_result.text_blocks:
                    if text_block.confidence < 0.85:
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

            # Entity Extraction Enhancement
            if ProcessingStage.ENTITY_EXTRACTION in route.llm_stages:
                logger.info("Applying LLM entity extraction")

                # Find missing critical entities
                missing_types = []
                has_part_number = any(
                    e.entity_type == EntityType.PART_NUMBER for e in state.entities
                )

                if not has_part_number:
                    missing_types.append("PART_NUMBER")

                if missing_types:
                    # Extract with LLM
                    llm_entities = self.llm_gateway.extract_entities_llm(
                        text=state.ocr_result.full_text,
                        context="Technical engineering drawing",
                        entity_types=missing_types,
                        drawing_id=state.drawing_id,
                    )

                    state.entities.extend(llm_entities)

        except BudgetExceededException as e:
            logger.warning(f"Budget exceeded during LLM enhancement: {e}")
            # Continue with baseline results

        except Exception as e:
            logger.error(f"LLM enhancement failed: {e}")
            # Continue with baseline results

    def _should_retry(
        self, error: Exception, attempt: int, max_retries: int = 3
    ) -> bool:
        """
        Determine if a failed stage should be retried.

        Args:
            error: Exception that occurred
            attempt: Current attempt number
            max_retries: Maximum retry attempts

        Returns:
            bool indicating retry decision
        """
        if attempt >= max_retries:
            return False

        # Check if error is retriable
        from ..utils.error_handlers import is_retriable_error

        return is_retriable_error(error)

    def _handle_stage_error(
        self, stage: ProcessingStage, error: Exception, state: WorkflowState
    ):
        """
        Log and handle stage errors.

        Args:
            stage: Stage that failed
            error: Exception that occurred
            state: Current workflow state
        """
        from ..utils.error_handlers import log_error_with_context

        context = {
            "drawing_id": state.drawing_id,
            "stage": stage.value,
            "source_file": state.source_file,
        }

        log_error_with_context(error, logger, context)

    def _create_processing_result(self, state: WorkflowState) -> ProcessingResult:
        """
        Create ProcessingResult from workflow state.

        Args:
            state: Workflow state

        Returns:
            ProcessingResult object
        """
        return ProcessingResult(
            drawing_id=state.drawing_id,
            source_file=state.source_file,
            processing_timestamp=datetime.now(),
            pipeline_type=PipelineType.BASELINE_ONLY,  # Updated by routing
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
            confidence_scores=None,  # Will be set by quality scorer
            review_flags=[],  # Will be set by quality scorer
            completeness_score=None,  # Will be set by quality scorer
            llm_usage=[],
            processing_times={},
            status="complete",
        )

    def _create_mock_drawing(self, drawing_id: str, pdf_path: str):
        """Create mock drawing for routing."""
        from ..orchestration.routing_engine import Drawing

        return Drawing(
            drawing_id=drawing_id,
            source_file=pdf_path,
            priority=Priority.MEDIUM,
            metadata={},
        )

    def _create_initial_result(self, state: WorkflowState):
        """Create initial result for routing."""

        class InitialResult:
            def __init__(self, state):
                self.entities = state.entities or []
                self.detections = state.detections or []
                self.ocr_avg_confidence = (
                    state.ocr_result.average_confidence if state.ocr_result else 0.0
                )
                self.text_block_count = (
                    len(state.ocr_result.text_blocks) if state.ocr_result else 0
                )

        return InitialResult(state)

    def _process_drawing_safe(self, pdf_path: str, batch_id: str) -> ProcessingResult:
        """
        Process drawing with error handling for batch processing.

        Args:
            pdf_path: Path to PDF file
            batch_id: Batch identifier

        Returns:
            ProcessingResult (may have status='failed')
        """
        try:
            return self.process_drawing(pdf_path)
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")

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
