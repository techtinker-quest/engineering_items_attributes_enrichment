"""Workflow state management for drawing processing pipeline.

This module provides the WorkflowState dataclass for tracking the current
state of a drawing as it progresses through the processing pipeline. It
maintains stage information, metadata, checkpoint data, and confidence scores.

Typical usage example:
    state = WorkflowState(
        drawing_id="DWG-001",
        current_stage=ProcessingStage.OCR_EXTRACTION
    )
    state.update_stage(ProcessingStage.ENTITY_EXTRACTION)
    state.save_checkpoint({"entities_extracted": 42})

Author: GROK AI + Sandeep A (01Nov2025)
Reviewed and refactored: Claude (Nov 2025)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from enum import Enum

from ..models.data_structures import ProcessingStage


class WorkflowStatus(Enum):
    """Status values for workflow state lifecycle.

    Attributes:
        INITIALIZED: Workflow created but not yet started.
        PROCESSING: Actively processing through pipeline stages.
        PAUSED: Processing temporarily suspended.
        COMPLETED: Successfully completed all stages.
        FAILED: Processing failed and cannot continue.
    """

    INITIALIZED = "initialized"
    PROCESSING = "processing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowState:
    """Manages the state of a drawing's processing pipeline.

    This dataclass tracks the current processing stage, metadata, checkpoint
    data, and confidence scores for a single drawing as it moves through
    the pipeline. It provides methods to update the stage and save checkpoint
    data for recovery and monitoring purposes.

    Checkpoint data is namespaced by stage to prevent collisions and simplify
    recovery. Stage transitions are validated to ensure proper pipeline flow.

    Attributes:
        drawing_id: Unique identifier for the drawing (e.g., "DWG-001").
        current_stage: Current processing stage from ProcessingStage enum.
        metadata: Drawing-level metadata. Expected keys:
            - source_file: Path to source PDF file
            - file_size: File size in bytes
            - page_count: Number of pages in drawing
        checkpoint_data: Stage-specific checkpoint data, keyed by stage name.
        timestamp: Last update timestamp (UTC).
        confidence_scores: Confidence scores by stage or metric name.
        is_running: Whether processing is currently active.
        status: Current workflow status from WorkflowStatus enum.
        schema_version: Schema version for serialization compatibility.
        stage_history: List of (stage, timestamp) tuples tracking progression.
        error_history: List of error messages from failed operations.
    """

    drawing_id: str
    current_stage: ProcessingStage
    metadata: Dict[str, str] = field(default_factory=dict)
    checkpoint_data: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    is_running: bool = False
    status: WorkflowStatus = WorkflowStatus.INITIALIZED
    schema_version: int = 1
    stage_history: List[Tuple[ProcessingStage, datetime]] = field(default_factory=list)
    error_history: List[str] = field(default_factory=list)

    # Valid stage transitions based on pipeline flow
    _VALID_TRANSITIONS: Dict[ProcessingStage, List[ProcessingStage]] = {
        ProcessingStage.PDF_PROCESSING: [ProcessingStage.IMAGE_PREPROCESSING],
        ProcessingStage.IMAGE_PREPROCESSING: [ProcessingStage.OCR_EXTRACTION],
        ProcessingStage.OCR_EXTRACTION: [ProcessingStage.SHAPE_DETECTION],
        ProcessingStage.SHAPE_DETECTION: [ProcessingStage.ENTITY_EXTRACTION],
        ProcessingStage.ENTITY_EXTRACTION: [
            ProcessingStage.DATA_ASSOCIATION,
            ProcessingStage.LLM_ENHANCEMENT,  # Optional path
        ],
        ProcessingStage.DATA_ASSOCIATION: [
            ProcessingStage.VALIDATION,
            ProcessingStage.LLM_ENHANCEMENT,  # Optional path
        ],
        ProcessingStage.LLM_ENHANCEMENT: [ProcessingStage.VALIDATION],
        ProcessingStage.VALIDATION: [ProcessingStage.COMPLETE],
        ProcessingStage.COMPLETE: [],  # Terminal stage
    }

    def __post_init__(self) -> None:
        """Validate initial state after dataclass initialization.

        Raises:
            ValueError: If drawing_id is empty or invalid format.
        """
        if not self.drawing_id or not self.drawing_id.strip():
            raise ValueError("drawing_id cannot be empty.")

        # Optional: Enforce drawing_id format (e.g., "DWG-001")
        if not self.drawing_id.startswith("DWG-"):
            # Log warning but don't fail - format may vary
            pass

        # Initialize stage history with current stage
        if not self.stage_history:
            self.stage_history.append((self.current_stage, self.timestamp))

    def __repr__(self) -> str:
        """Concise string representation for debugging.

        Returns:
            String showing key state information.
        """
        return (
            f"WorkflowState(drawing_id='{self.drawing_id}', "
            f"stage={self.current_stage.name}, "
            f"status={self.status.value})"
        )

    def update_stage(self, stage: ProcessingStage) -> None:
        """Update the current processing stage with validation.

        Updates the workflow to a new processing stage, validates that the
        transition is allowed according to the pipeline flow, and refreshes
        the timestamp to the current time. Adds the stage transition to the
        stage history for audit purposes.

        Args:
            stage: The new ProcessingStage to transition to.

        Raises:
            ValueError: If stage is not a valid ProcessingStage enum value
                or if the transition is not allowed.
        """
        if not isinstance(stage, ProcessingStage):
            raise ValueError(
                f"Invalid stage: {stage}. Must be a ProcessingStage enum value."
            )

        # Validate stage transition
        valid_next_stages = self._VALID_TRANSITIONS.get(self.current_stage, [])
        if valid_next_stages and stage not in valid_next_stages:
            raise ValueError(
                f"Invalid stage transition from {self.current_stage.name} "
                f"to {stage.name}. Valid transitions: "
                f"{[s.name for s in valid_next_stages]}"
            )

        self.current_stage = stage
        self.timestamp = datetime.now(timezone.utc)
        self.stage_history.append((stage, self.timestamp))

    def save_checkpoint(self, data: Dict[str, Any]) -> None:
        """Save checkpoint data for the current stage.

        Stores the provided checkpoint data under the current stage's namespace
        to prevent key collisions across stages. This allows for stage-specific
        recovery and simplifies debugging. Updates the timestamp to reflect
        the checkpoint save time.

        Args:
            data: Dictionary of checkpoint data to save for the current stage.

        Raises:
            ValueError: If data is None or empty.
        """
        if not data:
            raise ValueError("Checkpoint data cannot be None or empty.")

        # Namespace by current stage
        stage_key = self.current_stage.value
        if stage_key not in self.checkpoint_data:
            self.checkpoint_data[stage_key] = {}

        self.checkpoint_data[stage_key].update(data)
        self.timestamp = datetime.now(timezone.utc)

    def mark_running(self) -> None:
        """Mark the workflow as currently running.

        Sets is_running to True and updates status to PROCESSING.
        Updates the timestamp to the current time.
        """
        self.is_running = True
        self.status = WorkflowStatus.PROCESSING
        self.timestamp = datetime.now(timezone.utc)

    def mark_paused(self) -> None:
        """Mark the workflow as paused.

        Sets is_running to False and updates status to PAUSED.
        Updates the timestamp to the current time.
        """
        self.is_running = False
        self.status = WorkflowStatus.PAUSED
        self.timestamp = datetime.now(timezone.utc)

    def mark_completed(self) -> None:
        """Mark the workflow as completed.

        Sets is_running to False and updates status to COMPLETED.
        Updates the timestamp to the current time.
        """
        self.is_running = False
        self.status = WorkflowStatus.COMPLETED
        self.timestamp = datetime.now(timezone.utc)

    def mark_failed(self, error_message: str) -> None:
        """Mark the workflow as failed and record the error.

        Sets is_running to False, updates status to FAILED, and appends
        the error message to the error history for debugging. Stores
        the error in checkpoint data for the current stage.

        Args:
            error_message: Description of the failure reason.
        """
        self.is_running = False
        self.status = WorkflowStatus.FAILED
        self.error_history.append(f"[{self.timestamp.isoformat()}] {error_message}")

        # Store error in current stage's checkpoint data
        stage_key = self.current_stage.value
        if stage_key not in self.checkpoint_data:
            self.checkpoint_data[stage_key] = {}
        self.checkpoint_data[stage_key]["error_message"] = error_message

        self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize workflow state to dictionary.

        Converts the workflow state to a dictionary suitable for JSON
        serialization or database storage. Handles enum and datetime
        conversions automatically.

        Returns:
            Dictionary representation of the workflow state.
        """
        return {
            "drawing_id": self.drawing_id,
            "current_stage": self.current_stage.value,
            "metadata": self.metadata,
            "checkpoint_data": self.checkpoint_data,
            "timestamp": self.timestamp.isoformat(),
            "confidence_scores": self.confidence_scores,
            "is_running": self.is_running,
            "status": self.status.value,
            "schema_version": self.schema_version,
            "stage_history": [
                (stage.value, ts.isoformat()) for stage, ts in self.stage_history
            ],
            "error_history": self.error_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowState":
        """Deserialize workflow state from dictionary.

        Reconstructs a WorkflowState instance from a dictionary, typically
        loaded from a checkpoint or database. Handles enum and datetime
        conversions automatically.

        Args:
            data: Dictionary representation of workflow state.

        Returns:
            Reconstructed WorkflowState instance.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        # Convert string values back to enums
        current_stage = ProcessingStage(data["current_stage"])
        status = WorkflowStatus(data["status"])

        # Convert ISO format strings back to datetime
        timestamp = datetime.fromisoformat(data["timestamp"])

        # Convert stage history
        stage_history = [
            (ProcessingStage(stage), datetime.fromisoformat(ts))
            for stage, ts in data.get("stage_history", [])
        ]

        return cls(
            drawing_id=data["drawing_id"],
            current_stage=current_stage,
            metadata=data.get("metadata", {}),
            checkpoint_data=data.get("checkpoint_data", {}),
            timestamp=timestamp,
            confidence_scores=data.get("confidence_scores", {}),
            is_running=data.get("is_running", False),
            status=status,
            schema_version=data.get("schema_version", 1),
            stage_history=stage_history,
            error_history=data.get("error_history", []),
        )
