# orchestration/checkpoint_manager.py
"""
Dual checkpoint system:
1. Intra-drawing checkpoints: Save state after each processing stage
2. Batch checkpoints: Save state every N drawings in batch processing
"""
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Processing stages for intra-drawing checkpoints"""

    PDF_PROCESSING = "pdf_processing"
    IMAGE_PREPROCESSING = "image_preprocessing"
    OCR_EXTRACTION = "ocr_extraction"
    ENTITY_EXTRACTION = "entity_extraction"
    SHAPE_DETECTION = "shape_detection"
    DATA_ASSOCIATION = "data_association"
    QUALITY_SCORING = "quality_scoring"
    COMPLETE = "complete"


@dataclass
class IntraDrawingCheckpoint:
    """Checkpoint for a single drawing's processing state"""

    drawing_id: str
    source_file: str
    current_stage: ProcessingStage
    completed_stages: List[ProcessingStage]
    intermediate_results: Dict[str, Any]
    timestamp: str
    checkpoint_id: str

    def to_dict(self) -> Dict:
        data = asdict(self)
        # Convert enums to strings
        data["current_stage"] = self.current_stage.value
        data["completed_stages"] = [s.value for s in self.completed_stages]
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "IntraDrawingCheckpoint":
        # Convert strings back to enums
        data["current_stage"] = ProcessingStage(data["current_stage"])
        data["completed_stages"] = [
            ProcessingStage(s) for s in data["completed_stages"]
        ]
        return cls(**data)


@dataclass
class BatchCheckpoint:
    """Checkpoint for batch processing state"""

    batch_id: str
    total_drawings: int
    completed_count: int
    failed_count: int
    completed_drawing_ids: List[str]
    failed_drawing_ids: List[str]
    pending_drawing_ids: List[str]
    current_drawing_id: Optional[str]
    timestamp: str
    checkpoint_id: str

    # Statistics
    total_llm_cost: float
    average_processing_time_seconds: float

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "BatchCheckpoint":
        return cls(**data)


class CheckpointManager:
    """
    Manages both intra-drawing and batch-level checkpoints
    """

    def __init__(self, checkpoint_dir: str = "/tmp/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Separate directories for different checkpoint types
        self.intra_drawing_dir = self.checkpoint_dir / "intra_drawing"
        self.batch_dir = self.checkpoint_dir / "batch"

        self.intra_drawing_dir.mkdir(exist_ok=True)
        self.batch_dir.mkdir(exist_ok=True)

    # ============ INTRA-DRAWING CHECKPOINTS ============

    def save_intra_drawing_checkpoint(
        self,
        drawing_id: str,
        source_file: str,
        current_stage: ProcessingStage,
        completed_stages: List[ProcessingStage],
        intermediate_results: Dict[str, Any],
    ) -> str:
        """
        Save checkpoint after completing a stage within a drawing.
        Useful for resuming long-running drawings that crash mid-processing.

        Returns:
            checkpoint_id
        """
        checkpoint_id = (
            f"{drawing_id}_{current_stage.value}_{int(datetime.now().timestamp())}"
        )

        checkpoint = IntraDrawingCheckpoint(
            drawing_id=drawing_id,
            source_file=source_file,
            current_stage=current_stage,
            completed_stages=completed_stages,
            intermediate_results=intermediate_results,
            timestamp=datetime.now().isoformat(),
            checkpoint_id=checkpoint_id,
        )

        checkpoint_path = self.intra_drawing_dir / f"{checkpoint_id}.json"

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        logger.info(
            f"Intra-drawing checkpoint saved: {checkpoint_id} "
            f"(stage: {current_stage.value})"
        )

        return checkpoint_id

    def load_intra_drawing_checkpoint(
        self, drawing_id: str
    ) -> Optional[IntraDrawingCheckpoint]:
        """
        Load the most recent checkpoint for a drawing.
        Returns None if no checkpoint exists.
        """
        # Find all checkpoints for this drawing
        pattern = f"{drawing_id}_*.json"
        checkpoints = list(self.intra_drawing_dir.glob(pattern))

        if not checkpoints:
            return None

        # Get most recent
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)

        with open(latest_checkpoint, "r") as f:
            data = json.load(f)

        checkpoint = IntraDrawingCheckpoint.from_dict(data)
        logger.info(
            f"Loaded intra-drawing checkpoint: {checkpoint.checkpoint_id} "
            f"(stage: {checkpoint.current_stage.value})"
        )

        return checkpoint

    def delete_intra_drawing_checkpoints(self, drawing_id: str):
        """Delete all checkpoints for a drawing (after successful completion)"""
        pattern = f"{drawing_id}_*.json"
        for checkpoint_file in self.intra_drawing_dir.glob(pattern):
            checkpoint_file.unlink()
        logger.info(f"Deleted intra-drawing checkpoints for {drawing_id}")

    # ============ BATCH CHECKPOINTS ============

    def save_batch_checkpoint(
        self,
        batch_id: str,
        total_drawings: int,
        completed_drawing_ids: List[str],
        failed_drawing_ids: List[str],
        pending_drawing_ids: List[str],
        current_drawing_id: Optional[str],
        total_llm_cost: float,
        average_processing_time: float,
    ) -> str:
        """
        Save batch processing state every N drawings.
        Enables resuming batch jobs that are interrupted.

        Returns:
            checkpoint_id
        """
        checkpoint_id = f"{batch_id}_{int(datetime.now().timestamp())}"

        checkpoint = BatchCheckpoint(
            batch_id=batch_id,
            total_drawings=total_drawings,
            completed_count=len(completed_drawing_ids),
            failed_count=len(failed_drawing_ids),
            completed_drawing_ids=completed_drawing_ids,
            failed_drawing_ids=failed_drawing_ids,
            pending_drawing_ids=pending_drawing_ids,
            current_drawing_id=current_drawing_id,
            timestamp=datetime.now().isoformat(),
            checkpoint_id=checkpoint_id,
            total_llm_cost=total_llm_cost,
            average_processing_time_seconds=average_processing_time,
        )

        checkpoint_path = self.batch_dir / f"{checkpoint_id}.json"

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        logger.info(
            f"Batch checkpoint saved: {checkpoint_id} "
            f"({checkpoint.completed_count}/{total_drawings} completed)"
        )

        return checkpoint_id

    def load_batch_checkpoint(self, batch_id: str) -> Optional[BatchCheckpoint]:
        """
        Load the most recent batch checkpoint.
        Returns None if no checkpoint exists.
        """
        # Find all checkpoints for this batch
        pattern = f"{batch_id}_*.json"
        checkpoints = list(self.batch_dir.glob(pattern))

        if not checkpoints:
            return None

        # Get most recent
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)

        with open(latest_checkpoint, "r") as f:
            data = json.load(f)

        checkpoint = BatchCheckpoint.from_dict(data)
        logger.info(
            f"Loaded batch checkpoint: {checkpoint.checkpoint_id} "
            f"({checkpoint.completed_count}/{checkpoint.total_drawings} completed)"
        )

        return checkpoint

    def delete_batch_checkpoints(self, batch_id: str):
        """Delete all checkpoints for a batch (after completion)"""
        pattern = f"{batch_id}_*.json"
        for checkpoint_file in self.batch_dir.glob(pattern):
            checkpoint_file.unlink()
        logger.info(f"Deleted batch checkpoints for {batch_id}")

    def list_available_checkpoints(self) -> Dict[str, List[str]]:
        """List all available checkpoints"""
        intra_checkpoints = [f.stem for f in self.intra_drawing_dir.glob("*.json")]
        batch_checkpoints = [f.stem for f in self.batch_dir.glob("*.json")]

        return {"intra_drawing": intra_checkpoints, "batch": batch_checkpoints}


# Example usage demonstrating both checkpoint types:
if __name__ == "__main__":
    manager = CheckpointManager()

    # === INTRA-DRAWING CHECKPOINT EXAMPLE ===
    print("=== Intra-Drawing Checkpoint Example ===")

    # Simulate processing a drawing stage-by-stage
    drawing_id = "DWG-001"
    completed_stages = []
    intermediate_results = {}

    # After PDF processing
    completed_stages.append(ProcessingStage.PDF_PROCESSING)
    intermediate_results["pdf_pages"] = 3
    manager.save_intra_drawing_checkpoint(
        drawing_id=drawing_id,
        source_file="drawing001.pdf",
        current_stage=ProcessingStage.PDF_PROCESSING,
        completed_stages=completed_stages.copy(),
        intermediate_results=intermediate_results.copy(),
    )

    # After OCR
    completed_stages.append(ProcessingStage.OCR_EXTRACTION)
    intermediate_results["text_blocks"] = 45
    manager.save_intra_drawing_checkpoint(
        drawing_id=drawing_id,
        source_file="drawing001.pdf",
        current_stage=ProcessingStage.OCR_EXTRACTION,
        completed_stages=completed_stages.copy(),
        intermediate_results=intermediate_results.copy(),
    )

    # Resume from checkpoint
    checkpoint = manager.load_intra_drawing_checkpoint(drawing_id)
    if checkpoint:
        print(f"Resumed from stage: {checkpoint.current_stage.value}")
        print(f"Completed stages: {[s.value for s in checkpoint.completed_stages]}")

    # === BATCH CHECKPOINT EXAMPLE ===
    print("\n=== Batch Checkpoint Example ===")

    # Simulate batch processing
    batch_id = "batch_20251101"
    all_drawings = ["DWG-001", "DWG-002", "DWG-003", "DWG-004", "DWG-005"]

    # After processing 2 drawings
    manager.save_batch_checkpoint(
        batch_id=batch_id,
        total_drawings=len(all_drawings),
        completed_drawing_ids=["DWG-001", "DWG-002"],
        failed_drawing_ids=[],
        pending_drawing_ids=["DWG-003", "DWG-004", "DWG-005"],
        current_drawing_id="DWG-003",
        total_llm_cost=0.12,
        average_processing_time=45.3,
    )

    # Resume batch
    batch_checkpoint = manager.load_batch_checkpoint(batch_id)
    if batch_checkpoint:
        print(
            f"Batch progress: {batch_checkpoint.completed_count}/{batch_checkpoint.total_drawings}"
        )
        print(f"Resume from: {batch_checkpoint.pending_drawing_ids[0]}")
        print(f"LLM cost so far: ${batch_checkpoint.total_llm_cost:.2f}")

    # Cleanup
    manager.delete_intra_drawing_checkpoints(drawing_id)
    manager.delete_batch_checkpoints(batch_id)
