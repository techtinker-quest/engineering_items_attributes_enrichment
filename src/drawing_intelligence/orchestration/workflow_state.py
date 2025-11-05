# src/drawing_intelligence/orchestration/workflow_state.py
# Module to manage the state of a drawing's processing workflow
# Defines WorkflowState dataclass to hold state information
# Includes methods to update state and checkpoint data
# Author: GROK AI + Sandeep A (01Nov2025)


from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime


@dataclass
class WorkflowState:
    """Manages the state of a drawing's processing pipeline."""

    drawing_id: str
    current_stage: str
    metadata: Dict[str, str]
    checkpoint_data: Dict[str, any]
    timestamp: datetime
    confidence_scores: Optional[Dict[str, float]] = None

    def update_stage(self, stage: str) -> None:
        """Update the current processing stage."""
        self.current_stage = stage
        self.timestamp = datetime.now()

    def save_checkpoint(self, data: Dict[str, any]) -> None:
        """Save checkpoint data for the current stage."""
        self.checkpoint_data.update(data)
