"""
Drawing Intelligence System

Automated extraction of structured data from engineering drawings.
"""

__version__ = "1.0.0"
__author__ = "Drawing Intelligence Team"

# Core exports
from .orchestration import PipelineOrchestrator
from .database import DatabaseManager
from .models import ProcessingResult
from .export import ExportManager

__all__ = [
    "PipelineOrchestrator",
    "DatabaseManager",
    "ProcessingResult",
    "ExportManager",
    "__version__",
]
