"""
Error handling utilities for the Drawing Intelligence System.

Provides custom exceptions and error handling functions.
"""

import logging
import traceback
from typing import Dict, Any, Tuple, Optional


class DrawingProcessingError(Exception):
    """
    Base exception for drawing processing errors.

    Attributes:
        message: Error message
        drawing_id: Optional drawing ID
        stage: Optional processing stage where error occurred
        recoverable: Whether error is recoverable
    """

    def __init__(
        self,
        message: str,
        drawing_id: Optional[str] = None,
        stage: Optional[str] = None,
        recoverable: bool = False,
    ):
        self.message = message
        self.drawing_id = drawing_id
        self.stage = stage
        self.recoverable = recoverable
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/storage."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "drawing_id": self.drawing_id,
            "stage": self.stage,
            "recoverable": self.recoverable,
        }


class PDFProcessingError(DrawingProcessingError):
    """Exception for PDF-specific processing errors."""

    def __init__(
        self,
        message: str,
        drawing_id: Optional[str] = None,
        page_number: Optional[int] = None,
        recoverable: bool = False,
    ):
        super().__init__(
            message=message,
            drawing_id=drawing_id,
            stage="pdf_processing",
            recoverable=recoverable,
        )
        self.page_number = page_number

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["page_number"] = self.page_number
        return result


class OCRError(DrawingProcessingError):
    """Exception for OCR-specific errors."""

    def __init__(
        self,
        message: str,
        drawing_id: Optional[str] = None,
        ocr_engine: Optional[str] = None,
        recoverable: bool = True,  # OCR errors often recoverable with fallback
    ):
        super().__init__(
            message=message,
            drawing_id=drawing_id,
            stage="ocr_extraction",
            recoverable=recoverable,
        )
        self.ocr_engine = ocr_engine

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["ocr_engine"] = self.ocr_engine
        return result


class ShapeDetectionError(DrawingProcessingError):
    """Exception for shape detection errors."""

    def __init__(
        self,
        message: str,
        drawing_id: Optional[str] = None,
        model_version: Optional[str] = None,
        recoverable: bool = False,
    ):
        super().__init__(
            message=message,
            drawing_id=drawing_id,
            stage="shape_detection",
            recoverable=recoverable,
        )
        self.model_version = model_version

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["model_version"] = self.model_version
        return result


class LLMAPIError(DrawingProcessingError):
    """
    Exception for LLM API call errors.

    Additional attributes:
        provider: LLM provider name
        status_code: HTTP status code (if applicable)
    """

    def __init__(
        self,
        message: str,
        drawing_id: Optional[str] = None,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        recoverable: bool = True,  # API errors often recoverable with retry
    ):
        super().__init__(
            message=message,
            drawing_id=drawing_id,
            stage="llm_enhancement",
            recoverable=recoverable,
        )
        self.provider = provider
        self.status_code = status_code

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["provider"] = self.provider
        result["status_code"] = self.status_code
        return result


class BudgetExceededException(DrawingProcessingError):
    """
    Exception for budget limit exceeded.

    Additional attributes:
        current_cost: Current cost in USD
        budget_limit: Budget limit in USD
    """

    def __init__(
        self,
        message: str,
        current_cost: float,
        budget_limit: float,
        drawing_id: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            drawing_id=drawing_id,
            stage="llm_enhancement",
            recoverable=False,  # Budget exceeded is not recoverable
        )
        self.current_cost = current_cost
        self.budget_limit = budget_limit

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["current_cost"] = self.current_cost
        result["budget_limit"] = self.budget_limit
        return result


class DatabaseError(DrawingProcessingError):
    """Exception for database operation errors."""

    def __init__(
        self,
        message: str,
        drawing_id: Optional[str] = None,
        operation: Optional[str] = None,
        recoverable: bool = True,
    ):
        super().__init__(
            message=message,
            drawing_id=drawing_id,
            stage="database",
            recoverable=recoverable,
        )
        self.operation = operation

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["operation"] = self.operation
        return result


class ConfigurationError(DrawingProcessingError):
    """Exception for configuration errors."""

    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message=message,
            stage="initialization",
            recoverable=False,  # Config errors not recoverable without fix
        )
        self.config_key = config_key

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["config_key"] = self.config_key
        return result


class ValidationError(DrawingProcessingError):
    """Exception for data validation errors."""

    def __init__(
        self,
        message: str,
        drawing_id: Optional[str] = None,
        field_name: Optional[str] = None,
        recoverable: bool = False,
    ):
        super().__init__(
            message=message,
            drawing_id=drawing_id,
            stage="validation",
            recoverable=recoverable,
        )
        self.field_name = field_name

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["field_name"] = self.field_name
        return result


def handle_processing_error(
    error: Exception, context: Dict[str, Any], logger: Optional[logging.Logger] = None
) -> Tuple[bool, str]:
    """
    Generic error handler with logging and recovery decision.

    Args:
        error: Exception that occurred
        context: Dictionary with context (drawing_id, stage, etc.)
        logger: Optional logger instance

    Returns:
        Tuple of (should_retry, error_message)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Log error with context
    log_error_with_context(error, logger, context)

    # Determine if error is recoverable
    should_retry = False
    error_message = str(error)

    if isinstance(error, DrawingProcessingError):
        should_retry = error.recoverable
        error_message = error.message
    elif isinstance(error, (ConnectionError, TimeoutError)):
        # Network errors are recoverable
        should_retry = True
        error_message = f"Network error: {error}"
    elif isinstance(error, MemoryError):
        # Memory errors not recoverable
        should_retry = False
        error_message = f"Out of memory: {error}"
    elif isinstance(error, KeyboardInterrupt):
        # User interrupt not recoverable
        should_retry = False
        error_message = "Processing interrupted by user"
    else:
        # Unknown errors - don't retry by default
        should_retry = False
        error_message = f"Unexpected error: {error}"

    return should_retry, error_message


def log_error_with_context(
    error: Exception, logger: logging.Logger, context: Dict[str, Any]
) -> None:
    """
    Log error with full context for debugging.

    Args:
        error: Exception that occurred
        logger: Logger instance
        context: Dictionary with context information
    """
    # Build error message
    error_type = type(error).__name__
    error_message = str(error)

    # Get context details
    drawing_id = context.get("drawing_id", "unknown")
    stage = context.get("stage", "unknown")

    # Log error
    logger.error(
        f"Error in {stage} for drawing {drawing_id}: [{error_type}] {error_message}"
    )

    # Log additional context
    for key, value in context.items():
        if key not in ["drawing_id", "stage"]:
            logger.error(f"  {key}: {value}")

    # Log stack trace for debugging
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Stack trace:")
        logger.debug(traceback.format_exc())


def create_error_report(
    error: Exception, processing_result: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Create structured error report for storage/analysis.

    Args:
        error: Exception that occurred
        processing_result: Optional partial processing result

    Returns:
        Error report dictionary
    """
    report = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "timestamp": None,  # Will be set by caller
    }

    # Add custom error attributes if available
    if isinstance(error, DrawingProcessingError):
        report.update(error.to_dict())

    # Add partial results if available
    if processing_result:
        report["partial_results"] = {
            "drawing_id": getattr(processing_result, "drawing_id", None),
            "completed_stages": getattr(processing_result, "completed_stages", []),
            "overall_confidence": getattr(
                processing_result, "overall_confidence", None
            ),
        }

    return report


def wrap_with_error_handling(func):
    """
    Decorator to add error handling to functions.

    Usage:
        @wrap_with_error_handling
        def my_function():
            ...
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DrawingProcessingError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            # Wrap unknown errors
            raise DrawingProcessingError(
                message=f"Unexpected error in {func.__name__}: {e}", recoverable=False
            ) from e

    return wrapper


def is_retriable_error(error: Exception) -> bool:
    """
    Determine if an error should trigger a retry.

    Args:
        error: Exception to check

    Returns:
        True if error is retriable
    """
    # DrawingProcessingError has recoverable flag
    if isinstance(error, DrawingProcessingError):
        return error.recoverable

    # Network-related errors are retriable
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return True

    # Rate limit errors (HTTP 429) are retriable
    if hasattr(error, "status_code") and error.status_code == 429:
        return True

    # Default: not retriable
    return False


def get_retry_delay(attempt: int, base_delay: float = 1.0) -> float:
    """
    Calculate retry delay using exponential backoff.

    Args:
        attempt: Retry attempt number (1-indexed)
        base_delay: Base delay in seconds

    Returns:
        Delay in seconds
    """
    # Exponential backoff: base_delay * 2^(attempt-1)
    delay = base_delay * (2 ** (attempt - 1))
    # Cap at 60 seconds
    return min(delay, 60.0)
