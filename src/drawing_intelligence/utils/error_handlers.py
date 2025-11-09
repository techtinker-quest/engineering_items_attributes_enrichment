"""
Error handling utilities for the Drawing Intelligence System.

This module provides custom exceptions and error handling functions for robust
error management throughout the drawing processing pipeline.

Classes:
    DrawingProcessingError: Base exception for all drawing processing errors.
    PDFProcessingError: Exception for PDF-specific processing errors.
    OCRError: Exception for OCR-specific errors.
    ShapeDetectionError: Exception for shape detection errors.
    LLMAPIError: Exception for LLM API call errors.
    BudgetExceededException: Exception for budget limit exceeded scenarios.
    DatabaseError: Exception for database operation errors.
    ConfigurationError: Exception for configuration errors.
    ValidationError: Exception for data validation errors.

Functions:
    handle_processing_error: Generic error handler with logging and recovery decision.
    log_error_with_context: Log error with full context for debugging.
    create_error_report: Create structured error report for storage/analysis.
    wrap_with_error_handling: Decorator to add error handling to functions.
    is_retriable_error: Determine if an error should trigger a retry.
    get_retry_delay: Calculate retry delay using exponential backoff.
"""

import functools
import logging
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple


class DrawingProcessingError(Exception):
    """
    Base exception for drawing processing errors.

    Attributes:
        message: Error message describing what went wrong.
        drawing_id: Optional identifier of the drawing being processed.
        stage: Optional processing stage where the error occurred.
        recoverable: Whether the error is recoverable with retry.
        original_error: Optional underlying exception that was wrapped.
    """

    def __init__(
        self,
        message: str,
        drawing_id: Optional[str] = None,
        stage: Optional[str] = None,
        recoverable: bool = False,
        original_error: Optional[Exception] = None,
    ):
        """
        Initialize DrawingProcessingError.

        Args:
            message: Error message describing the issue.
            drawing_id: Optional drawing identifier.
            stage: Optional processing stage name.
            recoverable: Whether error can be recovered with retry.
            original_error: Optional underlying exception that caused this error.
        """
        self.message = message
        self.drawing_id = drawing_id
        self.stage = stage
        self.recoverable = recoverable
        self.original_error = original_error
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for logging and storage.

        Returns:
            Dictionary containing error_type, message, drawing_id, stage,
            recoverable status, and original error information if available.
        """
        result = {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "drawing_id": self.drawing_id,
            "stage": self.stage,
            "recoverable": self.recoverable,
        }

        if self.original_error:
            result["original_error_type"] = type(self.original_error).__name__
            result["original_error_message"] = str(self.original_error)

        return result


class PDFProcessingError(DrawingProcessingError):
    """
    Exception for PDF-specific processing errors.

    Attributes:
        page_number: Optional page number where error occurred.
    """

    def __init__(
        self,
        message: str,
        drawing_id: Optional[str] = None,
        page_number: Optional[int] = None,
        recoverable: bool = False,
        original_error: Optional[Exception] = None,
    ):
        """
        Initialize PDFProcessingError.

        Args:
            message: Error message describing the PDF issue.
            drawing_id: Optional drawing identifier.
            page_number: Optional PDF page number where error occurred.
            recoverable: Whether error can be recovered with retry.
            original_error: Optional underlying exception that caused this error.
        """
        super().__init__(
            message=message,
            drawing_id=drawing_id,
            stage="pdf_processing",
            recoverable=recoverable,
            original_error=original_error,
        )
        self.page_number = page_number

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary including page number.

        Returns:
            Dictionary with all base fields plus page_number.
        """
        result = super().to_dict()
        result["page_number"] = self.page_number
        return result


class OCRError(DrawingProcessingError):
    """
    Exception for OCR-specific errors.

    Attributes:
        ocr_engine: Optional OCR engine name that failed.
    """

    def __init__(
        self,
        message: str,
        drawing_id: Optional[str] = None,
        ocr_engine: Optional[str] = None,
        recoverable: bool = True,  # OCR errors often recoverable with fallback
        original_error: Optional[Exception] = None,
    ):
        """
        Initialize OCRError.

        Args:
            message: Error message describing the OCR issue.
            drawing_id: Optional drawing identifier.
            ocr_engine: Optional name of OCR engine that failed.
            recoverable: Whether error can be recovered (defaults to True).
            original_error: Optional underlying exception that caused this error.
        """
        super().__init__(
            message=message,
            drawing_id=drawing_id,
            stage="ocr_extraction",
            recoverable=recoverable,
            original_error=original_error,
        )
        self.ocr_engine = ocr_engine

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary including OCR engine.

        Returns:
            Dictionary with all base fields plus ocr_engine.
        """
        result = super().to_dict()
        result["ocr_engine"] = self.ocr_engine
        return result


class ShapeDetectionError(DrawingProcessingError):
    """
    Exception for shape detection errors.

    Attributes:
        model_version: Optional version of detection model that failed.
    """

    def __init__(
        self,
        message: str,
        drawing_id: Optional[str] = None,
        model_version: Optional[str] = None,
        recoverable: bool = False,
        original_error: Optional[Exception] = None,
    ):
        """
        Initialize ShapeDetectionError.

        Args:
            message: Error message describing the detection issue.
            drawing_id: Optional drawing identifier.
            model_version: Optional version of YOLO model that failed.
            recoverable: Whether error can be recovered with retry.
            original_error: Optional underlying exception that caused this error.
        """
        super().__init__(
            message=message,
            drawing_id=drawing_id,
            stage="shape_detection",
            recoverable=recoverable,
            original_error=original_error,
        )
        self.model_version = model_version

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary including model version.

        Returns:
            Dictionary with all base fields plus model_version.
        """
        result = super().to_dict()
        result["model_version"] = self.model_version
        return result


class LLMAPIError(DrawingProcessingError):
    """
    Exception for LLM API call errors.

    Attributes:
        provider: LLM provider name (openai, anthropic, google).
        status_code: HTTP status code if applicable.
    """

    def __init__(
        self,
        message: str,
        drawing_id: Optional[str] = None,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        recoverable: bool = True,  # API errors often recoverable with retry
        original_error: Optional[Exception] = None,
    ):
        """
        Initialize LLMAPIError.

        Args:
            message: Error message describing the API issue.
            drawing_id: Optional drawing identifier.
            provider: Optional LLM provider name.
            status_code: Optional HTTP status code.
            recoverable: Whether error can be recovered (defaults to True).
            original_error: Optional underlying exception that caused this error.
        """
        super().__init__(
            message=message,
            drawing_id=drawing_id,
            stage="llm_enhancement",
            recoverable=recoverable,
            original_error=original_error,
        )
        self.provider = provider
        self.status_code = status_code

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary including provider and status.

        Returns:
            Dictionary with all base fields plus provider and status_code.
        """
        result = super().to_dict()
        result["provider"] = self.provider
        result["status_code"] = self.status_code
        return result


class BudgetExceededException(DrawingProcessingError):
    """
    Exception for budget limit exceeded scenarios.

    Attributes:
        current_cost: Current accumulated cost in USD.
        budget_limit: Maximum allowed budget in USD.
    """

    def __init__(
        self,
        message: str,
        current_cost: float,
        budget_limit: float,
        drawing_id: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        Initialize BudgetExceededException.

        Args:
            message: Error message describing budget issue.
            current_cost: Current cost in USD.
            budget_limit: Budget limit in USD.
            drawing_id: Optional drawing identifier.
            original_error: Optional underlying exception that caused this error.
        """
        super().__init__(
            message=message,
            drawing_id=drawing_id,
            stage="llm_enhancement",
            recoverable=False,  # Budget exceeded is not recoverable
            original_error=original_error,
        )
        self.current_cost = current_cost
        self.budget_limit = budget_limit

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary including cost information.

        Returns:
            Dictionary with all base fields plus current_cost and budget_limit.
        """
        result = super().to_dict()
        result["current_cost"] = self.current_cost
        result["budget_limit"] = self.budget_limit
        return result


class DatabaseError(DrawingProcessingError):
    """
    Exception for database operation errors.

    Attributes:
        operation: Optional database operation that failed.
    """

    def __init__(
        self,
        message: str,
        drawing_id: Optional[str] = None,
        operation: Optional[str] = None,
        recoverable: bool = True,
        original_error: Optional[Exception] = None,
    ):
        """
        Initialize DatabaseError.

        Args:
            message: Error message describing database issue.
            drawing_id: Optional drawing identifier.
            operation: Optional database operation (insert, update, query).
            recoverable: Whether error can be recovered with retry.
            original_error: Optional underlying database driver exception.
        """
        super().__init__(
            message=message,
            drawing_id=drawing_id,
            stage="database",
            recoverable=recoverable,
            original_error=original_error,
        )
        self.operation = operation

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary including operation.

        Returns:
            Dictionary with all base fields plus operation.
        """
        result = super().to_dict()
        result["operation"] = self.operation
        return result


class ConfigurationError(DrawingProcessingError):
    """
    Exception for configuration errors.

    Attributes:
        config_key: Optional configuration key that caused the error.
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        Initialize ConfigurationError.

        Args:
            message: Error message describing configuration issue.
            config_key: Optional configuration key that is invalid.
            original_error: Optional underlying exception that caused this error.
        """
        super().__init__(
            message=message,
            stage="initialization",
            recoverable=False,  # Config errors not recoverable without fix
            original_error=original_error,
        )
        self.config_key = config_key

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary including config key.

        Returns:
            Dictionary with all base fields plus config_key.
        """
        result = super().to_dict()
        result["config_key"] = self.config_key
        return result


class ValidationError(DrawingProcessingError):
    """
    Exception for data validation errors.

    Attributes:
        field_name: Optional field name that failed validation.
    """

    def __init__(
        self,
        message: str,
        drawing_id: Optional[str] = None,
        field_name: Optional[str] = None,
        recoverable: bool = False,
        original_error: Optional[Exception] = None,
    ):
        """
        Initialize ValidationError.

        Args:
            message: Error message describing validation issue.
            drawing_id: Optional drawing identifier.
            field_name: Optional field name that failed validation.
            recoverable: Whether error can be recovered with retry.
            original_error: Optional underlying exception that caused this error.
        """
        super().__init__(
            message=message,
            drawing_id=drawing_id,
            stage="validation",
            recoverable=recoverable,
            original_error=original_error,
        )
        self.field_name = field_name

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary including field name.

        Returns:
            Dictionary with all base fields plus field_name.
        """
        result = super().to_dict()
        result["field_name"] = self.field_name
        return result


def handle_processing_error(
    error: Exception, context: Dict[str, Any], logger: Optional[logging.Logger] = None
) -> Tuple[bool, str]:
    """
    Handle processing errors with logging and recovery decision.

    Analyzes the exception type to determine if the operation should be retried
    and logs appropriate error information with context.

    Args:
        error: The exception that occurred during processing.
        context: Dictionary containing contextual information such as drawing_id,
            stage, and other relevant metadata.
        logger: Optional logger instance. If None, uses module logger.

    Returns:
        A tuple containing:
            - bool: True if the error is recoverable and should be retried.
            - str: Human-readable error message.

    Example:
        >>> context = {"drawing_id": "DWG-001", "stage": "ocr_extraction"}
        >>> should_retry, msg = handle_processing_error(OCRError("Failed"), context)
        >>> print(f"Retry: {should_retry}, Message: {msg}")
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Log error with context
    log_error_with_context(error, logger, context)

    # Determine if error is recoverable using the enhanced retry logic
    should_retry = is_retriable_error(error)

    # Build error message
    if isinstance(error, DrawingProcessingError):
        error_message = error.message
    elif isinstance(error, (ConnectionError, TimeoutError)):
        error_message = f"Network error: {error}"
    elif isinstance(error, MemoryError):
        error_message = f"Out of memory: {error}"
    elif isinstance(error, KeyboardInterrupt):
        error_message = "Processing interrupted by user"
    elif hasattr(error, "status_code") and error.status_code == 429:
        error_message = f"Rate limit exceeded: {error}"
    else:
        error_message = f"Unexpected error: {error}"

    return should_retry, error_message


def log_error_with_context(
    error: Exception, logger: logging.Logger, context: Dict[str, Any]
) -> None:
    """
    Log error with comprehensive context information for debugging.

    Logs error details including type, message, and all contextual information.
    In DEBUG mode, also logs the full stack trace.

    Args:
        error: The exception that occurred.
        logger: Logger instance to use for logging.
        context: Dictionary with contextual information (drawing_id, stage, etc.).

    Returns:
        None

    Note:
        Stack traces are only logged when logger is at DEBUG level or lower.
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

    # Log original error if available
    if isinstance(error, DrawingProcessingError) and error.original_error:
        original_type = type(error.original_error).__name__
        original_msg = str(error.original_error)
        logger.error(f"  Original error: [{original_type}] {original_msg}")

    # Log additional context
    for key, value in context.items():
        if key not in ["drawing_id", "stage"]:
            logger.error(f"  {key}: {value}")

    # Log stack trace for debugging
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Stack trace:")
        logger.debug(traceback.format_exc())


def create_error_report(
    error: Exception,
    processing_result: Optional[Any] = None,
    timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Create a structured error report for storage and analysis.

    Generates a comprehensive error report including error type, message,
    stack trace, and optional partial processing results.

    Args:
        error: The exception that occurred.
        processing_result: Optional partial processing result object that may
            contain drawing_id, completed_stages, and overall_confidence.
        timestamp: Optional timestamp for the error. Defaults to current time.

    Returns:
        A dictionary containing:
            - error_type: Name of the exception class.
            - error_message: String representation of the error.
            - traceback: Full stack trace string.
            - timestamp: ISO format timestamp of the error.
            - Additional fields from DrawingProcessingError if applicable.
            - partial_results: Dictionary of partial results if available.

    Example:
        >>> error = OCRError("PaddleOCR failed", drawing_id="DWG-001")
        >>> report = create_error_report(error)
        >>> print(report['error_type'])
        'OCRError'
    """
    if timestamp is None:
        timestamp = datetime.now()

    report = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "timestamp": timestamp.isoformat(),
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


def wrap_with_error_handling(func: Callable) -> Callable:
    """
    Decorator to add standardized error handling to functions.

    Wraps functions to catch and re-raise DrawingProcessingError instances,
    while converting unknown exceptions to DrawingProcessingError with
    recoverable=False.

    Args:
        func: The function to wrap with error handling.

    Returns:
        The wrapped function with error handling.

    Raises:
        DrawingProcessingError: For both custom and wrapped unknown errors.

    Example:
        >>> @wrap_with_error_handling
        ... def risky_operation():
        ...     raise ValueError("Something went wrong")
        >>> risky_operation()  # Raises DrawingProcessingError
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except DrawingProcessingError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            # Wrap unknown errors, preserving the original
            raise DrawingProcessingError(
                message=f"Unexpected error in {func.__name__}: {e}",
                recoverable=False,
                original_error=e,
            ) from e

    return wrapper


def is_retriable_error(error: Exception) -> bool:
    """
    Determine if an error should trigger a retry attempt.

    Evaluates error types and attributes to decide if the operation that
    caused the error can reasonably be retried. For DrawingProcessingError
    instances, inspects the original_error for more intelligent decisions.

    Args:
        error: The exception to evaluate.

    Returns:
        True if the error is retriable (network issues, rate limits, or
        DrawingProcessingError with recoverable=True), False otherwise.

    Note:
        - DrawingProcessingError instances use their recoverable flag.
        - For DatabaseError, inspects original_error for transient vs permanent issues.
        - Network errors (ConnectionError, TimeoutError, OSError) are retriable.
        - HTTP 429 (rate limit) errors are retriable.
        - All other errors default to non-retriable.

    Example:
        >>> error = OCRError("Temporary failure", recoverable=True)
        >>> is_retriable_error(error)
        True
        >>> import sqlite3
        >>> db_error = DatabaseError(
        ...     "Database locked",
        ...     original_error=sqlite3.OperationalError("database is locked")
        ... )
        >>> is_retriable_error(db_error)
        True
    """
    # DrawingProcessingError with enhanced original error inspection
    if isinstance(error, DrawingProcessingError):
        # First, check if there's an original error to inspect
        if error.original_error:
            original_str = str(error.original_error).lower()
            original_type = type(error.original_error).__name__.lower()

            # Database-specific transient errors (recoverable)
            transient_db_keywords = [
                "locked",
                "timeout",
                "connection",
                "busy",
                "temporary",
                "deadlock",
                "operational",
            ]
            if any(
                keyword in original_str or keyword in original_type
                for keyword in transient_db_keywords
            ):
                return True

            # Database-specific permanent errors (non-recoverable)
            permanent_db_keywords = [
                "integrity",
                "constraint",
                "foreign key",
                "unique",
                "not null",
                "check constraint",
            ]
            if any(
                keyword in original_str or keyword in original_type
                for keyword in permanent_db_keywords
            ):
                return False

            # API-specific transient errors
            api_transient_keywords = [
                "rate limit",
                "too many requests",
                "service unavailable",
                "gateway timeout",
                "bad gateway",
            ]
            if any(keyword in original_str for keyword in api_transient_keywords):
                return True

        # Fall back to the recoverable flag
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
    Calculate retry delay using exponential backoff with cap.

    Implements exponential backoff strategy: delay = base_delay * 2^(attempt-1),
    capped at 60 seconds to prevent excessive wait times.

    Args:
        attempt: Current retry attempt number (1-indexed). First retry is 1.
        base_delay: Base delay in seconds for the first retry. Defaults to 1.0.

    Returns:
        Calculated delay in seconds, capped at 60.0 seconds.

    Example:
        >>> get_retry_delay(1)  # First retry
        1.0
        >>> get_retry_delay(5)  # Fifth retry
        16.0
        >>> get_retry_delay(10)  # Would be 512, but capped
        60.0
    """
    # Exponential backoff: base_delay * 2^(attempt-1)
    delay = base_delay * (2 ** (attempt - 1))
    # Cap at 60 seconds
    return min(delay, 60.0)
