"""Export Manager Module

Manages all export operations for drawing intelligence results, coordinating
JSON, CSV, and report generation through specialized exporter classes.

This module provides a unified interface for exporting processing results in
multiple formats, including single drawing exports, batch exports, and various
report types (drawing, batch, cost).

Classes:
    JSONFormat: Enum for JSON formatting options.
    ReportFormat: Enum for report output formats.
    ExportConfig: Configuration for export operations.
    ExportManager: Coordinates all export operations.

Exceptions:
    ExportError: Base exception for export operations.
    DrawingNotFoundError: Raised when a drawing is not found in database.
    ExportFileError: Raised when file write operations fail.
    InvalidConfigError: Raised when export configuration is invalid.

Example:
    >>> from drawing_intelligence.database import DatabaseManager
    >>> from drawing_intelligence.export import ExportManager, ExportConfig, JSONFormat
    >>>
    >>> db = DatabaseManager("drawings.db")
    >>> config = ExportConfig(json_format=JSONFormat.PRETTY)
    >>> exporter = ExportManager(db, config)
    >>>
    >>> # Export single drawing to JSON
    >>> exporter.export_drawing_json("DWG-001", "output/drawing.json")
    >>>
    >>> # Export batch to CSV
    >>> drawing_ids = ["DWG-001", "DWG-002", "DWG-003"]
    >>> exporter.export_batch_csv(drawing_ids, "output/batch/")
"""

import logging
import os
import re
import tempfile
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from dataclasses import dataclass

from ..database.database_manager import DatabaseManager, DrawingRecord
from .json_exporter import JSONExporter
from .csv_exporter import CSVExporter
from .report_generator import ReportGenerator


logger = logging.getLogger(__name__)


# ============================================================================
# Enums for Type Safety
# ============================================================================


class JSONFormat(Enum):
    """JSON formatting options."""

    PRETTY = "pretty"
    COMPACT = "compact"


class ReportFormat(Enum):
    """Report output format options."""

    HTML = "html"
    PDF = "pdf"


# ============================================================================
# Custom Exceptions
# ============================================================================


class ExportError(Exception):
    """Base exception for export operations."""

    pass


class DrawingNotFoundError(ExportError):
    """Raised when a drawing is not found in the database."""

    def __init__(self, drawing_id: str):
        self.drawing_id = drawing_id
        super().__init__(f"Drawing {drawing_id} not found in database")


class ExportFileError(ExportError):
    """Raised when file write operations fail."""

    def __init__(self, path: Path, message: str, original_error: Exception = None):
        self.path = path
        self.original_error = original_error
        super().__init__(f"Failed to write to {path}: {message}")


class InvalidConfigError(ExportError):
    """Raised when export configuration is invalid."""

    pass


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class ExportConfig:
    """Configuration for export operations.

    Attributes:
        json_format: JSON formatting style (PRETTY or COMPACT).
        csv_delimiter: Delimiter character for CSV files.
        include_intermediate_results: Whether to include OCR/detection intermediates.
        include_images: Whether to include image data in exports.

    Raises:
        InvalidConfigError: If configuration is invalid.
    """

    json_format: JSONFormat = JSONFormat.PRETTY
    csv_delimiter: str = ","
    include_intermediate_results: bool = False
    include_images: bool = False

    def __post_init__(self) -> None:
        """Validate configuration.

        Raises:
            InvalidConfigError: If configuration parameters are invalid.
        """
        # Validate csv_delimiter
        if not self.csv_delimiter:
            raise InvalidConfigError("csv_delimiter cannot be empty")

        if len(self.csv_delimiter) != 1:
            raise InvalidConfigError(
                f"csv_delimiter must be a single character, got: '{self.csv_delimiter}'"
            )

        # Check for reserved/problematic characters
        if self.csv_delimiter in {"\n", "\r", "\t", '"', "'"}:
            raise InvalidConfigError(
                f"csv_delimiter cannot be a reserved character: '{self.csv_delimiter}'"
            )


# ============================================================================
# Export Manager
# ============================================================================


class ExportManager:
    """Manages all export operations for drawing intelligence results.

    Coordinates JSON, CSV, and report generation for drawing results by
    delegating to specialized exporter classes. Handles both single drawing
    and batch export operations with robust error handling and atomic writes.

    Attributes:
        db: Database manager instance for retrieving drawing data.
        config: Export configuration settings.
        json_exporter: JSON format exporter instance.
        csv_exporter: CSV format exporter instance.
        report_generator: Report generation instance.

    Example:
        >>> db = DatabaseManager("drawings.db")
        >>> exporter = ExportManager(db)
        >>> exporter.export_drawing_json("DWG-001", "output.json")
    """

    def __init__(
        self,
        db: DatabaseManager,
        config: Optional[ExportConfig] = None,
        json_exporter: Optional[JSONExporter] = None,
        csv_exporter: Optional[CSVExporter] = None,
        report_generator: Optional[ReportGenerator] = None,
    ) -> None:
        """Initialize export manager.

        Args:
            db: Database manager instance for data retrieval.
            config: Export configuration. If None, uses default configuration.
            json_exporter: Custom JSON exporter. If None, creates default.
            csv_exporter: Custom CSV exporter. If None, creates default.
            report_generator: Custom report generator. If None, creates default.

        Note:
            Custom exporters enable easier unit testing and format extensions.
        """
        self.db = db
        self.config = config or ExportConfig()

        # Initialize exporters (dependency injection)
        self.json_exporter = json_exporter or JSONExporter(self.config)
        self.csv_exporter = csv_exporter or CSVExporter(self.config)
        self.report_generator = report_generator or ReportGenerator()

        logger.info(
            "ExportManager initialized",
            extra={
                "json_format": self.config.json_format.value,
                "csv_delimiter": repr(self.config.csv_delimiter),
            },
        )

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal and invalid characters.

        Args:
            filename: Raw filename or ID to sanitize.

        Returns:
            Sanitized filename safe for file system use.
        """
        # Remove path separators and other dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename)
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip(". ")
        # Ensure not empty
        return sanitized or "unnamed"

    def _prepare_output_path(
        self,
        output_path: Union[str, Path],
        expected_format: Optional[ReportFormat] = None,
    ) -> Path:
        """Prepare and validate output path.

        Args:
            output_path: Destination file path.
            expected_format: Expected file format for validation.

        Returns:
            Validated Path object.

        Raises:
            ExportFileError: If path validation fails.
        """
        output_file = Path(output_path)

        # Validate file extension matches format if specified
        if expected_format is not None:
            expected_ext = f".{expected_format.value}"
            if output_file.suffix.lower() != expected_ext:
                logger.warning(
                    "Output path extension mismatch",
                    extra={
                        "path": str(output_file),
                        "actual_ext": output_file.suffix,
                        "expected_ext": expected_ext,
                    },
                )

        # Ensure parent directory exists
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ExportFileError(
                output_file.parent, f"Cannot create parent directory: {e}", e
            )

        return output_file

    def _atomic_write(self, content_writer, output_path: Path) -> None:
        """Write content atomically using temporary file.

        Args:
            content_writer: Callable that writes content given a file path.
            output_path: Final destination path.

        Raises:
            ExportFileError: If write operation fails.
        """
        # Create temporary file in same directory for atomic move
        temp_fd, temp_path = tempfile.mkstemp(
            dir=output_path.parent, suffix=output_path.suffix
        )
        temp_file = Path(temp_path)

        try:
            # Close the file descriptor (writer will open by path)
            os.close(temp_fd)

            # Write content to temporary file
            content_writer(str(temp_file))

            # Atomic move to final destination
            temp_file.replace(output_path)

        except Exception as e:
            # Clean up temporary file on failure
            try:
                temp_file.unlink(missing_ok=True)
            except Exception:
                pass
            raise ExportFileError(output_path, str(e), e)

    def _get_drawing_or_raise(self, drawing_id: str) -> DrawingRecord:
        """Retrieve drawing from database or raise exception.

        Args:
            drawing_id: Drawing identifier.

        Returns:
            Drawing record from database.

        Raises:
            DrawingNotFoundError: If drawing not found.
        """
        drawing_record = self.db.get_drawing_by_id(drawing_id)
        if drawing_record is None:
            raise DrawingNotFoundError(drawing_id)
        return drawing_record

    def _get_batch_records(self, drawing_ids: List[str]) -> List[DrawingRecord]:
        """Retrieve multiple drawings from database with logging.

        Args:
            drawing_ids: List of drawing identifiers.

        Returns:
            List of found drawing records.

        Raises:
            ExportError: If no valid drawings found.
        """
        # Batch query for performance
        drawing_records: List[DrawingRecord] = []

        # Use database batch query if available
        if hasattr(self.db, "get_drawings_by_ids"):
            # Optimized batch query
            records_dict = self.db.get_drawings_by_ids(drawing_ids)

            for drawing_id in drawing_ids:
                if drawing_id in records_dict:
                    drawing_records.append(records_dict[drawing_id])
                else:
                    logger.info(
                        "Drawing not found, skipping", extra={"drawing_id": drawing_id}
                    )
        else:
            # Fallback to individual queries
            for drawing_id in drawing_ids:
                try:
                    record = self.db.get_drawing_by_id(drawing_id)
                    if record is None:
                        logger.info(
                            "Drawing not found, skipping",
                            extra={"drawing_id": drawing_id},
                        )
                    else:
                        drawing_records.append(record)
                except Exception as e:
                    logger.warning(
                        "Error retrieving drawing, skipping",
                        extra={"drawing_id": drawing_id, "error": str(e)},
                    )

        if not drawing_records:
            raise ExportError("No valid drawings found to export")

        return drawing_records

    # ========================================================================
    # Public Export Methods
    # ========================================================================

    def export_drawing_json(
        self,
        drawing_id: str,
        output_path: Union[str, Path],
        config_override: Optional[ExportConfig] = None,
    ) -> Path:
        """Export single drawing to JSON file.

        Retrieves the specified drawing from the database and exports it to
        a JSON file using atomic write operations.

        Args:
            drawing_id: Unique identifier of the drawing to export.
            output_path: Destination file path for the JSON output.
            config_override: Temporary config override for this operation.

        Returns:
            Path object of the output file.

        Raises:
            DrawingNotFoundError: If the drawing is not found in the database.
            ExportFileError: If the file write operation fails.

        Example:
            >>> path = exporter.export_drawing_json("DWG-001", "results/dwg001.json")
            >>> print(path)
            PosixPath('results/dwg001.json')
        """
        sanitized_id = self._sanitize_filename(drawing_id)
        logger.info(
            "Exporting drawing to JSON",
            extra={"drawing_id": drawing_id, "output_path": str(output_path)},
        )

        # Prepare output path
        output_file = self._prepare_output_path(output_path)

        # Get drawing from database
        drawing_record = self._get_drawing_or_raise(drawing_id)

        # Use override config if provided
        exporter = self.json_exporter
        if config_override is not None:
            exporter = JSONExporter(config_override)

        # Atomic write
        self._atomic_write(
            lambda path: exporter.export_single(drawing_record, path), output_file
        )

        logger.info(
            "Successfully exported drawing to JSON",
            extra={
                "drawing_id": drawing_id,
                "output_path": str(output_file),
                "file_size": output_file.stat().st_size,
            },
        )
        return output_file

    def export_drawing_csv(
        self,
        drawing_id: str,
        output_dir: Union[str, Path],
        config_override: Optional[ExportConfig] = None,
    ) -> List[Path]:
        """Export single drawing to multiple CSV files.

        Creates separate CSV files for different data types:
        - {drawing_id}_summary.csv: High-level drawing information
        - {drawing_id}_entities.csv: Extracted entities (if present)
        - {drawing_id}_detections.csv: Shape detections (if present)
        - {drawing_id}_associations.csv: Entity-shape associations (if present)

        Args:
            drawing_id: Unique identifier of the drawing to export.
            output_dir: Destination directory for CSV files.
            config_override: Temporary config override for this operation.

        Returns:
            List of Path objects for all generated CSV files.

        Raises:
            DrawingNotFoundError: If the drawing is not found in the database.
            ExportFileError: If file write operations fail.

        Example:
            >>> files = exporter.export_drawing_csv("DWG-001", "results/")
            >>> print([f.name for f in files])
            ['DWG-001_summary.csv', 'DWG-001_entities.csv']
        """
        sanitized_id = self._sanitize_filename(drawing_id)
        logger.info(
            "Exporting drawing to CSV",
            extra={"drawing_id": drawing_id, "output_dir": str(output_dir)},
        )

        # Get drawing from database
        drawing_record = self._get_drawing_or_raise(drawing_id)

        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Use override config if provided
        exporter = self.csv_exporter
        if config_override is not None:
            exporter = CSVExporter(config_override)

        # Export to CSV files with atomic writes
        output_files: List[Path] = []

        try:
            # Summary CSV (always generated)
            summary_path = output_path / f"{sanitized_id}_summary.csv"
            self._atomic_write(
                lambda path: exporter.export_summary([drawing_record], path),
                summary_path,
            )
            output_files.append(summary_path)

            # Entities CSV (if present)
            if drawing_record.entities:
                entities_path = output_path / f"{sanitized_id}_entities.csv"
                self._atomic_write(
                    lambda path: exporter.export_entities([drawing_record], path),
                    entities_path,
                )
                output_files.append(entities_path)

            # Detections CSV (if present)
            if drawing_record.detections:
                detections_path = output_path / f"{sanitized_id}_detections.csv"
                self._atomic_write(
                    lambda path: exporter.export_detections([drawing_record], path),
                    detections_path,
                )
                output_files.append(detections_path)

            # Associations CSV (if present)
            if drawing_record.associations:
                associations_path = output_path / f"{sanitized_id}_associations.csv"
                self._atomic_write(
                    lambda path: exporter.export_associations([drawing_record], path),
                    associations_path,
                )
                output_files.append(associations_path)

        except Exception as e:
            # Clean up any successfully written files on partial failure
            for file_path in output_files:
                try:
                    file_path.unlink(missing_ok=True)
                except Exception:
                    pass
            raise

        logger.info(
            "Successfully exported drawing to CSV",
            extra={
                "drawing_id": drawing_id,
                "file_count": len(output_files),
                "total_size": sum(f.stat().st_size for f in output_files),
            },
        )
        return output_files

    def export_batch_json(
        self,
        drawing_ids: List[str],
        output_path: Union[str, Path],
        config_override: Optional[ExportConfig] = None,
    ) -> Path:
        """Export multiple drawings to single JSON file.

        Consolidates all specified drawings into a single JSON file. Drawings
        not found in the database are logged and skipped.

        Args:
            drawing_ids: List of drawing identifiers to export.
            output_path: Destination file path for the consolidated JSON.
            config_override: Temporary config override for this operation.

        Returns:
            Path object of the output file.

        Raises:
            ExportError: If no valid drawings are found to export.
            ExportFileError: If the file write operation fails.

        Example:
            >>> ids = ["DWG-001", "DWG-002", "DWG-003"]
            >>> path = exporter.export_batch_json(ids, "results/batch.json")
        """
        logger.info(
            "Exporting batch to JSON",
            extra={"drawing_count": len(drawing_ids), "output_path": str(output_path)},
        )

        # Prepare output path
        output_file = self._prepare_output_path(output_path)

        # Get all drawings (batch query for performance)
        drawing_records = self._get_batch_records(drawing_ids)

        # Use override config if provided
        exporter = self.json_exporter
        if config_override is not None:
            exporter = JSONExporter(config_override)

        # Atomic write
        self._atomic_write(
            lambda path: exporter.export_batch(drawing_records, path), output_file
        )

        logger.info(
            "Successfully exported batch to JSON",
            extra={
                "drawing_count": len(drawing_records),
                "output_path": str(output_file),
                "file_size": output_file.stat().st_size,
            },
        )
        return output_file

    def export_batch_csv(
        self,
        drawing_ids: List[str],
        output_dir: Union[str, Path],
        config_override: Optional[ExportConfig] = None,
    ) -> List[Path]:
        """Export batch of drawings to consolidated CSV files.

        Creates consolidated CSV files combining data from all drawings:
        - batch_summary.csv: Summary information for all drawings
        - batch_entities.csv: All entities from all drawings
        - batch_detections.csv: All detections from all drawings
        - batch_associations.csv: All associations from all drawings

        Args:
            drawing_ids: List of drawing identifiers to export.
            output_dir: Destination directory for CSV files.
            config_override: Temporary config override for this operation.

        Returns:
            List of Path objects for all generated CSV files.

        Raises:
            ExportError: If no valid drawings are found to export.
            ExportFileError: If file write operations fail.

        Note:
            Drawings not found in the database are logged and skipped.

        Example:
            >>> ids = ["DWG-001", "DWG-002", "DWG-003"]
            >>> files = exporter.export_batch_csv(ids, "results/batch/")
            >>> print(len(files))
            4
        """
        logger.info(
            "Exporting batch to CSV",
            extra={"drawing_count": len(drawing_ids), "output_dir": str(output_dir)},
        )

        # Get all drawings (batch query for performance)
        drawing_records = self._get_batch_records(drawing_ids)

        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Use override config if provided
        exporter = self.csv_exporter
        if config_override is not None:
            exporter = CSVExporter(config_override)

        # Export to CSV files with atomic writes
        output_files: List[Path] = []

        try:
            # Summary CSV (always generated)
            summary_path = output_path / "batch_summary.csv"
            self._atomic_write(
                lambda path: exporter.export_summary(drawing_records, path),
                summary_path,
            )
            output_files.append(summary_path)

            # Entities CSV (always generated, may be empty)
            entities_path = output_path / "batch_entities.csv"
            self._atomic_write(
                lambda path: exporter.export_entities(drawing_records, path),
                entities_path,
            )
            output_files.append(entities_path)

            # Detections CSV (always generated, may be empty)
            detections_path = output_path / "batch_detections.csv"
            self._atomic_write(
                lambda path: exporter.export_detections(drawing_records, path),
                detections_path,
            )
            output_files.append(detections_path)

            # Associations CSV (always generated, may be empty)
            associations_path = output_path / "batch_associations.csv"
            self._atomic_write(
                lambda path: exporter.export_associations(drawing_records, path),
                associations_path,
            )
            output_files.append(associations_path)

        except Exception as e:
            # Clean up any successfully written files on partial failure
            for file_path in output_files:
                try:
                    file_path.unlink(missing_ok=True)
                except Exception:
                    pass
            raise

        logger.info(
            "Successfully exported batch to CSV",
            extra={
                "file_count": len(output_files),
                "total_size": sum(f.stat().st_size for f in output_files),
            },
        )
        return output_files

    def generate_batch_report(
        self,
        batch_id: str,
        output_path: Union[str, Path],
        format: ReportFormat = ReportFormat.HTML,
    ) -> Path:
        """Generate comprehensive batch processing report.

        Creates a detailed report summarizing batch processing results,
        including success rates, review requirements, costs, and timing.

        Args:
            batch_id: Unique identifier of the batch to report on.
            output_path: Destination file path for the report.
            format: Report output format (HTML or PDF).

        Returns:
            Path object of the report file.

        Raises:
            ExportError: If batch is not found.
            ExportFileError: If file write operation fails.

        Warning:
            Batch tracking must be implemented in the database layer.

        Example:
            >>> path = exporter.generate_batch_report(
            ...     "BATCH-2025-01",
            ...     "reports/batch_report.html",
            ...     format=ReportFormat.HTML
            ... )
        """
        sanitized_id = self._sanitize_filename(batch_id)
        logger.info(
            "Generating batch report",
            extra={"batch_id": batch_id, "format": format.value},
        )

        # Prepare output path with format validation
        output_file = self._prepare_output_path(output_path, format)

        # TODO: Implement batch tracking in database
        logger.warning("Batch tracking not yet implemented - using placeholder data")

        from ..models.data_structures import BatchResult

        batch_result = BatchResult(
            batch_id=batch_id,
            total_drawings=0,
            successful=0,
            failed=0,
            needs_review=0,
            success_rate=0.0,
            review_rate=0.0,
            total_llm_cost=0.0,
            average_processing_time=0.0,
            drawing_results=[],
        )

        # Get drawing records (placeholder - should query by batch_id)
        drawing_records: List[DrawingRecord] = []

        # Atomic write
        self._atomic_write(
            lambda path: self.report_generator.generate_batch_report(
                batch_result, drawing_records, path, format.value
            ),
            output_file,
        )

        logger.info(
            "Successfully generated batch report",
            extra={
                "batch_id": batch_id,
                "output_path": str(output_file),
                "file_size": output_file.stat().st_size,
            },
        )
        return output_file

    def export_cost_report(
        self,
        start_date: datetime,
        end_date: datetime,
        output_path: Union[str, Path],
        format: ReportFormat = ReportFormat.HTML,
    ) -> Path:
        """Generate LLM usage and cost report for a date range.

        Creates a report detailing LLM API usage, costs by provider/model/use-case,
        and token consumption statistics for the specified period.

        Args:
            start_date: Beginning of the reporting period (inclusive).
            end_date: End of the reporting period (inclusive).
            output_path: Destination file path for the report.
            format: Report output format (HTML or PDF).

        Returns:
            Path object of the report file.

        Raises:
            ExportError: If start_date >= end_date.
            ExportFileError: If file write operation fails.

        Example:
            >>> from datetime import datetime, timedelta
            >>> end = datetime.now()
            >>> start = end - timedelta(days=30)
            >>> path = exporter.export_cost_report(
            ...     start, end,
            ...     "reports/monthly_costs.html"
            ... )
        """
        logger.info(
            "Generating cost report",
            extra={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "format": format.value,
            },
        )

        # Validate date range
        if start_date >= end_date:
            raise ExportError(
                f"start_date ({start_date}) must be before end_date ({end_date})"
            )

        # Prepare output path with format validation
        output_file = self._prepare_output_path(output_path, format)

        # Get cost data from database
        from ..llm.cost_tracker import CostTracker

        cost_tracker = CostTracker(self.db)
        cost_report = cost_tracker.generate_cost_report(
            period="custom", start_date=start_date, end_date=end_date
        )

        # Atomic write
        self._atomic_write(
            lambda path: self.report_generator.generate_cost_report(
                cost_report, path, format.value
            ),
            output_file,
        )

        logger.info(
            "Successfully generated cost report",
            extra={
                "output_path": str(output_file),
                "file_size": output_file.stat().st_size,
                "total_cost": cost_report.total_cost,
            },
        )
        return output_file

    def export_drawing_report(
        self,
        drawing_id: str,
        output_path: Union[str, Path],
        format: ReportFormat = ReportFormat.HTML,
    ) -> Path:
        """Generate detailed report for a single drawing.

        Creates a comprehensive report including processing results, extracted
        entities, detected shapes, quality metrics, and any validation issues.

        Args:
            drawing_id: Unique identifier of the drawing to report on.
            output_path: Destination file path for the report.
            format: Report output format (HTML or PDF).

        Returns:
            Path object of the report file.

        Raises:
            DrawingNotFoundError: If drawing is not found.
            ExportFileError: If file write operation fails.

        Example:
            >>> path = exporter.export_drawing_report(
            ...     "DWG-001",
            ...     "reports/dwg001_report.pdf",
            ...     format=ReportFormat.PDF
            ... )
        """
        sanitized_id = self._sanitize_filename(drawing_id)
        logger.info(
            "Generating drawing report",
            extra={"drawing_id": drawing_id, "format": format.value},
        )

        # Prepare output path with format validation
        output_file = self._prepare_output_path(output_path, format)

        # Get drawing from database
        drawing_record = self._get_drawing_or_raise(drawing_id)

        # Atomic write
        self._atomic_write(
            lambda path: self.report_generator.generate_drawing_report(
                drawing_record, path, format.value
            ),
            output_file,
        )

        logger.info(
            "Successfully generated drawing report",
            extra={
                "drawing_id": drawing_id,
                "output_path": str(output_file),
                "file_size": output_file.stat().st_size,
            },
        )
        return output_file
