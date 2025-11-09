"""CSV Exporter Module.

This module provides functionality to export drawing processing results
to CSV format with multiple export modes (summary, entities, detections,
associations).

Classes:
    CSVExporter: Handles CSV export operations for drawing records.

Exceptions:
    CSVExportError: Base exception for CSV export operations.
    InvalidDelimiterError: Raised when delimiter is invalid.
    EmptyDataError: Raised when no data is available for export.
    FileWriteError: Raised when file write operation fails.
"""

import csv
import io
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Protocol, Union

from ..database.database_manager import DrawingRecord
from ..models.data_structures import EntityType


logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS - Centralized Header Definitions
# ============================================================================

SUMMARY_HEADERS = [
    "drawing_id",
    "source_file",
    "processing_timestamp",
    "overall_confidence",
    "needs_review",
    "status",
    "entity_count",
    "detection_count",
    "association_count",
    "has_part_number",
    "has_title_block",
    "review_flag_count",
]

ENTITY_HEADERS = [
    "drawing_id",
    "source_file",
    "entity_id",
    "entity_type",
    "value",
    "original_text",
    "confidence",
    "extraction_method",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
]

DETECTION_HEADERS = [
    "drawing_id",
    "source_file",
    "detection_id",
    "class_name",
    "confidence",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "bbox_norm_x_center",
    "bbox_norm_y_center",
    "bbox_norm_width",
    "bbox_norm_height",
]

ASSOCIATION_HEADERS = [
    "drawing_id",
    "source_file",
    "association_id",
    "text_id",
    "shape_id",
    "relationship_type",
    "confidence",
    "distance_pixels",
]

# Reserved characters that could trigger CSV injection
CSV_INJECTION_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


# ============================================================================
# EXCEPTIONS
# ============================================================================


class CSVExportError(Exception):
    """Base exception for CSV export operations."""

    pass


class InvalidDelimiterError(CSVExportError):
    """Raised when CSV delimiter is invalid."""

    pass


class EmptyDataError(CSVExportError):
    """Raised when no data is available for export."""

    pass


class FileWriteError(CSVExportError):
    """Raised when file write operation fails."""

    pass


# ============================================================================
# CONFIGURATION PROTOCOL
# ============================================================================


class CSVConfigProtocol(Protocol):
    """Protocol defining required CSV configuration attributes."""

    csv_delimiter: str
    float_precision: int
    datetime_format: str
    encoding: str


# ============================================================================
# CSV EXPORTER
# ============================================================================


class CSVExporter:
    """Export drawing results to CSV format.

    This class provides multiple export methods to generate CSV files
    from drawing processing results, including summary reports and
    detailed entity/detection/association exports. All exports use
    streaming generators to minimize memory usage.

    Attributes:
        config: Export configuration object with CSV settings.
        delimiter: CSV field delimiter character (validated).
        float_precision: Number of decimal places for float formatting.
        datetime_format: strftime format string for datetime fields.
        encoding: Output file encoding.

    Example:
        >>> from drawing_intelligence.export.csv_exporter import CSVExporter
        >>> exporter = CSVExporter(config)
        >>> output_path = exporter.export_summary(records, "output/summary.csv")
    """

    def __init__(
        self,
        config: CSVConfigProtocol,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Initialize CSV exporter with configuration.

        Args:
            config: Export configuration object implementing CSVConfigProtocol.
                Must have: csv_delimiter, float_precision, datetime_format,
                encoding attributes.
            progress_callback: Optional callback function(current, total) for
                progress tracking during large exports.

        Raises:
            InvalidDelimiterError: If delimiter is invalid (length != 1,
                newline, or other reserved character).
        """
        self.config = config
        self.progress_callback = progress_callback

        # Validate and set delimiter
        delimiter = config.csv_delimiter
        if not delimiter or len(delimiter) != 1:
            raise InvalidDelimiterError(
                f"Delimiter must be exactly one character, got: '{delimiter}'"
            )
        if delimiter in ("\n", "\r", "\r\n"):
            raise InvalidDelimiterError("Delimiter cannot be a newline character")
        self.delimiter = delimiter

        # Set formatting options with defaults
        self.float_precision = getattr(config, "float_precision", 4)
        self.datetime_format = getattr(config, "datetime_format", "%Y-%m-%d %H:%M:%S")
        self.encoding = getattr(config, "encoding", "utf-8")

        logger.info(
            "CSVExporter initialized",
            extra={
                "delimiter": self.delimiter,
                "float_precision": self.float_precision,
                "encoding": self.encoding,
            },
        )

    def export_summary(
        self, drawing_records: List[DrawingRecord], output_path: Union[str, Path]
    ) -> Path:
        """Export summary CSV with one row per drawing.

        Creates a CSV file containing high-level statistics for each
        drawing including confidence scores, entity counts, and review flags.

        Args:
            drawing_records: List of DrawingRecord objects to export.
            output_path: Destination file path for the CSV output.

        Returns:
            Path object of the created CSV file.

        Raises:
            EmptyDataError: If drawing_records is empty.
            FileWriteError: If file write operation fails.

        Example:
            >>> path = exporter.export_summary(records, "summary.csv")
        """
        if not drawing_records:
            raise EmptyDataError("No drawing records provided for export")

        logger.info(
            "Starting summary CSV export",
            extra={
                "record_count": len(drawing_records),
                "output_path": str(output_path),
            },
        )

        # Generate rows
        def row_generator() -> Generator[Dict[str, Any], None, None]:
            for idx, record in enumerate(drawing_records):
                if self.progress_callback:
                    self.progress_callback(idx + 1, len(drawing_records))
                yield self.flatten_drawing_data(record)

        # Write CSV
        return self._write_csv_atomic(
            row_generator(), output_path, SUMMARY_HEADERS, len(drawing_records)
        )

    def export_entities(
        self, drawing_records: List[DrawingRecord], output_path: Union[str, Path]
    ) -> Path:
        """Export entities CSV with one row per entity.

        Flattens all entities across drawings into a detailed CSV format,
        including bounding box coordinates and extraction metadata.

        Args:
            drawing_records: List of DrawingRecord objects containing
                entities.
            output_path: Destination file path for the CSV output.

        Returns:
            Path object of the created CSV file.

        Raises:
            EmptyDataError: If no entities found in any drawing.
            FileWriteError: If file write operation fails.

        Note:
            Drawings without entities will be skipped automatically.
            Entities without bounding boxes will have empty bbox fields.
        """

        def entity_row_extractor(
            record: DrawingRecord,
        ) -> Generator[Dict[str, Any], None, None]:
            """Extract entity rows from a drawing record."""
            if not record.entities:
                return

            for entity in record.entities:
                # Handle optional bbox safely
                bbox_x = entity.bbox.x if entity.bbox else ""
                bbox_y = entity.bbox.y if entity.bbox else ""
                bbox_width = entity.bbox.width if entity.bbox else ""
                bbox_height = entity.bbox.height if entity.bbox else ""

                yield {
                    "drawing_id": record.drawing_id,
                    "source_file": record.source_file,
                    "entity_id": entity.entity_id,
                    "entity_type": entity.entity_type.value,
                    "value": self._sanitize_value(entity.value),
                    "original_text": self._sanitize_value(entity.original_text),
                    "confidence": self._format_float(entity.confidence),
                    "extraction_method": entity.extraction_method,
                    "bbox_x": bbox_x,
                    "bbox_y": bbox_y,
                    "bbox_width": bbox_width,
                    "bbox_height": bbox_height,
                }

        return self._export_detail_rows(
            drawing_records,
            output_path,
            ENTITY_HEADERS,
            entity_row_extractor,
            "entities",
        )

    def export_detections(
        self, drawing_records: List[DrawingRecord], output_path: Union[str, Path]
    ) -> Path:
        """Export detections CSV with one row per shape detection.

        Exports all shape detections with both pixel-based and normalized
        bounding box coordinates.

        Args:
            drawing_records: List of DrawingRecord objects with detections.
            output_path: Destination file path for the CSV output.

        Returns:
            Path object of the created CSV file.

        Raises:
            EmptyDataError: If no detections found in any drawing.
            FileWriteError: If file write operation fails.

        Note:
            Includes both absolute and normalized bounding box coordinates.
        """

        def detection_row_extractor(
            record: DrawingRecord,
        ) -> Generator[Dict[str, Any], None, None]:
            """Extract detection rows from a drawing record."""
            if not record.detections:
                return

            for detection in record.detections:
                yield {
                    "drawing_id": record.drawing_id,
                    "source_file": record.source_file,
                    "detection_id": detection.detection_id,
                    "class_name": detection.class_name,
                    "confidence": self._format_float(detection.confidence),
                    "bbox_x": detection.bbox.x,
                    "bbox_y": detection.bbox.y,
                    "bbox_width": detection.bbox.width,
                    "bbox_height": detection.bbox.height,
                    "bbox_norm_x_center": self._format_float(
                        detection.bbox_normalized.x_center
                    ),
                    "bbox_norm_y_center": self._format_float(
                        detection.bbox_normalized.y_center
                    ),
                    "bbox_norm_width": self._format_float(
                        detection.bbox_normalized.width
                    ),
                    "bbox_norm_height": self._format_float(
                        detection.bbox_normalized.height
                    ),
                }

        return self._export_detail_rows(
            drawing_records,
            output_path,
            DETECTION_HEADERS,
            detection_row_extractor,
            "detections",
        )

    def export_associations(
        self, drawing_records: List[DrawingRecord], output_path: Union[str, Path]
    ) -> Path:
        """Export associations CSV linking text and shapes.

        Creates a CSV file mapping relationships between text blocks
        and detected shapes, including relationship types and spatial
        distances.

        Args:
            drawing_records: List of DrawingRecord objects with
                associations.
            output_path: Destination file path for the CSV output.

        Returns:
            Path object of the created CSV file.

        Raises:
            EmptyDataError: If no associations found in any drawing.
            FileWriteError: If file write operation fails.
        """

        def association_row_extractor(
            record: DrawingRecord,
        ) -> Generator[Dict[str, Any], None, None]:
            """Extract association rows from a drawing record."""
            if not record.associations:
                return

            for association in record.associations:
                yield {
                    "drawing_id": record.drawing_id,
                    "source_file": record.source_file,
                    "association_id": association.association_id,
                    "text_id": association.text_id,
                    "shape_id": association.shape_id,
                    "relationship_type": association.relationship_type,
                    "confidence": self._format_float(association.confidence),
                    "distance_pixels": self._format_float(association.distance_pixels),
                }

        return self._export_detail_rows(
            drawing_records,
            output_path,
            ASSOCIATION_HEADERS,
            association_row_extractor,
            "associations",
        )

    def export_to_string(
        self, drawing_records: List[DrawingRecord], export_type: str = "summary"
    ) -> str:
        """Export drawing records to CSV string (for APIs/testing).

        Args:
            drawing_records: List of DrawingRecord objects to export.
            export_type: Type of export ('summary', 'entities', 'detections',
                'associations').

        Returns:
            CSV content as string.

        Raises:
            ValueError: If export_type is invalid.
            EmptyDataError: If no data available for export.
        """
        output = io.StringIO()

        if export_type == "summary":
            if not drawing_records:
                raise EmptyDataError("No drawing records provided")

            writer = csv.DictWriter(
                output,
                fieldnames=SUMMARY_HEADERS,
                delimiter=self.delimiter,
                extrasaction="ignore",
            )
            writer.writeheader()
            for record in drawing_records:
                writer.writerow(self.flatten_drawing_data(record))

        elif export_type in ("entities", "detections", "associations"):
            # Use appropriate headers and extractor
            headers_map = {
                "entities": ENTITY_HEADERS,
                "detections": DETECTION_HEADERS,
                "associations": ASSOCIATION_HEADERS,
            }
            headers = headers_map[export_type]

            writer = csv.DictWriter(
                output,
                fieldnames=headers,
                delimiter=self.delimiter,
                extrasaction="ignore",
            )
            writer.writeheader()

            # This is intentionally simple for string export
            # (no streaming needed for in-memory operation)
            row_count = 0
            for record in drawing_records:
                if export_type == "entities" and record.entities:
                    for row in self._get_entity_rows(record):
                        writer.writerow(row)
                        row_count += 1
                elif export_type == "detections" and record.detections:
                    for row in self._get_detection_rows(record):
                        writer.writerow(row)
                        row_count += 1
                elif export_type == "associations" and record.associations:
                    for row in self._get_association_rows(record):
                        writer.writerow(row)
                        row_count += 1

            if row_count == 0:
                raise EmptyDataError(f"No {export_type} found in records")
        else:
            raise ValueError(
                f"Invalid export_type: {export_type}. Must be 'summary', "
                "'entities', 'detections', or 'associations'."
            )

        return output.getvalue()

    def flatten_drawing_data(self, drawing_record: DrawingRecord) -> Dict[str, Any]:
        """Flatten drawing record to single-level dictionary.

        Converts a DrawingRecord object into a flat dictionary suitable
        for CSV export, computing derived fields like entity counts and
        flags.

        Args:
            drawing_record: DrawingRecord object to flatten.

        Returns:
            Dictionary with string keys and primitive values suitable for
            CSV. Includes fields defined in SUMMARY_HEADERS.

        Note:
            Converts datetime objects using configured format.
            None values are converted to empty strings.
        """
        # Check for part number
        has_part_number = False
        if drawing_record.entities:
            has_part_number = any(
                e.entity_type == EntityType.PART_NUMBER for e in drawing_record.entities
            )

        # Check for title block
        has_title_block = False
        if hasattr(drawing_record, "title_block"):
            has_title_block = drawing_record.title_block is not None

        return {
            "drawing_id": drawing_record.drawing_id,
            "source_file": drawing_record.source_file,
            "processing_timestamp": drawing_record.processing_timestamp.strftime(
                self.datetime_format
            ),
            "overall_confidence": self._format_float(drawing_record.overall_confidence),
            "needs_review": drawing_record.needs_review,
            "status": drawing_record.status,
            "entity_count": (
                len(drawing_record.entities) if drawing_record.entities else 0
            ),
            "detection_count": (
                len(drawing_record.detections) if drawing_record.detections else 0
            ),
            "association_count": (
                len(drawing_record.associations) if drawing_record.associations else 0
            ),
            "has_part_number": has_part_number,
            "has_title_block": has_title_block,
            "review_flag_count": (
                len(drawing_record.review_flags) if drawing_record.review_flags else 0
            ),
        }

    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================

    def _export_detail_rows(
        self,
        drawing_records: List[DrawingRecord],
        output_path: Union[str, Path],
        headers: List[str],
        row_extractor: Callable[[DrawingRecord], Generator[Dict[str, Any], None, None]],
        export_type: str,
    ) -> Path:
        """Generic method to export detail rows using streaming.

        Args:
            drawing_records: List of DrawingRecord objects.
            output_path: Destination file path.
            headers: CSV column headers.
            row_extractor: Function that yields rows from a record.
            export_type: Name of export type for logging.

        Returns:
            Path object of the created CSV file.

        Raises:
            EmptyDataError: If no rows generated.
            FileWriteError: If file write operation fails.
        """
        if not drawing_records:
            raise EmptyDataError(
                f"No drawing records provided for {export_type} export"
            )

        logger.info(
            f"Starting {export_type} CSV export",
            extra={
                "record_count": len(drawing_records),
                "output_path": str(output_path),
                "export_type": export_type,
            },
        )

        # Generate rows from all records
        def row_generator() -> Generator[Dict[str, Any], None, None]:
            total_rows = 0
            for idx, record in enumerate(drawing_records):
                if self.progress_callback:
                    self.progress_callback(idx + 1, len(drawing_records))

                for row in row_extractor(record):
                    total_rows += 1
                    yield row

            # Check if we generated any rows
            if total_rows == 0:
                raise EmptyDataError(f"No {export_type} found in any drawing record")

        # Write CSV
        return self._write_csv_atomic(
            row_generator(),
            output_path,
            headers,
            None,  # Unknown row count for detail exports
        )

    def _write_csv_atomic(
        self,
        data_generator: Generator[Dict[str, Any], None, None],
        output_path: Union[str, Path],
        headers: List[str],
        expected_rows: Optional[int],
    ) -> Path:
        """Write CSV file atomically using temp file + rename.

        Internal method that handles the actual CSV file writing operation
        with atomic write guarantee to prevent corruption.

        Args:
            data_generator: Generator yielding dictionaries to write as rows.
            output_path: Destination file path for the CSV output.
            headers: Ordered list of column headers.
            expected_rows: Expected number of rows (for logging), or None.

        Returns:
            Path object of the created CSV file.

        Raises:
            FileWriteError: If file write operation fails.

        Note:
            Uses atomic write pattern (temp file + rename) to prevent
            partial file corruption on failure.
        """
        output_path_obj = Path(output_path)
        start_time = time.time()
        row_count = 0

        # Create temp file in same directory as target for atomic rename
        try:
            # Ensure output directory exists (caller responsibility ideally,
            # but safe to do here)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Create temp file in same directory
            fd, temp_path = tempfile.mkstemp(
                suffix=".csv.tmp", dir=output_path_obj.parent, text=True
            )

            try:
                with os.fdopen(fd, "w", encoding=self.encoding, newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=headers,
                        delimiter=self.delimiter,
                        extrasaction="ignore",
                    )
                    writer.writeheader()

                    # Stream rows
                    for row in data_generator:
                        writer.writerow(row)
                        row_count += 1

                # Atomic rename (replace existing file if present)
                os.replace(temp_path, output_path_obj)

            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise

        except (IOError, OSError) as e:
            error_msg = f"Failed to write CSV to {output_path_obj}: {e}"
            logger.error(
                error_msg, extra={"output_path": str(output_path_obj), "error": str(e)}
            )
            raise FileWriteError(error_msg) from e

        # Log success with metrics
        duration = time.time() - start_time
        file_size = output_path_obj.stat().st_size

        logger.info(
            "CSV export completed successfully",
            extra={
                "output_path": str(output_path_obj),
                "row_count": row_count,
                "expected_rows": expected_rows,
                "file_size_bytes": file_size,
                "duration_seconds": f"{duration:.2f}",
            },
        )

        return output_path_obj

    def _format_float(self, value: float) -> str:
        """Format float value with configured precision.

        Args:
            value: Float value to format.

        Returns:
            Formatted string representation.
        """
        if value is None:
            return ""
        return f"{value:.{self.float_precision}f}"

    def _sanitize_value(self, value: Any) -> str:
        """Sanitize value for CSV output to prevent injection attacks.

        Args:
            value: Value to sanitize.

        Returns:
            Sanitized string value safe for CSV.

        Note:
            Prepends single quote to values starting with formula characters
            (=, +, -, @, tab, carriage return) to prevent CSV injection.
        """
        if value is None:
            return ""

        str_value = str(value)

        # Prevent CSV injection by prepending quote to formula characters
        if str_value and str_value[0] in CSV_INJECTION_PREFIXES:
            return "'" + str_value

        return str_value

    # Helper methods for export_to_string (non-streaming variants)
    def _get_entity_rows(
        self, record: DrawingRecord
    ) -> Generator[Dict[str, Any], None, None]:
        """Get entity rows for a record (helper for export_to_string)."""
        if not record.entities:
            return

        for entity in record.entities:
            bbox_x = entity.bbox.x if entity.bbox else ""
            bbox_y = entity.bbox.y if entity.bbox else ""
            bbox_width = entity.bbox.width if entity.bbox else ""
            bbox_height = entity.bbox.height if entity.bbox else ""

            yield {
                "drawing_id": record.drawing_id,
                "source_file": record.source_file,
                "entity_id": entity.entity_id,
                "entity_type": entity.entity_type.value,
                "value": self._sanitize_value(entity.value),
                "original_text": self._sanitize_value(entity.original_text),
                "confidence": self._format_float(entity.confidence),
                "extraction_method": entity.extraction_method,
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
            }

    def _get_detection_rows(
        self, record: DrawingRecord
    ) -> Generator[Dict[str, Any], None, None]:
        """Get detection rows for a record (helper for export_to_string)."""
        if not record.detections:
            return

        for detection in record.detections:
            yield {
                "drawing_id": record.drawing_id,
                "source_file": record.source_file,
                "detection_id": detection.detection_id,
                "class_name": detection.class_name,
                "confidence": self._format_float(detection.confidence),
                "bbox_x": detection.bbox.x,
                "bbox_y": detection.bbox.y,
                "bbox_width": detection.bbox.width,
                "bbox_height": detection.bbox.height,
                "bbox_norm_x_center": self._format_float(
                    detection.bbox_normalized.x_center
                ),
                "bbox_norm_y_center": self._format_float(
                    detection.bbox_normalized.y_center
                ),
                "bbox_norm_width": self._format_float(detection.bbox_normalized.width),
                "bbox_norm_height": self._format_float(
                    detection.bbox_normalized.height
                ),
            }

    def _get_association_rows(
        self, record: DrawingRecord
    ) -> Generator[Dict[str, Any], None, None]:
        """Get association rows for a record (helper for export_to_string)."""
        if not record.associations:
            return

        for association in record.associations:
            yield {
                "drawing_id": record.drawing_id,
                "source_file": record.source_file,
                "association_id": association.association_id,
                "text_id": association.text_id,
                "shape_id": association.shape_id,
                "relationship_type": association.relationship_type,
                "confidence": self._format_float(association.confidence),
                "distance_pixels": self._format_float(association.distance_pixels),
            }
