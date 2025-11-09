"""JSON Exporter Module.

This module provides functionality to export drawing processing results
to JSON format with schema validation and support for both single and
batch exports.

Example:
    >>> from export_manager import ExportConfig
    >>> config = ExportConfig(format="json", output_path="output.json")
    >>> exporter = JSONExporter(config)
    >>> exporter.export_single(drawing_record, Path("output.json"))
"""

import json
import logging
import tempfile
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Tuple, Union

from ..database.database_manager import DrawingRecord
from ..models.data_structures import (
    Association,
    BoundingBox,
    ComponentHierarchy,
    Detection,
    Entity,
    ReviewFlag,
    TextBlock,
)


logger = logging.getLogger(__name__)

# Version for schema tracking
EXPORTER_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"


class ExportConfigProtocol(Protocol):
    """Protocol defining the expected configuration interface."""

    json_format: Literal["pretty", "compact"]
    validate_schema: bool


class JSONExportError(Exception):
    """Base exception for JSON export operations."""

    def __init__(self, message: str, drawing_id: Optional[str] = None):
        self.drawing_id = drawing_id
        super().__init__(message)


class SchemaValidationError(JSONExportError):
    """Exception raised when schema validation fails."""

    def __init__(
        self, message: str, errors: List[str], drawing_id: Optional[str] = None
    ):
        self.errors = errors
        super().__init__(message, drawing_id)


class ExportFileError(JSONExportError):
    """Exception raised when file operations fail."""

    pass


class DrawingJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for drawing data structures.

    Handles serialization of:
    - datetime objects -> ISO 8601 strings
    - Enum objects -> their values
    - BoundingBox objects -> dictionaries
    - Path objects -> strings

    Example:
        >>> encoder = DrawingJSONEncoder()
        >>> json.dumps(data, cls=DrawingJSONEncoder)
    """

    def default(self, obj: Any) -> Any:
        """Override default serialization behavior.

        Args:
            obj: Object to serialize.

        Returns:
            JSON-serializable representation of the object.

        Raises:
            TypeError: If object type is not serializable.
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, BoundingBox):
            return {
                "x": obj.x,
                "y": obj.y,
                "width": obj.width,
                "height": obj.height,
            }
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


class JSONExporter:
    """Exports drawing results to JSON format with schema validation.

    This class handles serialization of complex drawing data structures
    including entities, detections, associations, and hierarchies into
    JSON format suitable for external consumption or archival.

    Attributes:
        config: Export configuration containing format preferences and options.

    Example:
        >>> config = ExportConfig(format="json", output_path="output.json")
        >>> exporter = JSONExporter(config)
        >>> exporter.export_single(record, Path("output.json"))
    """

    def __init__(self, config: ExportConfigProtocol) -> None:
        """Initialize JSON exporter with configuration.

        Args:
            config: Export configuration object containing format preferences
                and export options such as json_format ("pretty" or "compact").

        Example:
            >>> config = ExportConfig(format="json", output_path="out.json")
            >>> exporter = JSONExporter(config)
        """
        self.config = config
        logger.info("JSONExporter initialized")

    def export_single(
        self, drawing_record: DrawingRecord, output_path: Union[str, Path]
    ) -> None:
        """Export a single drawing record to a JSON file.

        Formats the drawing record, optionally validates the schema,
        and writes the result to the specified file path using atomic
        write operations. Creates parent directories if they don't exist.

        Args:
            drawing_record: Drawing record containing all processing results
                to be exported.
            output_path: File system path where the JSON file will be written.

        Raises:
            JSONExportError: If export operation fails.
            SchemaValidationError: If validation is enabled and fails.
            ExportFileError: If file write operation fails.

        Example:
            >>> exporter.export_single(record, Path("results/drawing_001.json"))
        """
        start_time = time.time()
        output_path = Path(output_path)

        logger.info(
            f"Exporting drawing {drawing_record.drawing_id} to JSON: " f"{output_path}"
        )

        try:
            # Format drawing data
            json_data = self.format_drawing_result(drawing_record)

            # Validate schema if configured
            if getattr(self.config, "validate_schema", False):
                self._validate_or_raise(json_data, drawing_record.drawing_id)

            # Write to file atomically
            self._write_json_file_atomic(json_data, output_path)

            duration = time.time() - start_time
            file_size = output_path.stat().st_size / 1024  # KB

            logger.info(
                f"Successfully exported {drawing_record.drawing_id} to "
                f"{output_path} ({file_size:.2f} KB in {duration:.2f}s)"
            )

        except SchemaValidationError:
            raise
        except (IOError, OSError) as e:
            error_msg = f"Failed to export drawing {drawing_record.drawing_id}: {e}"
            logger.error(error_msg)
            raise ExportFileError(error_msg, drawing_record.drawing_id) from e
        except Exception as e:
            error_msg = f"Unexpected error exporting {drawing_record.drawing_id}: {e}"
            logger.error(error_msg)
            raise JSONExportError(error_msg, drawing_record.drawing_id) from e

    def export_batch(
        self, drawing_records: List[DrawingRecord], output_path: Union[str, Path]
    ) -> None:
        """Export multiple drawing records to a single JSON file.

        Creates a batch export containing all provided drawings with
        metadata including export timestamp and total count. All drawings
        are wrapped in a top-level "drawings" array.

        Args:
            drawing_records: List of drawing records to export in batch.
            output_path: File system path for the output JSON file.

        Raises:
            JSONExportError: If export operation fails.
            ExportFileError: If file write operation fails.

        Example:
            >>> exporter.export_batch(records, Path("results/batch.json"))
        """
        start_time = time.time()
        output_path = Path(output_path)

        logger.info(
            f"Exporting batch of {len(drawing_records)} drawings to JSON: "
            f"{output_path}"
        )

        try:
            # Format all drawings
            batch_data = {
                "_export_metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "exporter_version": EXPORTER_VERSION,
                    "schema_version": SCHEMA_VERSION,
                    "total_drawings": len(drawing_records),
                },
                "drawings": [
                    self.format_drawing_result(record) for record in drawing_records
                ],
            }

            # Write to file atomically
            self._write_json_file_atomic(batch_data, output_path)

            duration = time.time() - start_time
            file_size = output_path.stat().st_size / 1024  # KB

            logger.info(
                f"Successfully exported batch to {output_path} "
                f"({file_size:.2f} KB in {duration:.2f}s)"
            )

        except (IOError, OSError) as e:
            error_msg = f"Failed to export batch: {e}"
            logger.error(error_msg)
            raise ExportFileError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error exporting batch: {e}"
            logger.error(error_msg)
            raise JSONExportError(error_msg) from e

    def export_batch_streaming(
        self, drawing_records: List[DrawingRecord], output_path: Union[str, Path]
    ) -> None:
        """Export multiple drawings using streaming to minimize memory usage.

        Writes each drawing incrementally rather than building the entire
        data structure in memory. Ideal for large batch exports.

        Args:
            drawing_records: List of drawing records to export.
            output_path: File system path for the output JSON file.

        Raises:
            JSONExportError: If export operation fails.
            ExportFileError: If file write operation fails.

        Example:
            >>> exporter.export_batch_streaming(records, "large_batch.json")
        """
        start_time = time.time()
        output_path = Path(output_path)

        logger.info(
            f"Streaming export of {len(drawing_records)} drawings to " f"{output_path}"
        )

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Use temporary file for atomic write
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_path.parent,
                delete=False,
                suffix=".tmp",
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)

                try:
                    # Write metadata and opening bracket
                    metadata = {
                        "export_timestamp": datetime.now().isoformat(),
                        "exporter_version": EXPORTER_VERSION,
                        "schema_version": SCHEMA_VERSION,
                        "total_drawings": len(drawing_records),
                    }
                    tmp_file.write('{"_export_metadata": ')
                    json.dump(metadata, tmp_file, cls=DrawingJSONEncoder)
                    tmp_file.write(', "drawings": [')

                    # Write each drawing
                    for i, record in enumerate(drawing_records):
                        if i > 0:
                            tmp_file.write(", ")
                        json_data = self.format_drawing_result(record)
                        json.dump(
                            json_data,
                            tmp_file,
                            cls=DrawingJSONEncoder,
                            ensure_ascii=False,
                        )

                    # Close array and object
                    tmp_file.write("]}")

                    # Ensure all data is written
                    tmp_file.flush()

                    # Atomic replace
                    tmp_path.replace(output_path)

                except Exception:
                    # Clean up temp file on error
                    if tmp_path.exists():
                        tmp_path.unlink()
                    raise

            duration = time.time() - start_time
            file_size = output_path.stat().st_size / 1024  # KB

            logger.info(
                f"Successfully streamed batch to {output_path} "
                f"({file_size:.2f} KB in {duration:.2f}s)"
            )

        except (IOError, OSError) as e:
            error_msg = f"Failed to stream export batch: {e}"
            logger.error(error_msg)
            raise ExportFileError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error in streaming export: {e}"
            logger.error(error_msg)
            raise JSONExportError(error_msg) from e

    def export_to_string(
        self, drawing_record: DrawingRecord, pretty: bool = False
    ) -> str:
        """Export drawing record to JSON string.

        Useful for API responses or testing without file I/O.

        Args:
            drawing_record: Drawing record to export.
            pretty: If True, format with indentation.

        Returns:
            JSON string representation of the drawing.

        Raises:
            JSONExportError: If serialization fails.

        Example:
            >>> json_str = exporter.export_to_string(record, pretty=True)
            >>> print(json_str)
        """
        try:
            json_data = self.format_drawing_result(drawing_record)
            indent = 2 if pretty else None
            return json.dumps(
                json_data, indent=indent, cls=DrawingJSONEncoder, ensure_ascii=False
            )
        except Exception as e:
            error_msg = f"Failed to serialize drawing {drawing_record.drawing_id}: {e}"
            logger.error(error_msg)
            raise JSONExportError(error_msg, drawing_record.drawing_id) from e

    def format_drawing_result(self, drawing_record: DrawingRecord) -> Dict[str, Any]:
        """Convert DrawingRecord to a JSON-serializable dictionary.

        Transforms all nested data structures (entities, detections, etc.)
        into plain dictionaries suitable for JSON serialization. Handles
        optional fields gracefully by checking for None values.

        Args:
            drawing_record: Drawing record to format for export.

        Returns:
            Dictionary containing all drawing data in serializable format,
            including metadata, entities, detections, associations, and
            hierarchies.

        Example:
            >>> formatted = exporter.format_drawing_result(record)
            >>> print(formatted.keys())
            dict_keys(['drawing_id', 'entities', 'detections', ...])
        """
        result = {
            "drawing_id": drawing_record.drawing_id,
            "source_file": drawing_record.source_file,
            "processing_timestamp": (
                drawing_record.processing_timestamp.isoformat()
                if drawing_record.processing_timestamp
                else None
            ),
            "pipeline_version": drawing_record.pipeline_version,
            "overall_confidence": drawing_record.overall_confidence,
            "needs_review": drawing_record.needs_review,
            "status": drawing_record.status,
            "text_blocks": self._format_list(
                drawing_record.text_blocks, self._serialize_text_block
            ),
            "entities": self._format_list(
                drawing_record.entities, self._serialize_entity
            ),
            "detections": self._format_list(
                drawing_record.detections, self._serialize_detection
            ),
            "associations": self._format_list(
                drawing_record.associations, self._serialize_association
            ),
            "hierarchy": (
                self._serialize_hierarchy(drawing_record.hierarchy)
                if drawing_record.hierarchy
                else None
            ),
            "review_flags": self._format_list(
                drawing_record.review_flags, self._serialize_review_flag
            ),
        }

        return result

    def _format_list(
        self, items: Optional[List[Any]], serializer: Callable[[Any], Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format a list of items using the provided serializer function.

        Args:
            items: Optional list of items to serialize.
            serializer: Function to serialize each item.

        Returns:
            List of serialized dictionaries, or empty list if items is None.
        """
        if not items:
            return []
        return [serializer(item) for item in items]

    def _serialize_text_block(self, text_block: TextBlock) -> Dict[str, Any]:
        """Serialize a text block to a dictionary.

        Args:
            text_block: Text block object from OCR processing.

        Returns:
            Dictionary containing text block data.
        """
        return {
            "text_id": text_block.text_id,
            "content": text_block.content,
            "bbox": self._serialize_bbox_safe(text_block.bbox),
            "confidence": text_block.confidence,
            "ocr_engine": text_block.ocr_engine,
            "region_type": text_block.region_type,
        }

    def _serialize_entity(self, entity: Entity) -> Dict[str, Any]:
        """Serialize an entity to a dictionary.

        Args:
            entity: Entity object containing extracted information.

        Returns:
            Dictionary with entity data.
        """
        return {
            "entity_id": entity.entity_id,
            "entity_type": entity.entity_type.value,
            "value": entity.value,
            "original_text": entity.original_text,
            "normalized_value": entity.normalized_value,
            "confidence": entity.confidence,
            "extraction_method": entity.extraction_method,
            "source_text_id": entity.source_text_id,
            "bbox": self._serialize_bbox_safe(entity.bbox),
        }

    def _serialize_detection(self, detection: Detection) -> Dict[str, Any]:
        """Serialize a shape detection to a dictionary.

        Args:
            detection: Detection object from shape detection pipeline.

        Returns:
            Dictionary containing detection data with both bbox formats.
        """
        return {
            "detection_id": detection.detection_id,
            "class_name": detection.class_name,
            "confidence": detection.confidence,
            "bbox": self._serialize_bbox(detection.bbox),
            "bbox_normalized": {
                "x_center": detection.bbox_normalized.x_center,
                "y_center": detection.bbox_normalized.y_center,
                "width": detection.bbox_normalized.width,
                "height": detection.bbox_normalized.height,
            },
        }

    def _serialize_association(self, association: Association) -> Dict[str, Any]:
        """Serialize a text-shape association to a dictionary.

        Args:
            association: Association linking text blocks to detected shapes.

        Returns:
            Dictionary with association metadata.
        """
        return {
            "association_id": association.association_id,
            "text_id": association.text_id,
            "shape_id": association.shape_id,
            "relationship_type": association.relationship_type,
            "confidence": association.confidence,
            "distance_pixels": association.distance_pixels,
        }

    def _serialize_hierarchy(self, hierarchy: ComponentHierarchy) -> Dict[str, Any]:
        """Serialize component hierarchy to a dictionary.

        Args:
            hierarchy: Component hierarchy object describing assembly
                structure.

        Returns:
            Dictionary containing root component ID, assemblies list, and
            hierarchy tree structure.
        """
        return {
            "root_component_id": hierarchy.root_component_id,
            "assemblies": [
                {
                    "parent_shape_id": assembly.parent_shape_id,
                    "child_shape_ids": assembly.child_shape_ids,
                    "relationship_type": assembly.relationship_type,
                    "confidence": assembly.confidence,
                }
                for assembly in hierarchy.assemblies
            ],
            "hierarchy_tree": hierarchy.hierarchy_tree,
        }

    def _serialize_review_flag(self, flag: ReviewFlag) -> Dict[str, Any]:
        """Serialize a review flag to a dictionary.

        Args:
            flag: Review flag indicating issues requiring human attention.

        Returns:
            Dictionary with flag metadata.
        """
        return {
            "flag_id": flag.flag_id,
            "flag_type": flag.flag_type.value,
            "severity": flag.severity.value,
            "reason": flag.reason,
            "details": flag.details,
            "suggested_action": flag.suggested_action,
            "affected_entities": flag.affected_entities,
            "affected_shapes": flag.affected_shapes,
        }

    def _serialize_bbox(self, bbox: BoundingBox) -> Dict[str, Any]:
        """Serialize a bounding box to a dictionary.

        Args:
            bbox: Bounding box with pixel coordinates.

        Returns:
            Dictionary with x, y, width, height keys.
        """
        return {
            "x": bbox.x,
            "y": bbox.y,
            "width": bbox.width,
            "height": bbox.height,
        }

    def _serialize_bbox_safe(
        self, bbox: Optional[BoundingBox]
    ) -> Optional[Dict[str, Any]]:
        """Serialize bounding box with None handling.

        Args:
            bbox: Optional bounding box with pixel coordinates.

        Returns:
            Dictionary with coordinates, or None if input is None.
        """
        if bbox is None:
            return None
        return self._serialize_bbox(bbox)

    def _write_json_file_atomic(self, data: Dict[str, Any], output_path: Path) -> None:
        """Write JSON data to file atomically using temporary file.

        This prevents partial file creation if the write operation fails.

        Args:
            data: Dictionary to serialize to JSON.
            output_path: File path for output.

        Raises:
            ExportFileError: If file write fails.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get format settings
        indent = 2 if self.config.json_format == "pretty" else None

        # Write to temporary file first
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            delete=False,
            suffix=".tmp",
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)

            try:
                json.dump(
                    data,
                    tmp_file,
                    indent=indent,
                    cls=DrawingJSONEncoder,
                    ensure_ascii=False,
                )
                tmp_file.flush()

                # Atomic replace
                tmp_path.replace(output_path)

            except Exception:
                # Clean up temp file on error
                if tmp_path.exists():
                    tmp_path.unlink()
                raise

    def _validate_or_raise(self, json_data: Dict[str, Any], drawing_id: str) -> None:
        """Validate JSON data and raise exception if invalid.

        Args:
            json_data: Dictionary to validate.
            drawing_id: ID of the drawing being validated.

        Raises:
            SchemaValidationError: If validation fails.
        """
        is_valid, errors = self.validate_schema(json_data)
        if not is_valid:
            error_msg = (
                f"Schema validation failed for {drawing_id}: " f"{len(errors)} error(s)"
            )
            logger.error(f"{error_msg}: {errors}")
            raise SchemaValidationError(error_msg, errors, drawing_id)

    def validate_schema(self, json_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate JSON data against expected schema.

        Performs basic structural validation including required field
        presence, data type checking, and value range validation for
        confidence scores.

        Args:
            json_data: Dictionary to validate before export.

        Returns:
            Tuple containing:
                - bool: True if validation passed, False otherwise
                - List[str]: List of validation error messages
                    (empty if valid)

        Example:
            >>> is_valid, errors = exporter.validate_schema(data)
            >>> if not is_valid:
            ...     print(f"Validation failed: {errors}")
        """
        errors = []

        # Required top-level fields
        required_fields = [
            "drawing_id",
            "source_file",
            "processing_timestamp",
        ]
        for field in required_fields:
            if field not in json_data:
                errors.append(f"Missing required field: {field}")

        # Validate confidence range
        if "overall_confidence" in json_data:
            conf = json_data["overall_confidence"]
            if not isinstance(conf, (int, float)):
                errors.append(f"Invalid confidence type: {type(conf).__name__}")
            elif not (0.0 <= conf <= 1.0):
                errors.append(f"Confidence out of range [0.0, 1.0]: {conf}")

        # Validate entities
        if "entities" in json_data:
            if not isinstance(json_data["entities"], list):
                errors.append("Entities must be a list")
            else:
                for i, entity in enumerate(json_data["entities"]):
                    if not isinstance(entity, dict):
                        errors.append(f"Entity {i} must be a dictionary")
                        continue
                    if "entity_type" not in entity:
                        errors.append(f"Entity {i} missing entity_type")
                    if "value" not in entity:
                        errors.append(f"Entity {i} missing value")

        # Validate detections
        if "detections" in json_data:
            if not isinstance(json_data["detections"], list):
                errors.append("Detections must be a list")
            else:
                for i, detection in enumerate(json_data["detections"]):
                    if not isinstance(detection, dict):
                        errors.append(f"Detection {i} must be a dictionary")
                        continue
                    if "class_name" not in detection:
                        errors.append(f"Detection {i} missing class_name")
                    if "bbox" not in detection:
                        errors.append(f"Detection {i} missing bbox")

        is_valid = len(errors) == 0
        return is_valid, errors
