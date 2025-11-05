"""
JSON Exporter Module

Exports drawing results to JSON format.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

from ..database.database_manager import DrawingRecord
from ..models.data_structures import (
    Entity,
    Detection,
    Association,
    BoundingBox,
    ReviewFlag,
    ComponentHierarchy,
)


logger = logging.getLogger(__name__)


class JSONExporter:
    """Export drawing results to JSON format."""

    def __init__(self, config):
        """
        Initialize JSON exporter.

        Args:
            config: Export configuration
        """
        self.config = config
        logger.info("JSONExporter initialized")

    def export_single(self, drawing_record: DrawingRecord, output_path: str) -> None:
        """
        Export single drawing to JSON file.

        Args:
            drawing_record: Drawing record to export
            output_path: Output file path

        Raises:
            IOError: If file write fails
        """
        logger.info(f"Exporting drawing {drawing_record.drawing_id} to JSON")

        # Format drawing data
        json_data = self.format_drawing_result(drawing_record)

        # Validate schema
        is_valid, errors = self.validate_schema(json_data)
        if not is_valid:
            logger.warning(f"Schema validation failed: {errors}")

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        indent = 2 if self.config.json_format == "pretty" else None

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=indent, default=str)

        logger.info(f"Successfully exported to {output_path}")

    def export_batch(
        self, drawing_records: List[DrawingRecord], output_path: str
    ) -> None:
        """
        Export multiple drawings to single JSON file.

        Args:
            drawing_records: List of drawing records
            output_path: Output file path

        Raises:
            IOError: If file write fails
        """
        logger.info(f"Exporting batch of {len(drawing_records)} drawings to JSON")

        # Format all drawings
        batch_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_drawings": len(drawing_records),
            "drawings": [
                self.format_drawing_result(record) for record in drawing_records
            ],
        }

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        indent = 2 if self.config.json_format == "pretty" else None

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(batch_data, f, indent=indent, default=str)

        logger.info(f"Successfully exported batch to {output_path}")

    def format_drawing_result(self, drawing_record: DrawingRecord) -> Dict[str, Any]:
        """
        Convert DrawingRecord to JSON-serializable dict.

        Args:
            drawing_record: Drawing record to format

        Returns:
            Formatted dictionary
        """
        result = {
            "drawing_id": drawing_record.drawing_id,
            "source_file": drawing_record.source_file,
            "processing_timestamp": drawing_record.processing_timestamp.isoformat(),
            "pipeline_version": drawing_record.pipeline_version,
            "overall_confidence": drawing_record.overall_confidence,
            "needs_review": drawing_record.needs_review,
            "status": drawing_record.status,
            # Text blocks
            "text_blocks": (
                [self._serialize_text_block(tb) for tb in drawing_record.text_blocks]
                if drawing_record.text_blocks
                else []
            ),
            # Entities
            "entities": (
                [self._serialize_entity(e) for e in drawing_record.entities]
                if drawing_record.entities
                else []
            ),
            # Detections
            "detections": (
                [self._serialize_detection(d) for d in drawing_record.detections]
                if drawing_record.detections
                else []
            ),
            # Associations
            "associations": (
                [self._serialize_association(a) for a in drawing_record.associations]
                if drawing_record.associations
                else []
            ),
            # Hierarchy
            "hierarchy": (
                self._serialize_hierarchy(drawing_record.hierarchy)
                if drawing_record.hierarchy
                else None
            ),
            # Review flags
            "review_flags": (
                [self._serialize_review_flag(f) for f in drawing_record.review_flags]
                if drawing_record.review_flags
                else []
            ),
        }

        return result

    def _serialize_text_block(self, text_block) -> Dict[str, Any]:
        """Serialize text block."""
        return {
            "text_id": text_block.text_id,
            "content": text_block.content,
            "bbox": self._serialize_bbox(text_block.bbox),
            "confidence": text_block.confidence,
            "ocr_engine": text_block.ocr_engine,
            "region_type": text_block.region_type,
        }

    def _serialize_entity(self, entity: Entity) -> Dict[str, Any]:
        """Serialize entity."""
        return {
            "entity_id": entity.entity_id,
            "entity_type": entity.entity_type.value,
            "value": entity.value,
            "original_text": entity.original_text,
            "normalized_value": entity.normalized_value,
            "confidence": entity.confidence,
            "extraction_method": entity.extraction_method,
            "source_text_id": entity.source_text_id,
            "bbox": self._serialize_bbox(entity.bbox),
        }

    def _serialize_detection(self, detection: Detection) -> Dict[str, Any]:
        """Serialize detection."""
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
        """Serialize association."""
        return {
            "association_id": association.association_id,
            "text_id": association.text_id,
            "shape_id": association.shape_id,
            "relationship_type": association.relationship_type,
            "confidence": association.confidence,
            "distance_pixels": association.distance_pixels,
        }

    def _serialize_hierarchy(self, hierarchy: ComponentHierarchy) -> Dict[str, Any]:
        """Serialize component hierarchy."""
        return {
            "root_component_id": hierarchy.root_component_id,
            "assemblies": [
                {
                    "parent_shape_id": a.parent_shape_id,
                    "child_shape_ids": a.child_shape_ids,
                    "relationship_type": a.relationship_type,
                    "confidence": a.confidence,
                }
                for a in hierarchy.assemblies
            ],
            "hierarchy_tree": hierarchy.hierarchy_tree,
        }

    def _serialize_review_flag(self, flag: ReviewFlag) -> Dict[str, Any]:
        """Serialize review flag."""
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
        """Serialize bounding box."""
        return {"x": bbox.x, "y": bbox.y, "width": bbox.width, "height": bbox.height}

    def validate_schema(self, json_data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate JSON against schema.

        Args:
            json_data: JSON data to validate

        Returns:
            Tuple of (is_valid, error_list)
        """
        errors = []

        # Required top-level fields
        required_fields = ["drawing_id", "source_file", "processing_timestamp"]
        for field in required_fields:
            if field not in json_data:
                errors.append(f"Missing required field: {field}")

        # Validate confidence range
        if "overall_confidence" in json_data:
            conf = json_data["overall_confidence"]
            if not (0.0 <= conf <= 1.0):
                errors.append(f"Invalid confidence: {conf}")

        # Validate entities
        if "entities" in json_data:
            for i, entity in enumerate(json_data["entities"]):
                if "entity_type" not in entity:
                    errors.append(f"Entity {i} missing entity_type")
                if "value" not in entity:
                    errors.append(f"Entity {i} missing value")

        # Validate detections
        if "detections" in json_data:
            for i, detection in enumerate(json_data["detections"]):
                if "class_name" not in detection:
                    errors.append(f"Detection {i} missing class_name")
                if "bbox" not in detection:
                    errors.append(f"Detection {i} missing bbox")

        is_valid = len(errors) == 0
        return is_valid, errors
