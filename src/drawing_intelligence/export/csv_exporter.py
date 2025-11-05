"""
CSV Exporter Module

Exports drawing results to CSV format.
"""

import csv
import logging
from pathlib import Path
from typing import List, Dict, Any

from ..database.database_manager import DrawingRecord


logger = logging.getLogger(__name__)


class CSVExporter:
    """Export drawing results to CSV format."""

    def __init__(self, config):
        """
        Initialize CSV exporter.

        Args:
            config: Export configuration
        """
        self.config = config
        self.delimiter = config.csv_delimiter
        logger.info("CSVExporter initialized")

    def export_summary(
        self, drawing_records: List[DrawingRecord], output_path: str
    ) -> None:
        """
        Export summary CSV (one row per drawing).

        Args:
            drawing_records: List of drawing records
            output_path: Output file path

        Raises:
            IOError: If file write fails
        """
        logger.info(f"Exporting summary CSV for {len(drawing_records)} drawings")

        # Define headers
        headers = [
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

        # Flatten drawing data
        rows = []
        for record in drawing_records:
            row = self.flatten_drawing_data(record)
            rows.append(row)

        # Write CSV
        self._write_csv(rows, output_path, headers)

        logger.info(f"Successfully exported summary CSV to {output_path}")

    def export_entities(
        self, drawing_records: List[DrawingRecord], output_path: str
    ) -> None:
        """
        Export entities CSV (one row per entity).

        Args:
            drawing_records: List of drawing records
            output_path: Output file path

        Raises:
            IOError: If file write fails
        """
        logger.info(f"Exporting entities CSV")

        # Define headers
        headers = [
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

        # Flatten entity data
        rows = []
        for record in drawing_records:
            if not record.entities:
                continue

            for entity in record.entities:
                row = {
                    "drawing_id": record.drawing_id,
                    "source_file": record.source_file,
                    "entity_id": entity.entity_id,
                    "entity_type": entity.entity_type.value,
                    "value": entity.value,
                    "original_text": entity.original_text,
                    "confidence": entity.confidence,
                    "extraction_method": entity.extraction_method,
                    "bbox_x": entity.bbox.x,
                    "bbox_y": entity.bbox.y,
                    "bbox_width": entity.bbox.width,
                    "bbox_height": entity.bbox.height,
                }
                rows.append(row)

        # Write CSV
        self._write_csv(rows, output_path, headers)

        logger.info(f"Successfully exported {len(rows)} entities to {output_path}")

    def export_detections(
        self, drawing_records: List[DrawingRecord], output_path: str
    ) -> None:
        """
        Export detections CSV (one row per detection).

        Args:
            drawing_records: List of drawing records
            output_path: Output file path

        Raises:
            IOError: If file write fails
        """
        logger.info(f"Exporting detections CSV")

        # Define headers
        headers = [
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

        # Flatten detection data
        rows = []
        for record in drawing_records:
            if not record.detections:
                continue

            for detection in record.detections:
                row = {
                    "drawing_id": record.drawing_id,
                    "source_file": record.source_file,
                    "detection_id": detection.detection_id,
                    "class_name": detection.class_name,
                    "confidence": detection.confidence,
                    "bbox_x": detection.bbox.x,
                    "bbox_y": detection.bbox.y,
                    "bbox_width": detection.bbox.width,
                    "bbox_height": detection.bbox.height,
                    "bbox_norm_x_center": detection.bbox_normalized.x_center,
                    "bbox_norm_y_center": detection.bbox_normalized.y_center,
                    "bbox_norm_width": detection.bbox_normalized.width,
                    "bbox_norm_height": detection.bbox_normalized.height,
                }
                rows.append(row)

        # Write CSV
        self._write_csv(rows, output_path, headers)

        logger.info(f"Successfully exported {len(rows)} detections to {output_path}")

    def export_associations(
        self, drawing_records: List[DrawingRecord], output_path: str
    ) -> None:
        """
        Export associations CSV.

        Args:
            drawing_records: List of drawing records
            output_path: Output file path

        Raises:
            IOError: If file write fails
        """
        logger.info(f"Exporting associations CSV")

        # Define headers
        headers = [
            "drawing_id",
            "source_file",
            "association_id",
            "text_id",
            "shape_id",
            "relationship_type",
            "confidence",
            "distance_pixels",
        ]

        # Flatten association data
        rows = []
        for record in drawing_records:
            if not record.associations:
                continue

            for association in record.associations:
                row = {
                    "drawing_id": record.drawing_id,
                    "source_file": record.source_file,
                    "association_id": association.association_id,
                    "text_id": association.text_id,
                    "shape_id": association.shape_id,
                    "relationship_type": association.relationship_type,
                    "confidence": association.confidence,
                    "distance_pixels": association.distance_pixels,
                }
                rows.append(row)

        # Write CSV
        self._write_csv(rows, output_path, headers)

        logger.info(f"Successfully exported {len(rows)} associations to {output_path}")

    def flatten_drawing_data(self, drawing_record: DrawingRecord) -> Dict[str, Any]:
        """
        Flatten drawing data to single-level dict for CSV.

        Args:
            drawing_record: Drawing record to flatten

        Returns:
            Flattened dictionary
        """
        # Check for part number
        has_part_number = False
        if drawing_record.entities:
            from ..models.data_structures import EntityType

            has_part_number = any(
                e.entity_type == EntityType.PART_NUMBER for e in drawing_record.entities
            )

        # Check for title block (placeholder - depends on implementation)
        has_title_block = False

        return {
            "drawing_id": drawing_record.drawing_id,
            "source_file": drawing_record.source_file,
            "processing_timestamp": drawing_record.processing_timestamp.isoformat(),
            "overall_confidence": drawing_record.overall_confidence,
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

    def _write_csv(
        self, data: List[Dict], output_path: str, headers: List[str]
    ) -> None:
        """
        Write list of dicts to CSV file.

        Args:
            data: List of dictionaries to write
            output_path: Output file path
            headers: CSV headers

        Raises:
            IOError: If file write fails
        """
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write CSV
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=headers,
                delimiter=self.delimiter,
                extrasaction="ignore",  # Ignore extra fields
            )
            writer.writeheader()
            writer.writerows(data)
