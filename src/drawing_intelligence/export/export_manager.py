"""
Export Manager Module

Manages all export operations for drawing intelligence results.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass

from ..database.database_manager import DatabaseManager, DrawingRecord
from .json_exporter import JSONExporter
from .csv_exporter import CSVExporter
from .report_generator import ReportGenerator


logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for export operations."""

    json_format: str = "pretty"  # 'pretty' or 'compact'
    csv_delimiter: str = ","
    include_intermediate_results: bool = False
    include_images: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.json_format not in ["pretty", "compact"]:
            raise ValueError("json_format must be 'pretty' or 'compact'")


class ExportManager:
    """
    Manages all export operations.

    Coordinates JSON, CSV, and report generation for drawing results.
    """

    def __init__(self, db: DatabaseManager, config: Optional[ExportConfig] = None):
        """
        Initialize export manager.

        Args:
            db: Database manager instance
            config: Export configuration
        """
        self.db = db
        self.config = config or ExportConfig()

        # Initialize exporters
        self.json_exporter = JSONExporter(self.config)
        self.csv_exporter = CSVExporter(self.config)
        self.report_generator = ReportGenerator()

        logger.info("ExportManager initialized")

    def export_drawing_json(self, drawing_id: str, output_path: str) -> str:
        """
        Export single drawing to JSON.

        Args:
            drawing_id: Drawing identifier
            output_path: Output file path

        Returns:
            Output file path

        Raises:
            ValueError: If drawing not found
            IOError: If file write fails
        """
        logger.info(f"Exporting drawing {drawing_id} to JSON: {output_path}")

        # Get drawing from database
        drawing_record = self.db.get_drawing_by_id(drawing_id)
        if drawing_record is None:
            raise ValueError(f"Drawing {drawing_id} not found in database")

        # Export to JSON
        self.json_exporter.export_single(drawing_record, output_path)

        logger.info(f"Successfully exported drawing to {output_path}")
        return output_path

    def export_drawing_csv(self, drawing_id: str, output_dir: str) -> List[str]:
        """
        Export single drawing to multiple CSV files.

        Creates separate CSV files for:
        - Summary
        - Entities
        - Detections
        - Associations

        Args:
            drawing_id: Drawing identifier
            output_dir: Output directory path

        Returns:
            List of generated file paths

        Raises:
            ValueError: If drawing not found
            IOError: If file write fails
        """
        logger.info(f"Exporting drawing {drawing_id} to CSV: {output_dir}")

        # Get drawing from database
        drawing_record = self.db.get_drawing_by_id(drawing_id)
        if drawing_record is None:
            raise ValueError(f"Drawing {drawing_id} not found in database")

        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export to CSV files
        output_files = []

        # Summary CSV
        summary_path = output_path / f"{drawing_id}_summary.csv"
        self.csv_exporter.export_summary([drawing_record], str(summary_path))
        output_files.append(str(summary_path))

        # Entities CSV
        if drawing_record.entities:
            entities_path = output_path / f"{drawing_id}_entities.csv"
            self.csv_exporter.export_entities([drawing_record], str(entities_path))
            output_files.append(str(entities_path))

        # Detections CSV
        if drawing_record.detections:
            detections_path = output_path / f"{drawing_id}_detections.csv"
            self.csv_exporter.export_detections([drawing_record], str(detections_path))
            output_files.append(str(detections_path))

        # Associations CSV
        if drawing_record.associations:
            associations_path = output_path / f"{drawing_id}_associations.csv"
            self.csv_exporter.export_associations(
                [drawing_record], str(associations_path)
            )
            output_files.append(str(associations_path))

        logger.info(f"Successfully exported {len(output_files)} CSV files")
        return output_files

    def export_batch_json(self, drawing_ids: List[str], output_path: str) -> str:
        """
        Export multiple drawings to single JSON file.

        Args:
            drawing_ids: List of drawing identifiers
            output_path: Output file path

        Returns:
            Output file path

        Raises:
            ValueError: If any drawing not found
            IOError: If file write fails
        """
        logger.info(f"Exporting batch of {len(drawing_ids)} drawings to JSON")

        # Get all drawings
        drawing_records = []
        for drawing_id in drawing_ids:
            record = self.db.get_drawing_by_id(drawing_id)
            if record is None:
                logger.warning(f"Drawing {drawing_id} not found, skipping")
                continue
            drawing_records.append(record)

        if not drawing_records:
            raise ValueError("No valid drawings found to export")

        # Export to JSON
        self.json_exporter.export_batch(drawing_records, output_path)

        logger.info(
            f"Successfully exported {len(drawing_records)} drawings to {output_path}"
        )
        return output_path

    def export_batch_csv(self, drawing_ids: List[str], output_dir: str) -> List[str]:
        """
        Export batch to CSV files.

        Creates consolidated CSV files for all drawings:
        - batch_summary.csv
        - batch_entities.csv
        - batch_detections.csv
        - batch_associations.csv

        Args:
            drawing_ids: List of drawing identifiers
            output_dir: Output directory path

        Returns:
            List of generated file paths

        Raises:
            ValueError: If no valid drawings found
            IOError: If file write fails
        """
        logger.info(f"Exporting batch of {len(drawing_ids)} drawings to CSV")

        # Get all drawings
        drawing_records = []
        for drawing_id in drawing_ids:
            record = self.db.get_drawing_by_id(drawing_id)
            if record is None:
                logger.warning(f"Drawing {drawing_id} not found, skipping")
                continue
            drawing_records.append(record)

        if not drawing_records:
            raise ValueError("No valid drawings found to export")

        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export to CSV files
        output_files = []

        # Summary CSV
        summary_path = output_path / "batch_summary.csv"
        self.csv_exporter.export_summary(drawing_records, str(summary_path))
        output_files.append(str(summary_path))

        # Entities CSV
        entities_path = output_path / "batch_entities.csv"
        self.csv_exporter.export_entities(drawing_records, str(entities_path))
        output_files.append(str(entities_path))

        # Detections CSV
        detections_path = output_path / "batch_detections.csv"
        self.csv_exporter.export_detections(drawing_records, str(detections_path))
        output_files.append(str(detections_path))

        # Associations CSV
        associations_path = output_path / "batch_associations.csv"
        self.csv_exporter.export_associations(drawing_records, str(associations_path))
        output_files.append(str(associations_path))

        logger.info(f"Successfully exported {len(output_files)} CSV files")
        return output_files

    def generate_batch_report(
        self, batch_id: str, output_path: str, format: str = "html"
    ) -> str:
        """
        Generate comprehensive batch report.

        Args:
            batch_id: Batch identifier
            output_path: Output file path
            format: Report format ('html' or 'pdf')

        Returns:
            Report file path

        Raises:
            ValueError: If batch not found or invalid format
            IOError: If file write fails
        """
        logger.info(f"Generating batch report for {batch_id}")

        if format not in ["html", "pdf"]:
            raise ValueError("Format must be 'html' or 'pdf'")

        # Get batch drawings (implement batch tracking in database)
        # For now, use placeholder
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
        drawing_records = []

        # Generate report
        self.report_generator.generate_batch_report(
            batch_result, drawing_records, output_path, format
        )

        logger.info(f"Successfully generated batch report: {output_path}")
        return output_path

    def export_cost_report(
        self,
        start_date: datetime,
        end_date: datetime,
        output_path: str,
        format: str = "html",
    ) -> str:
        """
        Generate cost report.

        Args:
            start_date: Report start date
            end_date: Report end date
            output_path: Output file path
            format: Report format ('html' or 'pdf')

        Returns:
            Report file path

        Raises:
            ValueError: If invalid format
            IOError: If file write fails
        """
        logger.info(f"Generating cost report from {start_date} to {end_date}")

        if format not in ["html", "pdf"]:
            raise ValueError("Format must be 'html' or 'pdf'")

        # Get cost data from database
        from ..llm.cost_tracker import CostTracker

        cost_tracker = CostTracker(self.db)
        cost_report = cost_tracker.generate_cost_report(
            period="custom", start_date=start_date, end_date=end_date
        )

        # Generate report
        self.report_generator.generate_cost_report(cost_report, output_path, format)

        logger.info(f"Successfully generated cost report: {output_path}")
        return output_path

    def export_drawing_report(
        self, drawing_id: str, output_path: str, format: str = "html"
    ) -> str:
        """
        Generate detailed drawing report.

        Args:
            drawing_id: Drawing identifier
            output_path: Output file path
            format: Report format ('html' or 'pdf')

        Returns:
            Report file path

        Raises:
            ValueError: If drawing not found or invalid format
            IOError: If file write fails
        """
        logger.info(f"Generating drawing report for {drawing_id}")

        if format not in ["html", "pdf"]:
            raise ValueError("Format must be 'html' or 'pdf'")

        # Get drawing from database
        drawing_record = self.db.get_drawing_by_id(drawing_id)
        if drawing_record is None:
            raise ValueError(f"Drawing {drawing_id} not found in database")

        # Generate report
        self.report_generator.generate_drawing_report(
            drawing_record, output_path, format
        )

        logger.info(f"Successfully generated drawing report: {output_path}")
        return output_path
