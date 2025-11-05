"""
Database manager for the Drawing Intelligence System.

Manages all database operations including storage, retrieval, and queries.
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from dataclasses import asdict

from ..utils.error_handlers import DatabaseError
from .query_filters import QueryFilters


logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages all database operations.

    Uses SQLite for simplicity and zero-configuration deployment.
    """

    def __init__(self, db_path: str, schema_path: Optional[str] = None):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
            schema_path: Optional path to schema SQL file
        """
        self.db_path = db_path
        self.schema_path = schema_path or str(Path(__file__).parent / "schema.sql")

        # Ensure database directory exists
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        # Initialize connection
        self.connection: Optional[sqlite3.Connection] = None
        self._connect()

        # Initialize database schema
        self.initialize_database()

    def _connect(self) -> None:
        """Establish database connection."""
        try:
            self.connection = sqlite3.connect(
                self.db_path, check_same_thread=False  # Allow multi-threaded access
            )
            # Enable foreign keys
            self.connection.execute("PRAGMA foreign_keys = ON")
            # Enable WAL mode for better concurrency
            self.connection.execute("PRAGMA journal_mode = WAL")
            # Use Row factory for dict-like access
            self.connection.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            raise DatabaseError(
                message=f"Failed to connect to database: {e}", operation="connect"
            )

    def initialize_database(self) -> None:
        """Create tables and indexes if they don't exist."""
        try:
            with open(self.schema_path, "r") as f:
                schema_sql = f.read()

            # Execute schema (may contain multiple statements)
            self.connection.executescript(schema_sql)
            self.connection.commit()
            logger.info(f"Database initialized: {self.db_path}")
        except FileNotFoundError:
            raise DatabaseError(
                message=f"Schema file not found: {self.schema_path}",
                operation="initialize",
            )
        except sqlite3.Error as e:
            raise DatabaseError(
                message=f"Failed to initialize database: {e}", operation="initialize"
            )

    def store_drawing(self, processing_result: Any) -> str:
        """
        Store complete drawing result in database.

        Args:
            processing_result: ProcessingResult object

        Returns:
            drawing_id

        Raises:
            DatabaseError: If storage fails
        """
        try:
            cursor = self.connection.cursor()

            # Serialize component hierarchy to JSON if present
            hierarchy_json = None
            if processing_result.hierarchy:
                hierarchy_json = json.dumps(asdict(processing_result.hierarchy))

            # Insert main drawing record
            cursor.execute(
                """
                INSERT INTO drawings (
                    drawing_id, source_file, processing_timestamp,
                    pipeline_version, overall_confidence, needs_review,
                    component_hierarchy, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    processing_result.drawing_id,
                    processing_result.source_file,
                    processing_result.processing_timestamp.isoformat(),
                    processing_result.pipeline_version,
                    processing_result.overall_confidence,
                    processing_result.needs_human_review(),
                    hierarchy_json,
                    processing_result.status,
                ),
            )

            self.connection.commit()
            logger.info(f"Stored drawing: {processing_result.drawing_id}")

            return processing_result.drawing_id

        except sqlite3.IntegrityError as e:
            self.connection.rollback()
            if "UNIQUE constraint failed" in str(e):
                raise DatabaseError(
                    message=f"Drawing already exists: {processing_result.drawing_id}",
                    drawing_id=processing_result.drawing_id,
                    operation="store_drawing",
                )
            raise DatabaseError(
                message=f"Database integrity error: {e}",
                drawing_id=processing_result.drawing_id,
                operation="store_drawing",
            )
        except sqlite3.Error as e:
            self.connection.rollback()
            raise DatabaseError(
                message=f"Failed to store drawing: {e}",
                drawing_id=processing_result.drawing_id,
                operation="store_drawing",
            )

    def store_text_blocks(self, drawing_id: str, text_blocks: List[Any]) -> None:
        """
        Store OCR text blocks for a drawing.

        Args:
            drawing_id: Drawing identifier
            text_blocks: List of TextBlock objects
        """
        try:
            cursor = self.connection.cursor()

            for block in text_blocks:
                cursor.execute(
                    """
                    INSERT INTO text_extractions (
                        text_id, drawing_id, content, bbox_x, bbox_y,
                        bbox_width, bbox_height, confidence, ocr_engine, region_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        block.text_id,
                        drawing_id,
                        block.content,
                        block.bbox.x,
                        block.bbox.y,
                        block.bbox.width,
                        block.bbox.height,
                        block.confidence,
                        block.ocr_engine,
                        block.region_type,
                    ),
                )

            self.connection.commit()
            logger.debug(f"Stored {len(text_blocks)} text blocks for {drawing_id}")

        except sqlite3.Error as e:
            self.connection.rollback()
            raise DatabaseError(
                message=f"Failed to store text blocks: {e}",
                drawing_id=drawing_id,
                operation="store_text_blocks",
            )

    def store_entities(self, drawing_id: str, entities: List[Any]) -> None:
        """
        Store extracted entities for a drawing.

        Args:
            drawing_id: Drawing identifier
            entities: List of Entity objects
        """
        try:
            cursor = self.connection.cursor()

            for entity in entities:
                cursor.execute(
                    """
                    INSERT INTO entities (
                        entity_id, drawing_id, entity_type, value,
                        normalized_value, confidence, extraction_method,
                        original_text, source_text_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        entity.entity_id,
                        drawing_id,
                        (
                            entity.entity_type.value
                            if hasattr(entity.entity_type, "value")
                            else entity.entity_type
                        ),
                        entity.value,
                        json.dumps(entity.normalized_value),
                        entity.confidence,
                        entity.extraction_method,
                        entity.original_text,
                        entity.source_text_id,
                    ),
                )

            self.connection.commit()
            logger.debug(f"Stored {len(entities)} entities for {drawing_id}")

        except sqlite3.Error as e:
            self.connection.rollback()
            raise DatabaseError(
                message=f"Failed to store entities: {e}",
                drawing_id=drawing_id,
                operation="store_entities",
            )

    def store_detections(self, drawing_id: str, detections: List[Any]) -> None:
        """
        Store shape detections for a drawing.

        Args:
            drawing_id: Drawing identifier
            detections: List of Detection objects
        """
        try:
            cursor = self.connection.cursor()

            for detection in detections:
                cursor.execute(
                    """
                    INSERT INTO shape_detections (
                        detection_id, drawing_id, class_name, confidence,
                        bbox_x, bbox_y, bbox_width, bbox_height, model_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        detection.detection_id,
                        drawing_id,
                        detection.class_name,
                        detection.confidence,
                        detection.bbox.x,
                        detection.bbox.y,
                        detection.bbox.width,
                        detection.bbox.height,
                        getattr(detection, "model_version", "unknown"),
                    ),
                )

            self.connection.commit()
            logger.debug(f"Stored {len(detections)} detections for {drawing_id}")

        except sqlite3.Error as e:
            self.connection.rollback()
            raise DatabaseError(
                message=f"Failed to store detections: {e}",
                drawing_id=drawing_id,
                operation="store_detections",
            )

    def store_associations(self, drawing_id: str, associations: List[Any]) -> None:
        """
        Store text-shape associations for a drawing.

        Args:
            drawing_id: Drawing identifier
            associations: List of Association objects
        """
        try:
            cursor = self.connection.cursor()

            for assoc in associations:
                cursor.execute(
                    """
                    INSERT INTO text_shape_associations (
                        association_id, drawing_id, text_id, detection_id,
                        relationship_type, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        assoc.association_id,
                        drawing_id,
                        assoc.text_id,
                        assoc.shape_id,
                        assoc.relationship_type,
                        assoc.confidence,
                    ),
                )

            self.connection.commit()
            logger.debug(f"Stored {len(associations)} associations for {drawing_id}")

        except sqlite3.Error as e:
            self.connection.rollback()
            raise DatabaseError(
                message=f"Failed to store associations: {e}",
                drawing_id=drawing_id,
                operation="store_associations",
            )

    def store_hierarchy(self, drawing_id: str, hierarchy: Any) -> None:
        """
        Store component hierarchy for a drawing.

        Args:
            drawing_id: Drawing identifier
            hierarchy: ComponentHierarchy object
        """
        try:
            hierarchy_json = json.dumps(asdict(hierarchy))

            cursor = self.connection.cursor()
            cursor.execute(
                """
                UPDATE drawings
                SET component_hierarchy = ?
                WHERE drawing_id = ?
            """,
                (hierarchy_json, drawing_id),
            )

            self.connection.commit()
            logger.debug(f"Stored hierarchy for {drawing_id}")

        except sqlite3.Error as e:
            self.connection.rollback()
            raise DatabaseError(
                message=f"Failed to store hierarchy: {e}",
                drawing_id=drawing_id,
                operation="store_hierarchy",
            )

    def store_llm_usage(
        self,
        drawing_id: str,
        provider: str,
        model: str,
        tokens: Any,
        cost: float,
        use_case: str,
    ) -> None:
        """
        Store LLM usage record.

        Args:
            drawing_id: Drawing identifier
            provider: LLM provider name
            model: Model identifier
            tokens: TokenUsage object
            cost: Cost in USD
            use_case: Use case type
        """
        try:
            from ..utils.file_utils import generate_unique_id

            cursor = self.connection.cursor()
            cursor.execute(
                """
                INSERT INTO llm_usage (
                    usage_id, drawing_id, use_case, provider, model,
                    tokens_input, tokens_output, cost_usd, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    generate_unique_id("LLM"),
                    drawing_id,
                    use_case,
                    provider,
                    model,
                    tokens.input_tokens,
                    tokens.output_tokens,
                    cost,
                    datetime.now().isoformat(),
                ),
            )

            self.connection.commit()

        except sqlite3.Error as e:
            self.connection.rollback()
            raise DatabaseError(
                message=f"Failed to store LLM usage: {e}",
                drawing_id=drawing_id,
                operation="store_llm_usage",
            )

    def store_audit_entry(
        self,
        drawing_id: str,
        stage: str,
        status: str,
        duration: float,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Store processing audit entry.

        Args:
            drawing_id: Drawing identifier
            stage: Processing stage name
            status: Status ('success', 'failed', 'skipped')
            duration: Duration in seconds
            error_message: Optional error message if failed
        """
        try:
            from ..utils.file_utils import generate_unique_id

            cursor = self.connection.cursor()
            cursor.execute(
                """
                INSERT INTO processing_audit (
                    audit_id, drawing_id, stage, status, 
                    duration_seconds, error_message, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    generate_unique_id("AUD"),
                    drawing_id,
                    stage,
                    status,
                    duration,
                    error_message,
                    datetime.now().isoformat(),
                ),
            )

            self.connection.commit()

        except sqlite3.Error as e:
            self.connection.rollback()
            logger.error(f"Failed to store audit entry: {e}")
            # Don't raise - audit failures shouldn't stop processing

    def get_drawing_by_id(self, drawing_id: str) -> Optional[Dict[str, Any]]:
        """
        Get drawing record by ID.

        Args:
            drawing_id: Drawing identifier

        Returns:
            Dictionary with drawing data or None if not found
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                SELECT * FROM drawings WHERE drawing_id = ?
            """,
                (drawing_id,),
            )

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

        except sqlite3.Error as e:
            raise DatabaseError(
                message=f"Failed to get drawing: {e}",
                drawing_id=drawing_id,
                operation="get_drawing_by_id",
            )

    def query_drawings(self, filters: QueryFilters) -> List[Dict[str, Any]]:
        """
        Query drawings with filters.

        Args:
            filters: QueryFilters object

        Returns:
            List of drawing records matching filters
        """
        try:
            cursor = self.connection.cursor()

            # Build query
            query = "SELECT * FROM drawings WHERE 1=1"
            params = []

            if filters.status:
                query += " AND status = ?"
                params.append(filters.status)

            if filters.needs_review is not None:
                query += " AND needs_review = ?"
                params.append(1 if filters.needs_review else 0)

            if filters.min_confidence is not None:
                query += " AND overall_confidence >= ?"
                params.append(filters.min_confidence)

            if filters.max_confidence is not None:
                query += " AND overall_confidence <= ?"
                params.append(filters.max_confidence)

            if filters.date_from:
                query += " AND processing_timestamp >= ?"
                params.append(filters.date_from.isoformat())

            if filters.date_to:
                query += " AND processing_timestamp <= ?"
                params.append(filters.date_to.isoformat())

            # Add ordering
            query += " ORDER BY processing_timestamp DESC"

            # Add pagination
            query += " LIMIT ? OFFSET ?"
            params.extend([filters.limit, filters.offset])

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

        except sqlite3.Error as e:
            raise DatabaseError(
                message=f"Failed to query drawings: {e}", operation="query_drawings"
            )

    def get_drawings_for_review(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get drawings flagged for review.

        Args:
            limit: Maximum number of drawings to return

        Returns:
            List of drawing records needing review
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                SELECT * FROM drawings
                WHERE needs_review = 1
                ORDER BY processing_timestamp DESC
                LIMIT ?
            """,
                (limit,),
            )

            return [dict(row) for row in cursor.fetchall()]

        except sqlite3.Error as e:
            raise DatabaseError(
                message=f"Failed to get drawings for review: {e}",
                operation="get_drawings_for_review",
            )

    def update_drawing_status(self, drawing_id: str, status: str) -> None:
        """
        Update drawing processing status.

        Args:
            drawing_id: Drawing identifier
            status: New status
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                UPDATE drawings
                SET status = ?
                WHERE drawing_id = ?
            """,
                (status, drawing_id),
            )

            self.connection.commit()

        except sqlite3.Error as e:
            self.connection.rollback()
            raise DatabaseError(
                message=f"Failed to update drawing status: {e}",
                drawing_id=drawing_id,
                operation="update_drawing_status",
            )

    def delete_drawing(self, drawing_id: str) -> None:
        """
        Delete drawing and all related data.

        Args:
            drawing_id: Drawing identifier
        """
        try:
            cursor = self.connection.cursor()

            # Foreign keys will cascade delete related records
            cursor.execute("DELETE FROM drawings WHERE drawing_id = ?", (drawing_id,))

            self.connection.commit()
            logger.info(f"Deleted drawing: {drawing_id}")

        except sqlite3.Error as e:
            self.connection.rollback()
            raise DatabaseError(
                message=f"Failed to delete drawing: {e}",
                drawing_id=drawing_id,
                operation="delete_drawing",
            )

    def get_llm_usage(
        self, start_date: datetime, end_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Get LLM usage records for date range.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of usage records
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                SELECT * FROM llm_usage
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
            """,
                (start_date.isoformat(), end_date.isoformat()),
            )

            return [dict(row) for row in cursor.fetchall()]

        except sqlite3.Error as e:
            raise DatabaseError(
                message=f"Failed to get LLM usage: {e}", operation="get_llm_usage"
            )

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with database statistics
        """
        try:
            cursor = self.connection.cursor()

            stats = {}

            # Total drawings
            cursor.execute("SELECT COUNT(*) FROM drawings")
            stats["total_drawings"] = cursor.fetchone()[0]

            # Drawings needing review
            cursor.execute("SELECT COUNT(*) FROM drawings WHERE needs_review = 1")
            stats["drawings_needs_review"] = cursor.fetchone()[0]

            # Total entities
            cursor.execute("SELECT COUNT(*) FROM entities")
            stats["total_entities"] = cursor.fetchone()[0]

            # Total detections
            cursor.execute("SELECT COUNT(*) FROM shape_detections")
            stats["total_detections"] = cursor.fetchone()[0]

            # Total LLM calls
            cursor.execute("SELECT COUNT(*) FROM llm_usage")
            stats["total_llm_calls"] = cursor.fetchone()[0]

            # Total LLM cost
            cursor.execute("SELECT SUM(cost_usd) FROM llm_usage")
            result = cursor.fetchone()[0]
            stats["total_llm_cost"] = result if result else 0.0

            # Database size
            db_size_bytes = Path(self.db_path).stat().st_size
            stats["database_size_mb"] = round(db_size_bytes / (1024 * 1024), 2)

            # Date range
            cursor.execute(
                "SELECT MIN(processing_timestamp), MAX(processing_timestamp) FROM drawings"
            )
            min_date, max_date = cursor.fetchone()
            stats["oldest_drawing"] = min_date
            stats["newest_drawing"] = max_date

            return stats

        except sqlite3.Error as e:
            raise DatabaseError(
                message=f"Failed to get database stats: {e}",
                operation="get_database_stats",
            )

    def backup_database(self, backup_path: str) -> bool:
        """
        Create database backup.

        Args:
            backup_path: Path for backup file

        Returns:
            True if successful
        """
        try:
            import shutil

            # Ensure backup directory exists
            backup_dir = Path(backup_path).parent
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Copy database file
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to: {backup_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            return False

    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
