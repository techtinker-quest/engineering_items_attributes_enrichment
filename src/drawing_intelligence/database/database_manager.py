"""
Database manager for the Drawing Intelligence System.

This module provides a comprehensive database management layer for storing and
retrieving drawing processing results, including OCR text blocks, extracted
entities, shape detections, component hierarchies, and LLM usage tracking.

The DatabaseManager uses SQLite with Write-Ahead Logging (WAL) mode for improved
concurrency. Thread safety is ensured via thread-local connections.

Classes:
    DatabaseManager: Main database operations manager.

Typical usage example:
    with DatabaseManager(db_path="./data/drawings.db") as db_manager:
        drawing_id = db_manager.store_drawing(processing_result)
        db_manager.store_entities(drawing_id, entities)
"""

import sqlite3
import json
import logging
import threading
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple, Sequence
from pathlib import Path
from dataclasses import asdict
from contextlib import contextmanager
from enum import Enum

from ..models.data_structures import (
    ProcessingResult,
    TextBlock,
    Entity,
    Detection,
    Association,
    ComponentHierarchy,
    TokenUsage,
)
from ..utils.error_handlers import DatabaseError
from ..utils.file_utils import generate_unique_id
from .query_filters import QueryFilter


logger = logging.getLogger(__name__)


# Constants
DEFAULT_BATCH_SIZE: int = 5000
DEFAULT_CONNECTION_TIMEOUT: float = 30.0
UNKNOWN_MODEL_VERSION: str = "unknown"
ID_PREFIX_LLM: str = "LLM"
ID_PREFIX_AUDIT: str = "AUD"


class ProcessingStatus(Enum):
    """Processing status enumeration for audit and status tracking."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class DatabaseManager:
    """
    Manages all database operations for the Drawing Intelligence System.

    This class provides a comprehensive interface for storing and querying
    drawing processing results in SQLite. It handles the complete lifecycle
    of drawing data including text blocks, entities, shape detections,
    associations, hierarchies, and audit logs.

    Thread Safety:
        This implementation uses thread-local storage for SQLite connections
        to ensure thread safety in multi-threaded batch processing scenarios.
        Each thread maintains its own connection, eliminating race conditions
        while maintaining the benefits of WAL mode for concurrency.

    Context Manager:
        DatabaseManager implements the context manager protocol. Use it with
        'with' statements to ensure proper cleanup:

            with DatabaseManager(db_path) as db:
                db.store_drawing(result)

    Attributes:
        db_path: Path to the SQLite database file.
        schema_path: Path to the SQL schema definition file.
        _local: Thread-local storage for connections.
        _lock: Threading lock for initialization operations.

    Raises:
        DatabaseError: For all database operation failures.
        FileNotFoundError: If schema file cannot be located during
            initialization.
    """

    # SQL Query Constants
    _SQL_INSERT_DRAWING = """
        INSERT INTO drawings (
            drawing_id, source_file, processing_timestamp,
            pipeline_version, overall_confidence, needs_review,
            component_hierarchy, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """

    _SQL_INSERT_TEXT_BLOCK = """
        INSERT INTO text_extractions (
            text_id, drawing_id, content, bbox_x, bbox_y,
            bbox_width, bbox_height, confidence, ocr_engine,
            region_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    _SQL_INSERT_ENTITY = """
        INSERT INTO entities (
            entity_id, drawing_id, entity_type, value,
            normalized_value, confidence, extraction_method,
            original_text, source_text_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    _SQL_INSERT_DETECTION = """
        INSERT INTO shape_detections (
            detection_id, drawing_id, class_name, confidence,
            bbox_x, bbox_y, bbox_width, bbox_height, model_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    _SQL_INSERT_ASSOCIATION = """
        INSERT INTO text_shape_associations (
            association_id, drawing_id, text_id, detection_id,
            relationship_type, confidence
        ) VALUES (?, ?, ?, ?, ?, ?)
    """

    _SQL_INSERT_LLM_USAGE = """
        INSERT INTO llm_usage (
            usage_id, drawing_id, use_case, provider, model,
            tokens_input, tokens_output, cost_usd, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    _SQL_INSERT_AUDIT = """
        INSERT INTO processing_audit (
            audit_id, drawing_id, stage, status, 
            duration_seconds, error_message, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """

    def __init__(self, db_path: str, schema_path: Optional[str] = None) -> None:
        """
        Initialize database manager and establish connection.

        Creates the database directory if it doesn't exist, establishes a
        connection with WAL mode enabled, and initializes the schema.

        Args:
            db_path: Path to SQLite database file. Parent directories will be
                created if they don't exist.
            schema_path: Optional path to schema SQL file. If not provided,
                defaults to schema.sql in the same directory as this module.

        Raises:
            DatabaseError: If connection establishment fails.
            FileNotFoundError: If schema_path is invalid.
        """
        self.db_path: str = db_path
        self.schema_path: str = schema_path or str(Path(__file__).parent / "schema.sql")

        # Thread-local storage for connections
        self._local: threading.local = threading.local()
        self._lock: threading.Lock = threading.Lock()

        # Ensure database directory exists
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database schema
        self.initialize_database()

    def __enter__(self) -> "DatabaseManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - closes connection for current thread."""
        self.close()

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get or create thread-local database connection.

        Returns:
            Thread-specific SQLite connection.

        Raises:
            DatabaseError: If connection cannot be established.
        """
        if not hasattr(self._local, "connection") or self._local.connection is None:
            try:
                self._local.connection = sqlite3.connect(
                    self.db_path,
                    timeout=DEFAULT_CONNECTION_TIMEOUT,
                    check_same_thread=True,
                )
                # Enable foreign keys
                self._local.connection.execute("PRAGMA foreign_keys = ON")

                # Enable WAL mode for better concurrency
                cursor = self._local.connection.execute("PRAGMA journal_mode = WAL")
                mode = cursor.fetchone()[0]
                if mode.upper() != "WAL":
                    logger.warning(f"Failed to enable WAL mode, using {mode} instead")

                # Set synchronous mode for optimal WAL performance
                self._local.connection.execute("PRAGMA synchronous = NORMAL")

                # Use Row factory for dict-like access
                self._local.connection.row_factory = sqlite3.Row

            except sqlite3.Error as e:
                raise DatabaseError(
                    message=f"Failed to connect to database: {e}",
                    operation="connect",
                ) from e
        return self._local.connection

    def _validate_connection(self) -> bool:
        """
        Validate that the connection is still alive.

        Returns:
            True if connection is valid, False otherwise.
        """
        try:
            conn = self._get_connection()
            conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    @contextmanager
    def _transaction(self):
        """
        Context manager for database transactions.

        Yields:
            sqlite3.Connection: Database connection with active transaction.

        Raises:
            DatabaseError: If transaction fails.
        """
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @contextmanager
    def transaction(self):
        """
        Public transaction context manager for multi-step operations.

        Yields:
            sqlite3.Connection: Database connection with active transaction.

        Example:
            with db_manager.transaction():
                db_manager.store_drawing(result)
                db_manager.store_entities(drawing_id, entities)
        """
        with self._transaction() as conn:
            yield conn

    def _serialize_hierarchy(
        self, hierarchy: Optional[ComponentHierarchy]
    ) -> Optional[str]:
        """
        Serialize component hierarchy to JSON.

        Args:
            hierarchy: ComponentHierarchy object or None.

        Returns:
            JSON string representation or None if hierarchy is None.

        Raises:
            DatabaseError: If serialization fails.
        """
        if hierarchy is None:
            return None
        try:
            return json.dumps(asdict(hierarchy))
        except (TypeError, ValueError) as e:
            raise DatabaseError(
                message=f"Failed to serialize hierarchy: {e}",
                operation="_serialize_hierarchy",
            ) from e

    def _serialize_normalized_value(self, value: Dict[str, Any]) -> str:
        """
        Serialize normalized value dictionary to JSON.

        Args:
            value: Normalized value dictionary.

        Returns:
            JSON string representation.

        Raises:
            DatabaseError: If serialization fails.
        """
        try:
            return json.dumps(value)
        except (TypeError, ValueError) as e:
            raise DatabaseError(
                message=f"Failed to serialize normalized value: {e}",
                operation="_serialize_normalized_value",
            ) from e

    def _batch_execute(
        self,
        cursor: sqlite3.Cursor,
        query: str,
        data: List[Tuple[Any, ...]],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """
        Execute batch insert in chunks to prevent memory issues.

        Args:
            cursor: Database cursor.
            query: SQL INSERT query.
            data: List of tuples containing row data.
            batch_size: Number of rows per batch (default: 5000).
        """
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            cursor.executemany(query, batch)

    def initialize_database(self) -> None:
        """
        Create database tables and indexes from schema file.

        Reads and executes the SQL schema file to create all required tables,
        indexes, and constraints. This operation is idempotent - it can be
        safely called multiple times as the schema uses CREATE TABLE IF NOT
        EXISTS.

        Raises:
            DatabaseError: If schema execution fails.
            FileNotFoundError: If schema file doesn't exist.
        """
        try:
            with open(self.schema_path, "r") as f:
                schema_sql = f.read()

            # Use lock to ensure only one thread initializes
            with self._lock:
                conn = self._get_connection()
                conn.executescript(schema_sql)
                conn.commit()
                logger.info(f"Database initialized: {self.db_path}")

        except FileNotFoundError:
            raise DatabaseError(
                message=f"Schema file not found: {self.schema_path}",
                operation="initialize",
            )
        except sqlite3.Error as e:
            raise DatabaseError(
                message=f"Failed to initialize database: {e}",
                operation="initialize",
            ) from e

    def store_drawing(self, processing_result: ProcessingResult) -> str:
        """
        Store complete drawing processing result in database.

        Inserts the main drawing record with metadata, confidence scores, and
        serialized component hierarchy. This method should be called before
        storing related entities, text blocks, or detections.

        Args:
            processing_result: Complete ProcessingResult object containing all
                drawing metadata, confidence scores, and hierarchy.

        Returns:
            The drawing_id that was stored.

        Raises:
            DatabaseError: If insertion fails or drawing_id already exists.
            ValueError: If required fields are missing or invalid.

        Note:
            The component_hierarchy is serialized to JSON using
            dataclasses.asdict(). Ensure all nested objects are
            JSON-serializable.
        """
        # Input validation
        if not processing_result.drawing_id:
            raise ValueError("drawing_id cannot be empty")
        if not processing_result.source_file:
            raise ValueError("source_file cannot be empty")

        try:
            with self._transaction() as conn:
                cursor = conn.cursor()

                # Serialize component hierarchy to JSON if present
                hierarchy_json = self._serialize_hierarchy(processing_result.hierarchy)

                # Insert main drawing record
                cursor.execute(
                    self._SQL_INSERT_DRAWING,
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

                logger.info(f"Stored drawing: {processing_result.drawing_id}")
                return processing_result.drawing_id

        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                raise DatabaseError(
                    message=(
                        f"Drawing already exists: " f"{processing_result.drawing_id}"
                    ),
                    drawing_id=processing_result.drawing_id,
                    operation="store_drawing",
                ) from e
            raise DatabaseError(
                message=f"Database integrity error: {e}",
                drawing_id=processing_result.drawing_id,
                operation="store_drawing",
            ) from e
        except sqlite3.Error as e:
            raise DatabaseError(
                message=f"Failed to store drawing: {e}",
                drawing_id=processing_result.drawing_id,
                operation="store_drawing",
            ) from e

    def store_text_blocks(self, drawing_id: str, text_blocks: List[TextBlock]) -> None:
        """
        Store OCR text blocks for a drawing.

        Inserts all text blocks extracted during OCR processing, including
        content, bounding boxes, confidence scores, and region types.

        Args:
            drawing_id: Drawing identifier (must exist in drawings table).
            text_blocks: List of TextBlock objects from OCR pipeline.

        Raises:
            DatabaseError: If insertion fails or drawing_id doesn't exist.

        Note:
            Uses executemany() with chunking for optimized batch insertion.
        """
        if not text_blocks:
            return

        try:
            with self._transaction() as conn:
                cursor = conn.cursor()

                # Prepare batch data
                batch_data: List[Tuple[Any, ...]] = [
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
                    )
                    for block in text_blocks
                ]

                self._batch_execute(cursor, self._SQL_INSERT_TEXT_BLOCK, batch_data)

                logger.debug(f"Stored {len(text_blocks)} text blocks for {drawing_id}")

        except sqlite3.Error as e:
            raise DatabaseError(
                message=f"Failed to store text blocks: {e}",
                drawing_id=drawing_id,
                operation="store_text_blocks",
            ) from e

    def store_entities(self, drawing_id: str, entities: List[Entity]) -> None:
        """
        Store extracted entities for a drawing.

        Inserts all entities (part numbers, dimensions, materials, etc.)
        extracted from the drawing, including normalized values and confidence
        scores.

        Args:
            drawing_id: Drawing identifier (must exist in drawings table).
            entities: List of Entity objects from entity extraction pipeline.

        Raises:
            DatabaseError: If insertion fails or drawing_id doesn't exist.

        Note:
            The normalized_value dict is serialized to JSON. Entity types are
            stored as strings (enum values if available, otherwise raw strings).
        """
        if not entities:
            return

        try:
            with self._transaction() as conn:
                cursor = conn.cursor()

                # Prepare batch data
                batch_data: List[Tuple[Any, ...]] = [
                    (
                        entity.entity_id,
                        drawing_id,
                        (
                            entity.entity_type.value
                            if hasattr(entity.entity_type, "value")
                            else str(entity.entity_type)
                        ),
                        entity.value,
                        self._serialize_normalized_value(entity.normalized_value),
                        entity.confidence,
                        entity.extraction_method,
                        entity.original_text,
                        entity.source_text_id,
                    )
                    for entity in entities
                ]

                self._batch_execute(cursor, self._SQL_INSERT_ENTITY, batch_data)

                logger.debug(f"Stored {len(entities)} entities for {drawing_id}")

        except sqlite3.Error as e:
            raise DatabaseError(
                message=f"Failed to store entities: {e}",
                drawing_id=drawing_id,
                operation="store_entities",
            ) from e

    def store_detections(self, drawing_id: str, detections: List[Detection]) -> None:
        """
        Store shape detections for a drawing.

        Inserts all shape/component detections from the YOLOv8 model, including
        class names, bounding boxes, and confidence scores.

        Args:
            drawing_id: Drawing identifier (must exist in drawings table).
            detections: List of Detection objects from shape detection pipeline.

        Raises:
            DatabaseError: If insertion fails or drawing_id doesn't exist.

        Note:
            Only pixel-based bounding boxes are stored (normalized boxes are
            computed on-the-fly when needed). Model version defaults to
            "unknown" if not provided in Detection object.
        """
        if not detections:
            return

        try:
            with self._transaction() as conn:
                cursor = conn.cursor()

                # Prepare batch data
                batch_data: List[Tuple[Any, ...]] = [
                    (
                        detection.detection_id,
                        drawing_id,
                        detection.class_name,
                        detection.confidence,
                        detection.bbox.x,
                        detection.bbox.y,
                        detection.bbox.width,
                        detection.bbox.height,
                        getattr(detection, "model_version", UNKNOWN_MODEL_VERSION),
                    )
                    for detection in detections
                ]

                self._batch_execute(cursor, self._SQL_INSERT_DETECTION, batch_data)

                logger.debug(f"Stored {len(detections)} detections for {drawing_id}")

        except sqlite3.Error as e:
            raise DatabaseError(
                message=f"Failed to store detections: {e}",
                drawing_id=drawing_id,
                operation="store_detections",
            ) from e

    def store_associations(
        self, drawing_id: str, associations: List[Association]
    ) -> None:
        """
        Store text-shape associations for a drawing.

        Inserts associations between text blocks and detected shapes, capturing
        relationships like dimensions, labels, and notes.

        Args:
            drawing_id: Drawing identifier (must exist in drawings table).
            associations: List of Association objects from data association
                pipeline.

        Raises:
            DatabaseError: If insertion fails or foreign key constraints are
                violated.
        """
        if not associations:
            return

        try:
            with self._transaction() as conn:
                cursor = conn.cursor()

                # Prepare batch data
                batch_data: List[Tuple[Any, ...]] = [
                    (
                        assoc.association_id,
                        drawing_id,
                        assoc.text_id,
                        assoc.shape_id,
                        assoc.relationship_type,
                        assoc.confidence,
                    )
                    for assoc in associations
                ]

                self._batch_execute(cursor, self._SQL_INSERT_ASSOCIATION, batch_data)

                logger.debug(
                    f"Stored {len(associations)} associations for {drawing_id}"
                )

        except sqlite3.Error as e:
            raise DatabaseError(
                message=f"Failed to store associations: {e}",
                drawing_id=drawing_id,
                operation="store_associations",
            ) from e

    def store_hierarchy(self, drawing_id: str, hierarchy: ComponentHierarchy) -> None:
        """
        Store component hierarchy for a drawing.

        Updates the drawings table with a serialized component hierarchy,
        replacing any existing hierarchy for this drawing.

        Args:
            drawing_id: Drawing identifier (must exist in drawings table).
            hierarchy: ComponentHierarchy object representing assembly structure.

        Raises:
            DatabaseError: If update fails or drawing_id doesn't exist.

        Note:
            The hierarchy is serialized to JSON using dataclasses.asdict(). This
            operation replaces (not appends to) any existing hierarchy.
        """
        try:
            hierarchy_json = self._serialize_hierarchy(hierarchy)

            with self._transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE drawings
                    SET component_hierarchy = ?
                    WHERE drawing_id = ?
                """,
                    (hierarchy_json, drawing_id),
                )

                logger.debug(f"Stored hierarchy for {drawing_id}")

        except sqlite3.Error as e:
            raise DatabaseError(
                message=f"Failed to store hierarchy: {e}",
                drawing_id=drawing_id,
                operation="store_hierarchy",
            ) from e

    def store_llm_usage(
        self,
        drawing_id: str,
        provider: str,
        model: str,
        tokens: TokenUsage,
        cost: float,
        use_case: str,
    ) -> None:
        """
        Store LLM API usage record for cost tracking.

        Records all commercial LLM API calls with token counts and costs for
        budget monitoring and cost attribution.

        Args:
            drawing_id: Drawing identifier (must exist in drawings table).
            provider: LLM provider name (e.g., "openai", "anthropic").
            model: Model identifier (e.g., "gpt-4o-2024-08-06").
            tokens: TokenUsage object with input/output token counts.
            cost: Cost in USD for this API call.
            use_case: Use case type (e.g., "ocr_verification",
                "entity_extraction").

        Raises:
            DatabaseError: If insertion fails.

        Note:
            Generates a unique usage_id with "LLM" prefix. Timestamp is set to
            current time in ISO format with UTC timezone.
        """
        try:
            with self._transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    self._SQL_INSERT_LLM_USAGE,
                    (
                        generate_unique_id(ID_PREFIX_LLM),
                        drawing_id,
                        use_case,
                        provider,
                        model,
                        tokens.input_tokens,
                        tokens.output_tokens,
                        cost,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )

        except sqlite3.Error as e:
            raise DatabaseError(
                message=f"Failed to store LLM usage: {e}",
                drawing_id=drawing_id,
                operation="store_llm_usage",
            ) from e

    def store_audit_entry(
        self,
        drawing_id: str,
        stage: str,
        status: str,
        duration: float,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Store processing audit entry for pipeline tracking.

        Records the execution of each pipeline stage with status, duration, and
        error details for debugging and performance monitoring.

        Args:
            drawing_id: Drawing identifier.
            stage: Processing stage name (e.g., "pdf_processing",
                "ocr_extraction").
            status: Status code (use ProcessingStatus enum values).
            duration: Execution duration in seconds.
            error_message: Optional error message if status is "failed".

        Note:
            This method does NOT raise exceptions on failure - it only logs
            errors. This prevents audit logging failures from disrupting the
            main pipeline.
        """
        try:
            with self._transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    self._SQL_INSERT_AUDIT,
                    (
                        generate_unique_id(ID_PREFIX_AUDIT),
                        drawing_id,
                        stage,
                        status,
                        duration,
                        error_message,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )

        except sqlite3.Error as e:
            logger.error(
                f"Failed to store audit entry for {drawing_id}: {e}",
                exc_info=True,
            )
            # Don't raise - audit failures shouldn't stop processing

    def get_drawing_by_id(self, drawing_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve drawing record by identifier.

        Args:
            drawing_id: Drawing identifier to look up.

        Returns:
            Dictionary containing all drawing table columns, or None if not
            found.

        Raises:
            DatabaseError: If query execution fails.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM drawings WHERE drawing_id = ?",
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
            ) from e

    def query_drawings(self, filters: QueryFilter) -> List[Dict[str, Any]]:
        """
        Query drawings with flexible filtering and pagination.

        Supports filtering by status, confidence range, review flag, and date
        range, with built-in pagination support.

        Args:
            filters: QueryFilter object specifying filter criteria, ordering,
                and pagination parameters.

        Returns:
            List of drawing records (as dicts) matching filter criteria, ordered
            by processing_timestamp descending.

        Raises:
            DatabaseError: If query execution fails.
            ValueError: If filter parameters are invalid.

        Note:
            Results are always ordered by processing_timestamp DESC. Pagination
            is controlled via filters.limit and filters.offset.
        """
        # Validate pagination parameters
        if filters.limit < 0:
            raise ValueError("filters.limit must be non-negative")
        if filters.offset < 0:
            raise ValueError("filters.offset must be non-negative")

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Build query with parameterized filters
            query = "SELECT * FROM drawings WHERE 1=1"
            params: List[Any] = []

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

            # Add fixed ordering (not dynamic to prevent injection)
            query += " ORDER BY processing_timestamp DESC"

            # Add pagination
            query += " LIMIT ? OFFSET ?"
            params.extend([filters.limit, filters.offset])

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

        except sqlite3.Error as e:
            raise DatabaseError(
                message=f"Failed to query drawings: {e}",
                operation="query_drawings",
            ) from e

    def query_drawings_count(self, filters: QueryFilter) -> int:
        """
        Get total count of drawings matching filters (for pagination).

        Args:
            filters: QueryFilter object specifying filter criteria.

        Returns:
            Total count of matching drawings.

        Raises:
            DatabaseError: If query execution fails.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Build count query with same filters as query_drawings
            query = "SELECT COUNT(*) FROM drawings WHERE 1=1"
            params: List[Any] = []

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

            cursor.execute(query, params)
            return cursor.fetchone()[0]

        except sqlite3.Error as e:
            raise DatabaseError(
                message=f"Failed to count drawings: {e}",
                operation="query_drawings_count",
            ) from e

    def get_drawings_for_review(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get drawings flagged for manual review.

        Retrieves drawings where needs_review=True, ordered by most recent
        first.

        Args:
            limit: Maximum number of drawings to return (default: 50).

        Returns:
            List of drawing records needing review.

        Raises:
            DatabaseError: If query execution fails.
            ValueError: If limit is negative.
        """
        if limit < 0:
            raise ValueError("limit must be non-negative")

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
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
            ) from e

    def update_drawing_status(self, drawing_id: str, status: str) -> None:
        """
        Update drawing processing status.

        Args:
            drawing_id: Drawing identifier to update.
            status: New status value (recommend using ProcessingStatus enum
                values: "completed", "failed", "in_progress", etc.).

        Raises:
            DatabaseError: If update fails.

        Note:
            This method does not validate status values. Consider using
            ProcessingStatus enum for type safety.
        """
        try:
            with self._transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE drawings SET status = ? WHERE drawing_id = ?",
                    (status, drawing_id),
                )

        except sqlite3.Error as e:
            raise DatabaseError(
                message=f"Failed to update drawing status: {e}",
                drawing_id=drawing_id,
                operation="update_drawing_status",
            ) from e

    def delete_drawing(self, drawing_id: str) -> None:
        """
        Delete drawing and all related data via cascade.

        Removes the drawing record and all foreign key related records
        (entities, text blocks, detections, associations, audit entries,
        LLM usage).

        Args:
            drawing_id: Drawing identifier to delete.

        Raises:
            DatabaseError: If deletion fails.

        Warning:
            This operation is irreversible and cascades to all related tables.
            Consider implementing soft deletes for production use.
        """
        try:
            with self._transaction() as conn:
                cursor = conn.cursor()

                # Foreign keys will cascade delete related records
                cursor.execute(
                    "DELETE FROM drawings WHERE drawing_id = ?", (drawing_id,)
                )

                logger.info(f"Deleted drawing: {drawing_id}")

        except sqlite3.Error as e:
            raise DatabaseError(
                message=f"Failed to delete drawing: {e}",
                drawing_id=drawing_id,
                operation="delete_drawing",
            ) from e

    def get_llm_usage(
        self, start_date: datetime, end_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Get LLM usage records for a date range.

        Retrieves all LLM API call records between start_date and end_date for
        cost reporting and budget analysis.

        Args:
            start_date: Start of date range (inclusive).
            end_date: End of date range (inclusive).

        Returns:
            List of LLM usage records ordered by timestamp descending.

        Raises:
            DatabaseError: If query execution fails.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
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
                message=f"Failed to get LLM usage: {e}",
                operation="get_llm_usage",
            ) from e

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics.

        Computes aggregate statistics including counts, costs, and date ranges
        for monitoring and reporting.

        Returns:
            Dictionary containing:
                - total_drawings: Total number of drawings processed
                - drawings_needs_review: Count of drawings flagged for review
                - total_entities: Total entities extracted across all drawings
                - total_detections: Total shape detections across all drawings
                - total_llm_calls: Total LLM API calls made
                - total_llm_cost: Total LLM cost in USD
                - database_size_mb: Database file size in megabytes
                - oldest_drawing: Timestamp of oldest drawing (or None)
                - newest_drawing: Timestamp of newest drawing (or None)

        Raises:
            DatabaseError: If any statistics query fails.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            stats: Dict[str, Any] = {}

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
                """SELECT MIN(processing_timestamp), MAX(processing_timestamp)
                FROM drawings"""
            )
            min_date, max_date = cursor.fetchone()
            stats["oldest_drawing"] = min_date
            stats["newest_drawing"] = max_date

            return stats

        except sqlite3.Error as e:
            raise DatabaseError(
                message=f"Failed to get database stats: {e}",
                operation="get_database_stats",
            ) from e

    def backup_database(self, backup_path: str) -> bool:
        """
        Create a consistent backup using SQLite's backup API.

        Uses the SQLite backup API to create a consistent snapshot of the
        database, even during active writes. This is safer than file copying.

        Args:
            backup_path: Destination path for backup file.

        Returns:
            True if backup succeeded, False otherwise.

        Note:
            This method uses SQLite's backup API which ensures consistency
            during concurrent writes. The backup is performed page-by-page.
        """
        try:
            # Ensure backup directory exists
            backup_dir = Path(backup_path).parent
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Create backup connection
            source_conn = self._get_connection()
            backup_conn = sqlite3.connect(backup_path)

            # Use SQLite backup API
            with backup_conn:
                source_conn.backup(backup_conn)

            backup_conn.close()
            logger.info(f"Database backed up to: {backup_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to backup database: {e}", exc_info=True)
            return False

    def close(self) -> None:
        """
        Close the database connection for the current thread.

        Should be called when the DatabaseManager is no longer needed to release
        resources. After calling close(), no further database operations can be
        performed without reconnecting (which happens automatically on next
        operation).
        """
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
            logger.debug("Database connection closed for current thread")

    def close_all(self) -> None:
        """
        Close all database connections across all threads.

        This method should be called during application shutdown to ensure
        all connections are properly closed. Note that it can only close
        connections from the current thread - connections in other threads
        must be closed by those threads.

        Warning:
            This is a best-effort cleanup. Thread-local connections in other
            threads cannot be accessed from here.
        """
        self.close()
        logger.info("Database manager shutdown complete")
