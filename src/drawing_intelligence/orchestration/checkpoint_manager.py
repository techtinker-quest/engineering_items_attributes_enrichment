# orchestration/checkpoint_manager.py
"""Dual checkpoint system for drawing processing pipeline.

This module provides checkpoint management at two levels:
1. Intra-drawing: Save state after each processing stage within a single
   drawing
2. Batch: Save state periodically during batch processing of multiple
   drawings

The checkpoint system enables graceful recovery from failures and supports
resuming long-running operations from the last known good state.

Features:
    - Atomic writes with checksums for data integrity
    - File locking for concurrent access safety
    - Automatic checkpoint rotation and cleanup
    - Schema versioning and migration support
    - Optional compression for large checkpoints
    - Retry logic for transient failures

Example:
    Basic usage for intra-drawing checkpoints::

        manager = CheckpointManager(checkpoint_dir="./checkpoints")

        # Save checkpoint after completing a stage
        checkpoint_id = manager.save_intra_drawing_checkpoint(
            drawing_id="DWG-001",
            source_file="drawing.pdf",
            current_stage=ProcessingStage.OCR_EXTRACTION,
            completed_stages=[ProcessingStage.PDF_PROCESSING],
            intermediate_results={"text_blocks": 45}
        )

        # Resume from checkpoint
        checkpoint = manager.load_intra_drawing_checkpoint("DWG-001")
        if checkpoint:
            print(f"Resume from: {checkpoint.current_stage}")
"""
import gzip
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging

try:
    import portalocker

    HAS_PORTALOCKER = True
except ImportError:
    HAS_PORTALOCKER = False

from ..models.data_structures import ProcessingStage

logger = logging.getLogger(__name__)

# Constants
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S_%f"
CHECKPOINT_SCHEMA_VERSION = "1.0"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 0.5
COMPRESSION_THRESHOLD_BYTES = 10240  # 10KB


def _validate_identifier(identifier: str, name: str) -> None:
    """Validate that an identifier is safe (no path traversal).

    Args:
        identifier: The string to validate.
        name: Name of the parameter (for error messages).

    Raises:
        ValueError: If identifier contains unsafe characters.
    """
    if not identifier or not isinstance(identifier, str):
        raise ValueError(f"{name} must be a non-empty string")

    # Check for path traversal attempts
    if ".." in identifier or "/" in identifier or "\\" in identifier:
        raise ValueError(f"{name} contains invalid characters (path traversal attempt)")

    # Check for other unsafe characters
    if not re.match(r"^[a-zA-Z0-9_\-]+$", identifier):
        raise ValueError(
            f"{name} must contain only alphanumeric, underscore, "
            f"and hyphen characters"
        )


def _compute_checksum(data: str) -> str:
    """Compute SHA256 checksum of data.

    Args:
        data: String data to checksum.

    Returns:
        Hexadecimal checksum string.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _validate_json_structure(data: Dict[str, Any], required_fields: List[str]) -> None:
    """Validate that JSON data has required fields.

    Args:
        data: Dictionary to validate.
        required_fields: List of required field names.

    Raises:
        ValueError: If required fields are missing.
    """
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")


@dataclass
class IntraDrawingCheckpoint:
    """Checkpoint for a single drawing's processing state.

    Captures the complete state of a drawing at a specific processing stage,
    enabling recovery if processing is interrupted. Stores both metadata
    and intermediate processing results.

    Attributes:
        drawing_id: Unique identifier for the drawing.
        source_file: Original PDF filename.
        current_stage: The most recently completed processing stage.
        completed_stages: List of all stages completed so far.
        intermediate_results: Stage-specific data (OCR results, detections).
        timestamp: UTC ISO format timestamp when checkpoint was created.
        checkpoint_id: Unique identifier for this checkpoint instance.
        schema_version: Checkpoint format version for future compatibility.
        checksum: SHA256 checksum of checkpoint data for integrity.

    Example:
        >>> checkpoint = IntraDrawingCheckpoint(
        ...     drawing_id="DWG-001",
        ...     source_file="drawing.pdf",
        ...     current_stage=ProcessingStage.OCR_EXTRACTION,
        ...     completed_stages=[ProcessingStage.PDF_PROCESSING],
        ...     intermediate_results={"text_blocks": 45},
        ...     timestamp="2025-11-08T10:30:00.000000+00:00",
        ...     checkpoint_id="550e8400-e29b-41d4-a716-446655440000",
        ...     schema_version="1.0",
        ...     checksum="abc123..."
        ... )
    """

    drawing_id: str
    source_file: str
    current_stage: ProcessingStage
    completed_stages: List[ProcessingStage]
    intermediate_results: Dict[str, Any]
    timestamp: str
    checkpoint_id: str
    schema_version: str = CHECKPOINT_SCHEMA_VERSION
    checksum: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary format for serialization.

        Converts enum values to strings to ensure JSON compatibility.
        Excludes checksum field from serialization (computed separately).

        Returns:
            Dictionary representation of the checkpoint with all enums
            converted to string values.
        """
        data = asdict(self)
        # Convert enums to strings
        data["current_stage"] = self.current_stage.value
        data["completed_stages"] = [s.value for s in self.completed_stages]
        # Checksum is computed separately
        data.pop("checksum", None)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntraDrawingCheckpoint":
        """Reconstruct checkpoint from dictionary data.

        Converts string values back to enum types during deserialization.
        Handles schema migration for older versions.

        Args:
            data: Dictionary containing checkpoint data with string enum
                values.

        Returns:
            Reconstructed IntraDrawingCheckpoint instance.

        Raises:
            ValueError: If enum conversion fails or required fields missing.
        """
        # Validate required fields
        required = [
            "drawing_id",
            "source_file",
            "current_stage",
            "completed_stages",
            "intermediate_results",
            "timestamp",
            "checkpoint_id",
        ]
        _validate_json_structure(data, required)

        # Handle schema migration
        schema_version = data.get("schema_version", "1.0")
        if schema_version != CHECKPOINT_SCHEMA_VERSION:
            logger.warning(
                f"Loading checkpoint with old schema version "
                f"{schema_version}, current is {CHECKPOINT_SCHEMA_VERSION}"
            )
            # Add migration logic here when schema changes

        # Convert strings back to enums
        data["current_stage"] = ProcessingStage(data["current_stage"])
        data["completed_stages"] = [
            ProcessingStage(s) for s in data["completed_stages"]
        ]

        # Extract and store checksum separately
        stored_checksum = data.pop("checksum", "")

        checkpoint = cls(**data)
        checkpoint.checksum = stored_checksum

        return checkpoint


@dataclass
class BatchCheckpoint:
    """Checkpoint for batch processing state across multiple drawings.

    Tracks progress through a batch of drawings, including completion status,
    failures, and cost metrics. Enables resuming batch jobs after
    interruptions.

    Attributes:
        batch_id: Unique identifier for this batch processing run.
        total_drawings: Total number of drawings in the batch.
        completed_count: Number of drawings successfully processed.
        failed_count: Number of drawings that failed processing.
        completed_drawing_ids: List of successfully processed drawing IDs.
        failed_drawing_ids: List of failed drawing IDs.
        pending_drawing_ids: List of drawings not yet processed.
        current_drawing_id: ID of drawing currently being processed (if any).
        timestamp: UTC ISO format timestamp when checkpoint was created.
        checkpoint_id: Unique identifier for this checkpoint instance.
        total_llm_cost: Cumulative LLM API cost in USD for this batch.
        average_processing_time_seconds: Mean processing time per drawing.
        schema_version: Checkpoint format version for future compatibility.
        checksum: SHA256 checksum of checkpoint data for integrity.
    """

    batch_id: str
    total_drawings: int
    completed_count: int
    failed_count: int
    completed_drawing_ids: List[str]
    failed_drawing_ids: List[str]
    pending_drawing_ids: List[str]
    current_drawing_id: Optional[str]
    timestamp: str
    checkpoint_id: str
    total_llm_cost: float
    average_processing_time_seconds: float
    schema_version: str = CHECKPOINT_SCHEMA_VERSION
    checksum: str = ""

    def __post_init__(self) -> None:
        """Validate internal consistency after initialization."""
        # Validate counts match list lengths
        if self.completed_count != len(self.completed_drawing_ids):
            raise ValueError(
                f"completed_count ({self.completed_count}) does not match "
                f"completed_drawing_ids length "
                f"({len(self.completed_drawing_ids)})"
            )
        if self.failed_count != len(self.failed_drawing_ids):
            raise ValueError(
                f"failed_count ({self.failed_count}) does not match "
                f"failed_drawing_ids length ({len(self.failed_drawing_ids)})"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary format for serialization.

        Excludes checksum field from serialization (computed separately).

        Returns:
            Dictionary representation of the checkpoint.
        """
        data = asdict(self)
        # Checksum is computed separately
        data.pop("checksum", None)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchCheckpoint":
        """Reconstruct checkpoint from dictionary data.

        Handles schema migration for older versions.

        Args:
            data: Dictionary containing checkpoint data.

        Returns:
            Reconstructed BatchCheckpoint instance.

        Raises:
            ValueError: If required fields missing or validation fails.
        """
        # Validate required fields
        required = [
            "batch_id",
            "total_drawings",
            "completed_count",
            "failed_count",
            "completed_drawing_ids",
            "failed_drawing_ids",
            "pending_drawing_ids",
            "timestamp",
            "checkpoint_id",
            "total_llm_cost",
            "average_processing_time_seconds",
        ]
        _validate_json_structure(data, required)

        # Handle schema migration
        schema_version = data.get("schema_version", "1.0")
        if schema_version != CHECKPOINT_SCHEMA_VERSION:
            logger.warning(
                f"Loading checkpoint with old schema version "
                f"{schema_version}, current is {CHECKPOINT_SCHEMA_VERSION}"
            )
            # Add migration logic here when schema changes

        # Extract and store checksum separately
        stored_checksum = data.pop("checksum", "")

        checkpoint = cls(**data)
        checkpoint.checksum = stored_checksum

        return checkpoint


class CheckpointManager:
    """Manages persistence and recovery of processing checkpoints.

    Provides checkpoint management at two granularities:
    - Intra-drawing: Stage-level checkpoints within a single drawing
    - Batch: Periodic checkpoints during batch processing

    Checkpoints are stored as JSON files (optionally compressed) in separate
    subdirectories with automatic cleanup, file locking, checksums, and
    retry logic for robustness.

    Attributes:
        checkpoint_dir: Root directory for checkpoint storage.
        intra_drawing_dir: Subdirectory for intra-drawing checkpoints.
        batch_dir: Subdirectory for batch checkpoints.
        max_checkpoints_per_id: Maximum checkpoints to keep per ID.
        checkpoint_ttl_days: Time-to-live for checkpoints in days.
        enable_compression: Whether to compress large checkpoints.

    Example:
        >>> manager = CheckpointManager(
        ...     checkpoint_dir="./checkpoints",
        ...     max_checkpoints_per_id=5,
        ...     checkpoint_ttl_days=7
        ... )
        >>> checkpoint_id = manager.save_intra_drawing_checkpoint(
        ...     drawing_id="DWG-001",
        ...     source_file="drawing.pdf",
        ...     current_stage=ProcessingStage.OCR_EXTRACTION,
        ...     completed_stages=[ProcessingStage.PDF_PROCESSING],
        ...     intermediate_results={"text_blocks": 45}
        ... )
    """

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        max_checkpoints_per_id: int = 10,
        checkpoint_ttl_days: int = 7,
        enable_compression: bool = True,
    ) -> None:
        """Initialize the checkpoint manager.

        Creates the checkpoint directory structure if it doesn't exist.

        Args:
            checkpoint_dir: Root directory path for storing checkpoints.
                If None, uses system temp directory + 'checkpoints'.
            max_checkpoints_per_id: Maximum number of checkpoints to keep
                per drawing_id or batch_id. Older ones are deleted.
            checkpoint_ttl_days: Time-to-live for checkpoints in days.
                Checkpoints older than this are cleaned up.
            enable_compression: Enable gzip compression for large
                checkpoints (>10KB).
        """
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(tempfile.gettempdir(), "checkpoints")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Separate directories for different checkpoint types
        self.intra_drawing_dir = self.checkpoint_dir / "intra_drawing"
        self.batch_dir = self.checkpoint_dir / "batch"

        self.intra_drawing_dir.mkdir(exist_ok=True)
        self.batch_dir.mkdir(exist_ok=True)

        # Configuration
        self.max_checkpoints_per_id = max_checkpoints_per_id
        self.checkpoint_ttl_days = checkpoint_ttl_days
        self.enable_compression = enable_compression

    def _acquire_lock(self, file_path: Path) -> Any:
        """Acquire file lock for safe concurrent access.

        Args:
            file_path: Path to the file to lock.

        Returns:
            Lock context manager or dummy context if portalocker unavailable.
        """
        if HAS_PORTALOCKER:
            # Use portalocker for cross-platform file locking
            lock_file = open(file_path, "a")
            portalocker.lock(lock_file, portalocker.LOCK_EX)
            return lock_file
        else:
            # Fallback: no locking (log warning)
            logger.warning(
                "portalocker not available, file locking disabled. "
                "Install with: pip install portalocker"
            )
            # Return dummy context manager
            from contextlib import nullcontext

            return nullcontext()

    def _atomic_write_json(
        self,
        target_path: Path,
        data: Dict[str, Any],
        compress: bool = False,
    ) -> str:
        """Write JSON data atomically with checksum.

        Uses a temporary file and atomic rename to ensure the checkpoint
        file is never left in a partially written state. Computes checksum
        for data integrity verification.

        Args:
            target_path: Destination path for the JSON file.
            data: Dictionary to serialize as JSON.
            compress: Whether to gzip compress the output.

        Returns:
            Checksum (SHA256 hex) of the written data.

        Raises:
            IOError: If file write or rename fails.
            OSError: If permissions or disk space issues occur.
        """
        # Serialize to JSON
        json_str = json.dumps(data, indent=2)
        checksum = _compute_checksum(json_str)

        # Add checksum to data
        data_with_checksum = data.copy()
        data_with_checksum["checksum"] = checksum
        json_str_with_checksum = json.dumps(data_with_checksum, indent=2)

        # Determine if compression is needed
        should_compress = (
            compress
            and self.enable_compression
            and len(json_str_with_checksum) > COMPRESSION_THRESHOLD_BYTES
        )

        # Use NamedTemporaryFile for cleaner temp file handling
        suffix = ".json.gz" if should_compress else ".json"

        with tempfile.NamedTemporaryFile(
            mode="wb" if should_compress else "w",
            dir=target_path.parent,
            suffix=suffix,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

            try:
                if should_compress:
                    with gzip.open(temp_file.name, "wt", encoding="utf-8") as gz:
                        gz.write(json_str_with_checksum)
                else:
                    temp_file.write(json_str_with_checksum)

                # Ensure data is flushed to disk
                if hasattr(temp_file, "flush"):
                    temp_file.flush()
                    os.fsync(temp_file.fileno())

            except Exception:
                # Clean up temp file on error
                try:
                    temp_path.unlink()
                except OSError:
                    pass
                raise

        # Atomic rename
        try:
            os.replace(temp_path, target_path)
        except Exception:
            # Clean up if rename fails
            try:
                temp_path.unlink()
            except OSError:
                pass
            raise

        return checksum

    def _read_json_with_verification(self, file_path: Path) -> Dict[str, Any]:
        """Read JSON file and verify checksum.

        Args:
            file_path: Path to JSON file (may be gzipped).

        Returns:
            Parsed JSON data dictionary.

        Raises:
            ValueError: If checksum verification fails or invalid JSON.
            IOError: If file cannot be read.
        """
        # Check file extension
        if not (file_path.suffix == ".json" or file_path.name.endswith(".json.gz")):
            raise ValueError(f"Invalid file extension: {file_path}")

        # Determine if file is compressed
        is_compressed = file_path.name.endswith(".json.gz")

        # Read file
        with open(file_path, "rb" if is_compressed else "r") as f:
            if is_compressed:
                with gzip.open(f, "rt", encoding="utf-8") as gz:
                    json_str = gz.read()
            else:
                json_str = f.read()

        # Parse JSON
        data = json.loads(json_str)

        # Verify checksum if present
        stored_checksum = data.get("checksum")
        if stored_checksum:
            # Recompute checksum without the checksum field
            data_without_checksum = {k: v for k, v in data.items() if k != "checksum"}
            recomputed_json = json.dumps(data_without_checksum, indent=2)
            computed_checksum = _compute_checksum(recomputed_json)

            if computed_checksum != stored_checksum:
                logger.error(
                    f"Checksum mismatch for {file_path}: "
                    f"stored={stored_checksum[:8]}..., "
                    f"computed={computed_checksum[:8]}..."
                )
                raise ValueError("Checkpoint data corrupted (checksum failed)")

        return data

    def _retry_operation(self, operation: callable, operation_name: str) -> Any:
        """Retry an operation with exponential backoff.

        Args:
            operation: Callable to retry.
            operation_name: Name for logging.

        Returns:
            Result of the operation.

        Raises:
            Exception: Re-raises last exception if all retries fail.
        """
        last_exception = None

        for attempt in range(MAX_RETRIES):
            try:
                return operation()
            except (IOError, OSError) as e:
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY_SECONDS * (2**attempt)
                    logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/"
                        f"{MAX_RETRIES}): {e}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"{operation_name} failed after {MAX_RETRIES} " f"attempts: {e}"
                    )

        raise last_exception

    def _cleanup_old_checkpoints(self, directory: Path, id_pattern: str) -> None:
        """Remove old checkpoints based on TTL and max count.

        Args:
            directory: Directory to clean up.
            id_pattern: Glob pattern for checkpoint files.
        """
        checkpoints = list(directory.glob(id_pattern))

        if not checkpoints:
            return

        # Filter by TTL
        cutoff_time = datetime.now(timezone.utc) - timedelta(
            days=self.checkpoint_ttl_days
        )

        valid_checkpoints = []
        for cp_file in checkpoints:
            # Parse timestamp from filename
            try:
                timestamp_str = self._extract_timestamp_from_filename(cp_file.stem)
                cp_time = datetime.strptime(timestamp_str, TIMESTAMP_FORMAT).replace(
                    tzinfo=timezone.utc
                )

                if cp_time < cutoff_time:
                    logger.info(f"Deleting expired checkpoint: {cp_file.name}")
                    cp_file.unlink()
                else:
                    valid_checkpoints.append((cp_file, cp_time))
            except (ValueError, IndexError):
                # Keep files we can't parse
                valid_checkpoints.append((cp_file, datetime.now(timezone.utc)))

        # Keep only max_checkpoints_per_id most recent
        if len(valid_checkpoints) > self.max_checkpoints_per_id:
            # Sort by timestamp (newest first)
            valid_checkpoints.sort(key=lambda x: x[1], reverse=True)

            # Delete oldest checkpoints
            for cp_file, _ in valid_checkpoints[self.max_checkpoints_per_id :]:
                logger.info(f"Deleting old checkpoint (quota): {cp_file.name}")
                cp_file.unlink()

    def _extract_timestamp_from_filename(self, filename: str) -> str:
        """Extract timestamp from checkpoint filename.

        Args:
            filename: Checkpoint filename (without extension).

        Returns:
            Timestamp string in TIMESTAMP_FORMAT.

        Raises:
            ValueError: If timestamp cannot be extracted.
        """
        # Expected format: {id}_{stage}_{timestamp}_{uuid}
        # or: {id}_{timestamp}_{uuid}
        parts = filename.split("_")

        # Timestamp is in format YYYYMMDD_HHMMSS_ffffff
        # Look for pattern matching this format
        for i in range(len(parts) - 2):
            candidate = f"{parts[i]}_{parts[i+1]}_{parts[i+2]}"
            if re.match(r"\d{8}_\d{6}_\d{6}", candidate):
                return candidate

        raise ValueError(f"Cannot extract timestamp from: {filename}")

    def _get_latest_checkpoint_path(
        self, directory: Path, id_prefix: str
    ) -> Optional[Path]:
        """Find latest checkpoint file by parsing embedded timestamp.

        Args:
            directory: Directory to search.
            id_prefix: Prefix to match (drawing_id or batch_id).

        Returns:
            Path to latest checkpoint file, or None if not found.
        """
        pattern = f"{id_prefix}_*.json*"
        checkpoints = list(directory.glob(pattern))

        if not checkpoints:
            return None

        # Parse timestamps and find latest
        latest_file = None
        latest_time = None

        for cp_file in checkpoints:
            try:
                timestamp_str = self._extract_timestamp_from_filename(cp_file.stem)
                cp_time = datetime.strptime(timestamp_str, TIMESTAMP_FORMAT).replace(
                    tzinfo=timezone.utc
                )

                if latest_time is None or cp_time > latest_time:
                    latest_time = cp_time
                    latest_file = cp_file
            except (ValueError, IndexError):
                logger.warning(
                    f"Cannot parse timestamp from checkpoint: " f"{cp_file.name}"
                )

        return latest_file

    # ============ INTRA-DRAWING CHECKPOINTS ============

    def save_intra_drawing_checkpoint(
        self,
        drawing_id: str,
        source_file: str,
        current_stage: ProcessingStage,
        completed_stages: List[ProcessingStage],
        intermediate_results: Dict[str, Any],
    ) -> str:
        """Save checkpoint after completing a processing stage.

        Creates a timestamped checkpoint file containing the complete
        processing state. Useful for resuming drawings that crash
        mid-processing. Uses atomic writes with checksums and file locking.
        Automatically includes current_stage in completed_stages.

        Args:
            drawing_id: Unique identifier for the drawing.
            source_file: Original PDF filename.
            current_stage: The stage just completed.
            completed_stages: List of stages completed before current stage.
            intermediate_results: Stage-specific processing results and data.

        Returns:
            Unique checkpoint_id (UUID) for the saved checkpoint.

        Raises:
            ValueError: If drawing_id is invalid.
            IOError: If checkpoint file cannot be written after retries.

        Example:
            >>> manager.save_intra_drawing_checkpoint(
            ...     drawing_id="DWG-001",
            ...     source_file="drawing.pdf",
            ...     current_stage=ProcessingStage.OCR_EXTRACTION,
            ...     completed_stages=[ProcessingStage.PDF_PROCESSING],
            ...     intermediate_results={"text_blocks": 45, "pages": 3}
            ... )
            '550e8400-e29b-41d4-a716-446655440000'
        """
        # Validate inputs
        _validate_identifier(drawing_id, "drawing_id")

        # Ensure current_stage is in completed_stages
        all_completed = completed_stages.copy()
        if current_stage not in all_completed:
            all_completed.append(current_stage)

        # Generate deterministic identifiers
        timestamp_now = datetime.now(timezone.utc)
        timestamp_str = timestamp_now.strftime(TIMESTAMP_FORMAT)
        checkpoint_uuid = str(uuid.uuid4())
        checkpoint_id = (
            f"{drawing_id}_{current_stage.value}_{timestamp_str}_" f"{checkpoint_uuid}"
        )

        checkpoint = IntraDrawingCheckpoint(
            drawing_id=drawing_id,
            source_file=source_file,
            current_stage=current_stage,
            completed_stages=all_completed,
            intermediate_results=intermediate_results,
            timestamp=timestamp_now.isoformat(),
            checkpoint_id=checkpoint_id,
        )

        checkpoint_path = self.intra_drawing_dir / f"{checkpoint_id}.json"

        def _save():
            with self._acquire_lock(checkpoint_path):
                checksum = self._atomic_write_json(
                    checkpoint_path, checkpoint.to_dict(), compress=True
                )
                checkpoint.checksum = checksum

        try:
            self._retry_operation(_save, "Save intra-drawing checkpoint")

            logger.info(
                "Intra-drawing checkpoint saved",
                extra={
                    "checkpoint_id": checkpoint_id,
                    "drawing_id": drawing_id,
                    "stage": current_stage.value,
                    "path": str(checkpoint_path),
                },
            )

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints(
                self.intra_drawing_dir, f"{drawing_id}_*.json*"
            )

        except Exception as e:
            logger.error(
                f"Failed to save intra-drawing checkpoint " f"for {drawing_id}: {e}",
                extra={"drawing_id": drawing_id, "stage": current_stage.value},
            )
            raise

        return checkpoint_uuid

    def load_intra_drawing_checkpoint(
        self, drawing_id: str
    ) -> Optional[IntraDrawingCheckpoint]:
        """Load the most recent checkpoint for a drawing.

        Searches for all checkpoints matching the drawing_id and returns
        the most recently created one based on embedded timestamp.
        Verifies checksum for data integrity.

        Args:
            drawing_id: Unique identifier for the drawing.

        Returns:
            The most recent IntraDrawingCheckpoint for this drawing,
            or None if no checkpoint exists.

        Example:
            >>> checkpoint = manager.load_intra_drawing_checkpoint(
            ...     "DWG-001"
            ... )
            >>> if checkpoint:
            ...     print(f"Resume from: {checkpoint.current_stage}")
        """
        # Validate input
        _validate_identifier(drawing_id, "drawing_id")

        # Find latest checkpoint
        latest_checkpoint = self._get_latest_checkpoint_path(
            self.intra_drawing_dir, drawing_id
        )

        if not latest_checkpoint:
            logger.debug(f"No intra-drawing checkpoints found for {drawing_id}")
            return None

        def _load():
            data = self._read_json_with_verification(latest_checkpoint)
            return IntraDrawingCheckpoint.from_dict(data)

        try:
            checkpoint = self._retry_operation(_load, "Load intra-drawing checkpoint")

            logger.info(
                "Loaded intra-drawing checkpoint",
                extra={
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "drawing_id": drawing_id,
                    "stage": checkpoint.current_stage.value,
                    "path": str(latest_checkpoint),
                },
            )
            return checkpoint

        except Exception as e:
            logger.error(
                f"Failed to load checkpoint from {latest_checkpoint}: {e}",
                extra={"drawing_id": drawing_id, "path": str(latest_checkpoint)},
            )
            return None

    def delete_intra_drawing_checkpoints(self, drawing_id: str) -> None:
        """Delete all checkpoints for a drawing.

        Should be called after successful completion of drawing processing
        to clean up checkpoint files and free disk space.

        Args:
            drawing_id: Unique identifier for the drawing.

        Example:
            >>> manager.delete_intra_drawing_checkpoints("DWG-001")
        """
        # Validate input
        _validate_identifier(drawing_id, "drawing_id")

        pattern = f"{drawing_id}_*.json*"
        deleted_count = 0

        for checkpoint_file in self.intra_drawing_dir.glob(pattern):
            try:
                checkpoint_file.unlink()
                deleted_count += 1
            except OSError as e:
                logger.warning(f"Failed to delete checkpoint {checkpoint_file}: {e}")

        if deleted_count > 0:
            logger.info(
                f"Deleted {deleted_count} intra-drawing checkpoint(s) "
                f"for {drawing_id}",
                extra={"drawing_id": drawing_id, "count": deleted_count},
            )

    # ============ BATCH CHECKPOINTS ============

    def save_batch_checkpoint(
        self,
        batch_id: str,
        total_drawings: int,
        completed_drawing_ids: List[str],
        failed_drawing_ids: List[str],
        pending_drawing_ids: List[str],
        current_drawing_id: Optional[str],
        total_llm_cost: float,
        average_processing_time_seconds: float,
    ) -> str:
        """Save batch processing state.

        Creates a checkpoint capturing the current state of batch processing,
        including progress, failures, and cost metrics. Typically called
        periodically (e.g., every N drawings) or before shutdown.
        Uses atomic writes with checksums and file locking.

        Args:
            batch_id: Unique identifier for this batch.
            total_drawings: Total number of drawings in batch.
            completed_drawing_ids: List of successfully processed drawing IDs.
            failed_drawing_ids: List of failed drawing IDs.
            pending_drawing_ids: List of drawings not yet processed.
            current_drawing_id: ID of drawing currently being processed
                (if any).
            total_llm_cost: Cumulative LLM API cost in USD.
            average_processing_time_seconds: Mean processing time per drawing
                in seconds.

        Returns:
            Unique checkpoint_id (UUID) for the saved checkpoint.

        Raises:
            ValueError: If batch_id is invalid or data inconsistent.
            IOError: If checkpoint file cannot be written after retries.

        Example:
            >>> manager.save_batch_checkpoint(
            ...     batch_id="batch_20251108",
            ...     total_drawings=100,
            ...     completed_drawing_ids=["DWG-001", "DWG-002"],
            ...     failed_drawing_ids=["DWG-003"],
            ...     pending_drawing_ids=["DWG-004", "DWG-005"],
            ...     current_drawing_id="DWG-004",
            ...     total_llm_cost=1.25,
            ...     average_processing_time_seconds=45.3
            ... )
            '550e8400-e29b-41d4-a716-446655440000'
        """
        # Validate inputs
        _validate_identifier(batch_id, "batch_id")

        # Generate deterministic identifiers
        timestamp_now = datetime.now(timezone.utc)
        timestamp_str = timestamp_now.strftime(TIMESTAMP_FORMAT)
        checkpoint_uuid = str(uuid.uuid4())
        checkpoint_id = f"{batch_id}_{timestamp_str}_{checkpoint_uuid}"

        checkpoint = BatchCheckpoint(
            batch_id=batch_id,
            total_drawings=total_drawings,
            completed_count=len(completed_drawing_ids),
            failed_count=len(failed_drawing_ids),
            completed_drawing_ids=completed_drawing_ids,
            failed_drawing_ids=failed_drawing_ids,
            pending_drawing_ids=pending_drawing_ids,
            current_drawing_id=current_drawing_id,
            timestamp=timestamp_now.isoformat(),
            checkpoint_id=checkpoint_id,
            total_llm_cost=total_llm_cost,
            average_processing_time_seconds=average_processing_time_seconds,
        )

        checkpoint_path = self.batch_dir / f"{checkpoint_id}.json"

        def _save():
            with self._acquire_lock(checkpoint_path):
                checksum = self._atomic_write_json(
                    checkpoint_path, checkpoint.to_dict(), compress=True
                )
                checkpoint.checksum = checksum

        try:
            self._retry_operation(_save, "Save batch checkpoint")

            logger.info(
                "Batch checkpoint saved",
                extra={
                    "checkpoint_id": checkpoint_id,
                    "batch_id": batch_id,
                    "progress": f"{checkpoint.completed_count}/" f"{total_drawings}",
                    "path": str(checkpoint_path),
                },
            )

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints(self.batch_dir, f"{batch_id}_*.json*")

        except Exception as e:
            logger.error(
                f"Failed to save batch checkpoint for {batch_id}: {e}",
                extra={"batch_id": batch_id},
            )
            raise

        return checkpoint_uuid

    def load_batch_checkpoint(self, batch_id: str) -> Optional[BatchCheckpoint]:
        """Load the most recent batch checkpoint.

        Searches for all checkpoints matching the batch_id and returns
        the most recently created one based on embedded timestamp.
        Verifies checksum for data integrity.

        Args:
            batch_id: Unique identifier for the batch.

        Returns:
            The most recent BatchCheckpoint for this batch,
            or None if no checkpoint exists.

        Example:
            >>> checkpoint = manager.load_batch_checkpoint(
            ...     "batch_20251108"
            ... )
            >>> if checkpoint:
            ...     print(f"Resume: {checkpoint.pending_drawing_ids[0]}")
        """
        # Validate input
        _validate_identifier(batch_id, "batch_id")

        # Find latest checkpoint
        latest_checkpoint = self._get_latest_checkpoint_path(self.batch_dir, batch_id)

        if not latest_checkpoint:
            logger.debug(f"No batch checkpoints found for {batch_id}")
            return None

        def _load():
            data = self._read_json_with_verification(latest_checkpoint)
            return BatchCheckpoint.from_dict(data)

        try:
            checkpoint = self._retry_operation(_load, "Load batch checkpoint")

            logger.info(
                "Loaded batch checkpoint",
                extra={
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "batch_id": batch_id,
                    "progress": f"{checkpoint.completed_count}/"
                    f"{checkpoint.total_drawings}",
                    "path": str(latest_checkpoint),
                },
            )
            return checkpoint

        except Exception as e:
            logger.error(
                f"Failed to load checkpoint from {latest_checkpoint}: {e}",
                extra={"batch_id": batch_id, "path": str(latest_checkpoint)},
            )
            return None

    def delete_batch_checkpoints(self, batch_id: str) -> None:
        """Delete all checkpoints for a batch.

        Should be called after successful completion of batch processing
        to clean up checkpoint files and free disk space.

        Args:
            batch_id: Unique identifier for the batch.

        Example:
            >>> manager.delete_batch_checkpoints("batch_20251108")
        """
        # Validate input
        _validate_identifier(batch_id, "batch_id")

        pattern = f"{batch_id}_*.json*"
        deleted_count = 0

        for checkpoint_file in self.batch_dir.glob(pattern):
            try:
                checkpoint_file.unlink()
                deleted_count += 1
            except OSError as e:
                logger.warning(f"Failed to delete checkpoint {checkpoint_file}: {e}")

        if deleted_count > 0:
            logger.info(
                f"Deleted {deleted_count} batch checkpoint(s) for {batch_id}",
                extra={"batch_id": batch_id, "count": deleted_count},
            )

    def list_available_checkpoints(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all available checkpoints with metadata.

        Scans both intra-drawing and batch checkpoint directories
        and returns metadata (ID, timestamp, stage) without loading
        full checkpoint data.

        Returns:
            Dictionary with two keys:
            - 'intra_drawing': List of dicts with checkpoint metadata
            - 'batch': List of dicts with checkpoint metadata

            Each metadata dict contains:
            - 'checkpoint_id': Full checkpoint ID
            - 'id': Drawing ID or batch ID
            - 'timestamp': ISO format timestamp string
            - 'stage': Processing stage (intra-drawing only)
            - 'file_size_bytes': Size of checkpoint file

        Example:
            >>> checkpoints = manager.list_available_checkpoints()
            >>> for cp in checkpoints['batch']:
            ...     print(f"{cp['id']}: {cp['timestamp']}")
        """

        def _extract_metadata(file_path: Path, is_batch: bool) -> Dict[str, Any]:
            """Extract metadata from checkpoint filename."""
            stem = file_path.stem
            if stem.endswith(".json"):
                stem = stem[:-5]  # Remove .json from .json.gz files

            parts = stem.split("_")

            try:
                # Extract timestamp
                timestamp_str = self._extract_timestamp_from_filename(stem)
                timestamp_dt = datetime.strptime(
                    timestamp_str, TIMESTAMP_FORMAT
                ).replace(tzinfo=timezone.utc)

                # Extract ID (first part before stage or timestamp)
                if is_batch:
                    # Batch format: {batch_id}_{timestamp}_{uuid}
                    batch_id = parts[0]
                    metadata = {
                        "checkpoint_id": stem,
                        "id": batch_id,
                        "timestamp": timestamp_dt.isoformat(),
                        "file_size_bytes": file_path.stat().st_size,
                    }
                else:
                    # Intra-drawing format: {drawing_id}_{stage}_{timestamp}_{uuid}
                    drawing_id = parts[0]
                    stage = parts[1] if len(parts) > 1 else "unknown"
                    metadata = {
                        "checkpoint_id": stem,
                        "id": drawing_id,
                        "stage": stage,
                        "timestamp": timestamp_dt.isoformat(),
                        "file_size_bytes": file_path.stat().st_size,
                    }

                return metadata

            except (ValueError, IndexError, OSError) as e:
                logger.warning(f"Cannot extract metadata from {file_path.name}: {e}")
                return {
                    "checkpoint_id": stem,
                    "id": "unknown",
                    "timestamp": "unknown",
                    "file_size_bytes": 0,
                }

        try:
            intra_checkpoints = [
                _extract_metadata(f, is_batch=False)
                for f in self.intra_drawing_dir.glob("*.json*")
            ]
            batch_checkpoints = [
                _extract_metadata(f, is_batch=True)
                for f in self.batch_dir.glob("*.json*")
            ]

            return {
                "intra_drawing": intra_checkpoints,
                "batch": batch_checkpoints,
            }
        except OSError as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return {"intra_drawing": [], "batch": []}


# Example usage demonstrating both checkpoint types:
if __name__ == "__main__":
    from contextlib import suppress

    # Use temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = CheckpointManager(
            checkpoint_dir=temp_dir,
            max_checkpoints_per_id=5,
            checkpoint_ttl_days=7,
            enable_compression=True,
        )

        # === INTRA-DRAWING CHECKPOINT EXAMPLE ===
        print("=== Intra-Drawing Checkpoint Example ===")

        # Simulate processing a drawing stage-by-stage
        drawing_id = "DWG-001"
        completed_stages = []
        intermediate_results = {}

        # After PDF processing
        intermediate_results["pdf_pages"] = 3
        checkpoint_id_1 = manager.save_intra_drawing_checkpoint(
            drawing_id=drawing_id,
            source_file="drawing001.pdf",
            current_stage=ProcessingStage.PDF_PROCESSING,
            completed_stages=completed_stages.copy(),
            intermediate_results=intermediate_results.copy(),
        )
        print(f"Saved checkpoint: {checkpoint_id_1}")

        # After OCR
        completed_stages.append(ProcessingStage.PDF_PROCESSING)
        intermediate_results["text_blocks"] = 45
        checkpoint_id_2 = manager.save_intra_drawing_checkpoint(
            drawing_id=drawing_id,
            source_file="drawing001.pdf",
            current_stage=ProcessingStage.OCR_EXTRACTION,
            completed_stages=completed_stages.copy(),
            intermediate_results=intermediate_results.copy(),
        )
        print(f"Saved checkpoint: {checkpoint_id_2}")

        # Resume from checkpoint
        checkpoint = manager.load_intra_drawing_checkpoint(drawing_id)
        if checkpoint:
            print(f"Resumed from stage: {checkpoint.current_stage.value}")
            print(
                f"Completed stages: "
                f"{[s.value for s in checkpoint.completed_stages]}"
            )
            print(f"Checksum verified: {checkpoint.checksum[:16]}...")

        # === BATCH CHECKPOINT EXAMPLE ===
        print("\n=== Batch Checkpoint Example ===")

        # Simulate batch processing
        batch_id = "batch_20251108"
        all_drawings = ["DWG-001", "DWG-002", "DWG-003", "DWG-004", "DWG-005"]

        # After processing 2 drawings
        batch_checkpoint_id = manager.save_batch_checkpoint(
            batch_id=batch_id,
            total_drawings=len(all_drawings),
            completed_drawing_ids=["DWG-001", "DWG-002"],
            failed_drawing_ids=[],
            pending_drawing_ids=["DWG-003", "DWG-004", "DWG-005"],
            current_drawing_id="DWG-003",
            total_llm_cost=0.12,
            average_processing_time_seconds=45.3,
        )
        print(f"Saved batch checkpoint: {batch_checkpoint_id}")

        # Resume batch
        batch_checkpoint = manager.load_batch_checkpoint(batch_id)
        if batch_checkpoint:
            print(
                f"Batch progress: {batch_checkpoint.completed_count}/"
                f"{batch_checkpoint.total_drawings}"
            )
            print(f"Resume from: {batch_checkpoint.pending_drawing_ids[0]}")
            print(f"LLM cost so far: ${batch_checkpoint.total_llm_cost:.2f}")
            print(f"Checksum verified: {batch_checkpoint.checksum[:16]}...")

        # === LIST CHECKPOINTS ===
        print("\n=== Available Checkpoints ===")
        all_checkpoints = manager.list_available_checkpoints()
        print(f"Intra-drawing: {len(all_checkpoints['intra_drawing'])}")
        for cp in all_checkpoints["intra_drawing"]:
            print(f"  - {cp['id']} @ {cp['stage']}: " f"{cp['file_size_bytes']} bytes")
        print(f"Batch: {len(all_checkpoints['batch'])}")
        for cp in all_checkpoints["batch"]:
            print(f"  - {cp['id']}: {cp['file_size_bytes']} bytes")

        # Cleanup happens automatically when temp_dir context exits
        print("\n=== Cleanup ===")
        print("Temporary directory will be cleaned up automatically.")
