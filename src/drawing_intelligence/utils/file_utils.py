"""
File utilities for the Drawing Intelligence System.

Provides functions for file operations, validation, and path management.
"""

import os
import hashlib
import shutil
import uuid
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime


def ensure_directory(path: str) -> None:
    """
    Create directory if it doesn't exist, including parent directories.

    Args:
        path: Directory path to create

    Raises:
        OSError: If directory creation fails
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create directory {path}: {e}")


def get_file_hash(file_path: str, algorithm: str = "md5") -> str:
    """
    Calculate hash of file contents.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha256')

    Returns:
        Hexadecimal hash string

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If algorithm not supported
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if algorithm not in ["md5", "sha256", "sha1"]:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    hash_obj = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        # Read file in chunks for memory efficiency
        for chunk in iter(lambda: f.read(8192), b""):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        Size in MB (rounded to 2 decimal places)

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return round(size_mb, 2)


def list_files(
    directory: str, extension: Optional[str] = None, recursive: bool = False
) -> List[str]:
    """
    List files in directory with optional filtering.

    Args:
        directory: Directory path to scan
        extension: Optional file extension filter (e.g., '.pdf', '.png')
        recursive: Whether to scan subdirectories

    Returns:
        List of absolute file paths

    Raises:
        NotADirectoryError: If path is not a directory
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Not a directory: {directory}")

    files = []

    if recursive:
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if extension is None or filename.lower().endswith(extension.lower()):
                    files.append(os.path.abspath(os.path.join(root, filename)))
    else:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                if extension is None or item.lower().endswith(extension.lower()):
                    files.append(os.path.abspath(item_path))

    return sorted(files)


def safe_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename for safe filesystem use.

    Removes/replaces unsafe characters and limits length.

    Args:
        filename: Original filename
        max_length: Maximum filename length (default: 255)

    Returns:
        Safe filename string
    """
    # Remove path separators
    filename = os.path.basename(filename)

    # Replace unsafe characters
    unsafe_chars = ["<", ">", ":", '"', "/", "\\", "|", "?", "*"]
    for char in unsafe_chars:
        filename = filename.replace(char, "_")

    # Remove leading/trailing spaces and dots
    filename = filename.strip(". ")

    # Ensure not empty
    if not filename:
        filename = "unnamed"

    # Truncate if too long (preserve extension)
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        max_name_length = max_length - len(ext)
        filename = name[:max_name_length] + ext

    return filename


def generate_unique_id(prefix: str = "") -> str:
    """
    Generate unique ID with optional prefix.

    Format: PREFIX-YYYYMMDD-HHMMSS-UUID
    Example: DWG-20251102-143022-a1b2c3d4

    Args:
        prefix: Optional prefix (e.g., 'DWG', 'BATCH')

    Returns:
        Unique ID string
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_suffix = uuid.uuid4().hex[:8]

    if prefix:
        return f"{prefix}-{timestamp}-{unique_suffix}"
    else:
        return f"{timestamp}-{unique_suffix}"


def copy_file_with_backup(source: str, destination: str) -> None:
    """
    Copy file, backing up destination if it exists.

    If destination exists, it's renamed with .bak extension before copying.

    Args:
        source: Source file path
        destination: Destination file path

    Raises:
        FileNotFoundError: If source doesn't exist
        OSError: If copy fails
    """
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source file not found: {source}")

    # Create destination directory if needed
    dest_dir = os.path.dirname(destination)
    if dest_dir:
        ensure_directory(dest_dir)

    # Backup existing destination
    if os.path.exists(destination):
        backup_path = f"{destination}.bak"
        # If backup exists, add timestamp
        if os.path.exists(backup_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{destination}.{timestamp}.bak"
        shutil.copy2(destination, backup_path)

    # Copy file
    shutil.copy2(source, destination)


def get_relative_path(path: str, base_dir: str) -> str:
    """
    Get path relative to base directory.

    Args:
        path: Absolute or relative path
        base_dir: Base directory

    Returns:
        Relative path string
    """
    return os.path.relpath(path, base_dir)


def split_path_components(path: str) -> Tuple[str, str, str]:
    """
    Split path into directory, filename, and extension.

    Args:
        path: File path

    Returns:
        Tuple of (directory, filename_without_ext, extension)
    """
    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    name, ext = os.path.splitext(filename)
    return directory, name, ext


def ensure_file_extension(filename: str, extension: str) -> str:
    """
    Ensure filename has the correct extension.

    Args:
        filename: Original filename
        extension: Desired extension (with or without leading dot)

    Returns:
        Filename with correct extension
    """
    if not extension.startswith("."):
        extension = f".{extension}"

    current_ext = os.path.splitext(filename)[1].lower()
    if current_ext != extension.lower():
        return filename + extension
    return filename


def is_path_safe(path: str, base_dir: str) -> bool:
    """
    Check if path is safe (doesn't escape base directory).

    Prevents directory traversal attacks.

    Args:
        path: Path to check
        base_dir: Base directory that should contain the path

    Returns:
        True if path is safe, False otherwise
    """
    try:
        abs_path = os.path.abspath(path)
        abs_base = os.path.abspath(base_dir)
        return abs_path.startswith(abs_base)
    except (ValueError, OSError):
        return False


def atomic_write(file_path: str, content: str, encoding: str = "utf-8") -> None:
    """
    Write file atomically using temporary file and rename.

    Prevents partial writes in case of interruption.

    Args:
        file_path: Destination file path
        content: Content to write
        encoding: Text encoding (default: utf-8)

    Raises:
        OSError: If write fails
    """
    directory = os.path.dirname(file_path)
    if directory:
        ensure_directory(directory)

    # Write to temporary file
    temp_path = f"{file_path}.tmp.{uuid.uuid4().hex[:8]}"

    try:
        with open(temp_path, "w", encoding=encoding) as f:
            f.write(content)

        # Atomic rename
        os.replace(temp_path, file_path)
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise OSError(f"Atomic write failed for {file_path}: {e}")


def get_available_disk_space(path: str) -> float:
    """
    Get available disk space in GB for the filesystem containing path.

    Args:
        path: Any path on the filesystem to check

    Returns:
        Available space in GB

    Raises:
        OSError: If path doesn't exist or space check fails
    """
    if not os.path.exists(path):
        raise OSError(f"Path does not exist: {path}")

    stat = os.statvfs(path) if hasattr(os, "statvfs") else None

    if stat:
        # Unix/Linux
        available_bytes = stat.f_bavail * stat.f_frsize
    else:
        # Windows fallback
        import shutil

        available_bytes = shutil.disk_usage(path).free

    available_gb = available_bytes / (1024**3)
    return round(available_gb, 2)
