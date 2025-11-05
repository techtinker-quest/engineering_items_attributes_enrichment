"""
Validation utilities for the Drawing Intelligence System.

Provides validation functions for various data types and inputs.
"""

import os
import re
from typing import Tuple
import numpy as np


def validate_pdf_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate PDF file exists and is readable.

    Args:
        file_path: Path to PDF file

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_path:
        return False, "File path is empty"

    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"

    if not os.path.isfile(file_path):
        return False, f"Path is not a file: {file_path}"

    if not file_path.lower().endswith(".pdf"):
        return False, f"File is not a PDF: {file_path}"

    # Check if file is readable
    try:
        with open(file_path, "rb") as f:
            # Read first few bytes to check PDF header
            header = f.read(5)
            if not header.startswith(b"%PDF-"):
                return False, f"File does not appear to be a valid PDF: {file_path}"
    except PermissionError:
        return False, f"Permission denied to read file: {file_path}"
    except Exception as e:
        return False, f"Error reading file: {e}"

    return True, ""


def validate_image_array(image: np.ndarray) -> Tuple[bool, str]:
    """
    Validate numpy array is a valid image.

    Args:
        image: Numpy array to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if image is None:
        return False, "Image is None"

    if not isinstance(image, np.ndarray):
        return False, f"Image is not a numpy array: {type(image)}"

    if image.size == 0:
        return False, "Image is empty (size 0)"

    if len(image.shape) not in [2, 3]:
        return False, f"Invalid image shape: {image.shape} (expected 2D or 3D)"

    # Check for 3-channel images
    if len(image.shape) == 3:
        if image.shape[2] not in [1, 3, 4]:
            return (
                False,
                f"Invalid number of channels: {image.shape[2]} (expected 1, 3, or 4)",
            )

    # Check data type
    if image.dtype not in [np.uint8, np.float32, np.float64]:
        return False, f"Unsupported image dtype: {image.dtype}"

    # Check for reasonable dimensions
    height, width = image.shape[:2]
    if height < 10 or width < 10:
        return False, f"Image too small: {width}x{height} (minimum 10x10)"

    if height > 50000 or width > 50000:
        return False, f"Image too large: {width}x{height} (maximum 50000x50000)"

    return True, ""


def validate_bbox(bbox: "BoundingBox", image_width: int, image_height: int) -> bool:
    """
    Validate bounding box is within image bounds and has positive dimensions.

    Args:
        bbox: BoundingBox object
        image_width: Image width
        image_height: Image height

    Returns:
        True if bbox is valid
    """
    if bbox.width <= 0 or bbox.height <= 0:
        return False

    if bbox.x < 0 or bbox.y < 0:
        return False

    if bbox.x + bbox.width > image_width:
        return False

    if bbox.y + bbox.height > image_height:
        return False

    return True


def validate_confidence_score(score: float) -> bool:
    """
    Validate confidence score is in valid range [0.0, 1.0].

    Args:
        score: Confidence score

    Returns:
        True if score is valid
    """
    if not isinstance(score, (int, float)):
        return False

    return 0.0 <= score <= 1.0


def validate_entity_type(entity_type: str) -> bool:
    """
    Validate entity type is recognized.

    Args:
        entity_type: Entity type string

    Returns:
        True if entity type is valid
    """
    # Import here to avoid circular dependency
    from ..models.data_structures import EntityType

    valid_types = [e.value for e in EntityType]
    return entity_type in valid_types


def validate_api_key(provider: str, api_key: str) -> Tuple[bool, str]:
    """
    Validate API key format for provider.

    Args:
        provider: Provider name ('openai', 'anthropic', 'google')
        api_key: API key string

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not api_key:
        return False, "API key is empty"

    if not isinstance(api_key, str):
        return False, "API key must be a string"

    # Provider-specific validation
    provider_lower = provider.lower()

    if provider_lower == "openai":
        # OpenAI keys start with 'sk-'
        if not api_key.startswith("sk-"):
            return False, "OpenAI API keys should start with 'sk-'"
        if len(api_key) < 20:
            return False, "OpenAI API key too short"

    elif provider_lower == "anthropic":
        # Anthropic keys start with 'sk-ant-'
        if not api_key.startswith("sk-ant-"):
            return False, "Anthropic API keys should start with 'sk-ant-'"
        if len(api_key) < 20:
            return False, "Anthropic API key too short"

    elif provider_lower == "google":
        # Google API keys are alphanumeric
        if len(api_key) < 20:
            return False, "Google API key too short"

    else:
        return False, f"Unknown provider: {provider}"

    return True, ""


def validate_drawing_id(drawing_id: str) -> bool:
    """
    Validate drawing ID format.

    Expected format: PREFIX-YYYYMMDD-HHMMSS-HASH
    Example: DWG-20251102-143022-a1b2c3d4

    Args:
        drawing_id: Drawing ID string

    Returns:
        True if ID format is valid
    """
    if not drawing_id:
        return False

    # Pattern: PREFIX-YYYYMMDD-HHMMSS-HASH
    pattern = r"^[A-Z]+-\d{8}-\d{6}-[a-f0-9]{8}$"
    return bool(re.match(pattern, drawing_id))


def validate_file_size(file_path: str, max_size_mb: float) -> Tuple[bool, str]:
    """
    Validate file size is within limit.

    Args:
        file_path: Path to file
        max_size_mb: Maximum size in megabytes

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"

    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)

    if size_mb > max_size_mb:
        return False, f"File size {size_mb:.2f}MB exceeds limit of {max_size_mb}MB"

    return True, ""


def validate_directory_writable(directory: str) -> Tuple[bool, str]:
    """
    Validate directory exists and is writable.

    Args:
        directory: Directory path

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(directory):
        return False, f"Directory does not exist: {directory}"

    if not os.path.isdir(directory):
        return False, f"Path is not a directory: {directory}"

    if not os.access(directory, os.W_OK):
        return False, f"Directory is not writable: {directory}"

    return True, ""


def validate_model_path(model_path: str) -> Tuple[bool, str]:
    """
    Validate model file exists and has correct extension.

    Args:
        model_path: Path to model file

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not model_path:
        return False, "Model path is empty"

    if not os.path.exists(model_path):
        return False, f"Model file does not exist: {model_path}"

    valid_extensions = [".pt", ".pth", ".onnx", ".pb", ".h5"]
    if not any(model_path.lower().endswith(ext) for ext in valid_extensions):
        return (
            False,
            f"Invalid model file extension. Expected one of: {valid_extensions}",
        )

    return True, ""


def validate_coordinate(x: int, y: int, max_x: int, max_y: int) -> bool:
    """
    Validate coordinate is within bounds.

    Args:
        x: X coordinate
        y: Y coordinate
        max_x: Maximum X value (exclusive)
        max_y: Maximum Y value (exclusive)

    Returns:
        True if coordinate is valid
    """
    return 0 <= x < max_x and 0 <= y < max_y


def validate_percentage(value: float) -> bool:
    """
    Validate value is a valid percentage (0-100).

    Args:
        value: Percentage value

    Returns:
        True if value is valid
    """
    return isinstance(value, (int, float)) and 0.0 <= value <= 100.0


def validate_email(email: str) -> bool:
    """
    Validate email address format.

    Args:
        email: Email address

    Returns:
        True if email format is valid
    """
    if not email:
        return False

    # Basic email pattern
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def validate_url(url: str) -> bool:
    """
    Validate URL format.

    Args:
        url: URL string

    Returns:
        True if URL format is valid
    """
    if not url:
        return False

    # Basic URL pattern
    pattern = r"^https?://[^\s/$.?#].[^\s]*$"
    return bool(re.match(pattern, url))


def validate_color(color: Tuple[int, int, int]) -> bool:
    """
    Validate BGR color tuple.

    Args:
        color: Tuple of (B, G, R) values

    Returns:
        True if color is valid
    """
    if not isinstance(color, tuple) or len(color) != 3:
        return False

    return all(isinstance(c, int) and 0 <= c <= 255 for c in color)


def validate_json_serializable(obj: any) -> Tuple[bool, str]:
    """
    Validate object is JSON serializable.

    Args:
        obj: Object to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    import json

    try:
        json.dumps(obj)
        return True, ""
    except (TypeError, ValueError) as e:
        return False, f"Object is not JSON serializable: {e}"
