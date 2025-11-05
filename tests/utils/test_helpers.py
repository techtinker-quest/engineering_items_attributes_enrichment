"""
Test Helper Functions

Utilities for testing drawing intelligence components.
"""

import tempfile
import shutil
from pathlib import Path
import numpy as np
from typing import Optional

from src.drawing_intelligence.database.database_manager import DatabaseManager
from src.drawing_intelligence.utils.config_loader import Config
from src.drawing_intelligence.models.data_structures import BoundingBox


def setup_test_database() -> DatabaseManager:
    """
    Create temporary test database.

    Returns:
        DatabaseManager instance with temp database
    """
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    db = DatabaseManager(str(db_path))
    return db


def cleanup_test_database(db: DatabaseManager) -> None:
    """
    Clean up test database.

    Args:
        db: DatabaseManager instance to cleanup
    """
    if db:
        db_path = Path(db.db_path)
        db.close()

        if db_path.exists():
            db_path.unlink()

        # Remove parent directory if empty
        if db_path.parent.exists() and not list(db_path.parent.iterdir()):
            shutil.rmtree(db_path.parent)


def create_test_config() -> Config:
    """
    Create test configuration.

    Returns:
        Config object with test settings
    """
    config = Config()

    # Test paths
    config.paths = type(
        "obj",
        (object,),
        {
            "data_dir": "tests/test_data",
            "models_dir": "tests/test_models",
            "output_dir": "tests/test_output",
            "temp_dir": "tests/test_temp",
            "log_dir": "tests/test_logs",
        },
    )()

    # Test database
    config.database = type("obj", (object,), {"path": "tests/test_data/test.db"})()

    # Processing configs with test settings
    config.pdf_processing = type(
        "obj", (object,), {"dpi": 150, "max_file_size_mb": 10, "max_pages": 5}
    )()

    return config


def assert_bbox_equal(
    bbox1: BoundingBox, bbox2: BoundingBox, tolerance: int = 2
) -> None:
    """
    Assert two bounding boxes are equal within tolerance.

    Args:
        bbox1: First bounding box
        bbox2: Second bounding box
        tolerance: Pixel tolerance for comparison
    """
    assert (
        abs(bbox1.x - bbox2.x) <= tolerance
    ), f"X coordinates differ: {bbox1.x} vs {bbox2.x}"
    assert (
        abs(bbox1.y - bbox2.y) <= tolerance
    ), f"Y coordinates differ: {bbox1.y} vs {bbox2.y}"
    assert (
        abs(bbox1.width - bbox2.width) <= tolerance
    ), f"Widths differ: {bbox1.width} vs {bbox2.width}"
    assert (
        abs(bbox1.height - bbox2.height) <= tolerance
    ), f"Heights differ: {bbox1.height} vs {bbox2.height}"


def assert_confidence_in_range(
    confidence: float, min_val: float = 0.0, max_val: float = 1.0
) -> None:
    """
    Assert confidence score is in valid range.

    Args:
        confidence: Confidence score to check
        min_val: Minimum valid value
        max_val: Maximum valid value
    """
    assert (
        min_val <= confidence <= max_val
    ), f"Confidence {confidence} not in range [{min_val}, {max_val}]"


def load_test_image(filename: str) -> np.ndarray:
    """
    Load test image from fixtures directory.

    Args:
        filename: Image filename

    Returns:
        Numpy array image
    """
    test_dir = Path(__file__).parent.parent / "fixtures" / "sample_data"
    image_path = test_dir / filename

    if not image_path.exists():
        # Create mock image if not exists
        from ..fixtures.test_data_generator import TestDataGenerator

        return TestDataGenerator.create_mock_image()

    import cv2

    return cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)


def load_test_pdf(filename: str) -> str:
    """
    Load test PDF from fixtures directory.

    Args:
        filename: PDF filename

    Returns:
        Path to PDF file
    """
    test_dir = Path(__file__).parent.parent / "fixtures" / "sample_data"
    pdf_path = test_dir / filename

    if not pdf_path.exists():
        # Create mock PDF if not exists
        from ..fixtures.test_data_generator import TestDataGenerator

        try:
            return TestDataGenerator.create_mock_pdf(str(pdf_path))
        except ImportError:
            raise FileNotFoundError(f"Test PDF not found and cannot create: {filename}")

    return str(pdf_path)


def compare_images_similar(
    img1: np.ndarray, img2: np.ndarray, threshold: float = 0.95
) -> bool:
    """
    Compare if two images are similar.

    Args:
        img1: First image
        img2: Second image
        threshold: Similarity threshold (0-1)

    Returns:
        True if images are similar
    """
    if img1.shape != img2.shape:
        return False

    # Calculate normalized cross-correlation
    correlation = np.corrcoef(img1.flatten(), img2.flatten())[0, 1]

    return correlation >= threshold


def create_temp_directory() -> str:
    """
    Create temporary directory for testing.

    Returns:
        Path to temporary directory
    """
    return tempfile.mkdtemp()


def cleanup_temp_directory(temp_dir: str) -> None:
    """
    Remove temporary directory.

    Args:
        temp_dir: Path to temporary directory
    """
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


def assert_processing_result_valid(result) -> None:
    """
    Assert processing result has valid structure.

    Args:
        result: ProcessingResult to validate
    """
    assert result.drawing_id is not None
    assert result.source_file is not None
    assert result.processing_timestamp is not None
    assert result.status in ["complete", "failed", "pending"]
    assert 0.0 <= result.overall_confidence <= 1.0
    assert isinstance(result.entities, list)
    assert isinstance(result.detections, list)


def mock_ocr_response(confidence: float = 0.9) -> dict:
    """
    Create mock OCR response.

    Args:
        confidence: OCR confidence score

    Returns:
        Mock OCR response dictionary
    """
    return {
        "text": "PART NO: ABC-12345",
        "confidence": confidence,
        "bbox": [100, 100, 250, 130],
    }


def mock_detection_response(class_name: str = "bolt", confidence: float = 0.85) -> dict:
    """
    Create mock detection response.

    Args:
        class_name: Detected class
        confidence: Detection confidence

    Returns:
        Mock detection response dictionary
    """
    return {"class": class_name, "confidence": confidence, "bbox": [200, 200, 300, 300]}


def skip_if_no_gpu():
    """Skip test if GPU not available."""
    import pytest

    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
    except ImportError:
        pytest.skip("PyTorch not installed")


def skip_if_no_llm():
    """Skip test if LLM not configured."""
    import pytest
    import os

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        pytest.skip("No LLM API keys configured")


class MockLLMGateway:
    """Mock LLM Gateway for testing."""

    def __init__(self):
        self.call_count = 0
        self.last_call = None

    def verify_ocr(self, image_crop, ocr_text, drawing_id, region_type):
        """Mock OCR verification."""
        self.call_count += 1
        self.last_call = "verify_ocr"

        from src.drawing_intelligence.llm.llm_gateway import OCRVerification

        return OCRVerification(
            corrected_text=ocr_text, corrections_made=[], confidence=0.95
        )

    def extract_entities_llm(self, text, context, entity_types, drawing_id):
        """Mock entity extraction."""
        self.call_count += 1
        self.last_call = "extract_entities"
        return []


class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        import time

        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        import time

        self.elapsed = time.time() - self.start_time
        print(f"{self.name} took {self.elapsed:.3f}s")
