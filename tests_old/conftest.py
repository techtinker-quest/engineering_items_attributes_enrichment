"""
Pytest configuration and fixtures.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np

from src.drawing_intelligence.utils.config_loader import Config
from src.drawing_intelligence.database.database_manager import DatabaseManager
from src.drawing_intelligence.models.data_structures import *


@pytest.fixture(scope="session")
def test_config():
    """Create test configuration."""
    config = Config()

    # Override with test settings
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

    config.database = type(
        "obj", (object,), {"path": "tests/test_data/test_drawings.db"}
    )()

    config.pdf_processing = type(
        "obj",
        (object,),
        {
            "dpi": 150,  # Lower for tests
            "max_file_size_mb": 10,
            "max_pages": 5,
            "convert_to_grayscale": True,
        },
    )()

    config.ocr = type(
        "obj",
        (object,),
        {
            "primary_engine": "paddleocr",
            "fallback_engine": "easyocr",
            "confidence_threshold": 0.85,
            "languages": ["en"],
        },
    )()

    config.shape_detection = type(
        "obj",
        (object,),
        {
            "model_path": "tests/test_models/dummy_yolo.pt",
            "confidence_threshold": 0.45,
            "nms_threshold": 0.45,
            "device": "cpu",  # CPU for tests
            "batch_size": 4,
        },
    )()

    config.llm_integration = type(
        "obj",
        (object,),
        {
            "enabled": False,  # Disabled for most tests
            "cost_controls": type(
                "obj",
                (object,),
                {"daily_budget_usd": 1.0, "per_drawing_limit_usd": 0.10},
            )(),
        },
    )()

    return config


@pytest.fixture(scope="function")
def temp_dir():
    """Create temporary directory for test."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="function")
def test_db(temp_dir):
    """Create test database."""
    db_path = Path(temp_dir) / "test.db"
    db = DatabaseManager(str(db_path))
    yield db
    db.close()


@pytest.fixture(scope="function")
def sample_image():
    """Create sample test image."""
    # Create a simple test image (800x600, grayscale)
    image = np.ones((600, 800), dtype=np.uint8) * 255

    # Add some features
    # Rectangle
    image[100:200, 100:300] = 0
    # Circle (approximation)
    center_y, center_x = 400, 600
    y, x = np.ogrid[:600, :800]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= 50**2
    image[mask] = 0

    return image


@pytest.fixture(scope="function")
def sample_bbox():
    """Create sample bounding box."""
    return BoundingBox(x=100, y=100, width=200, height=150)


@pytest.fixture(scope="function")
def sample_text_block():
    """Create sample text block."""
    return TextBlock(
        text_id="TXT-001",
        content="Ø25.4 ± 0.1 mm",
        bbox=BoundingBox(x=100, y=100, width=150, height=30),
        confidence=0.92,
        ocr_engine="paddleocr",
        region_type="dimension",
    )


@pytest.fixture(scope="function")
def sample_entity():
    """Create sample entity."""
    return Entity(
        entity_id="ENT-001",
        entity_type=EntityType.DIMENSION,
        value="25.4",
        original_text="Ø25.4 ± 0.1 mm",
        normalized_value={"value": 25.4, "unit": "mm", "tolerance": 0.1},
        confidence=0.90,
        extraction_method="regex",
        source_text_id="TXT-001",
        bbox=BoundingBox(x=100, y=100, width=150, height=30),
    )


@pytest.fixture(scope="function")
def sample_detection():
    """Create sample detection."""
    return Detection(
        detection_id="DET-001",
        class_name="bolt",
        confidence=0.88,
        bbox=BoundingBox(x=200, y=200, width=100, height=100),
        bbox_normalized=NormalizedBBox(
            x_center=0.3125, y_center=0.4167, width=0.125, height=0.1667
        ),
    )


@pytest.fixture(scope="function")
def sample_association():
    """Create sample association."""
    return Association(
        association_id="ASSOC-001",
        text_id="TXT-001",
        shape_id="DET-001",
        relationship_type="dimension",
        confidence=0.85,
        distance_pixels=120.5,
    )


@pytest.fixture(scope="function")
def sample_processing_result():
    """Create sample processing result."""
    from datetime import datetime

    return ProcessingResult(
        drawing_id="DWG-TEST-001",
        source_file="test_drawing.pdf",
        processing_timestamp=datetime.now(),
        pipeline_type=PipelineType.BASELINE_ONLY,
        pipeline_version="1.0.0",
        pdf_pages=[],
        ocr_result=None,
        entities=[],
        title_block=None,
        detections=[],
        associations=[],
        hierarchy=None,
        validation_report=None,
        overall_confidence=0.85,
        confidence_scores=None,
        review_flags=[],
        completeness_score=None,
        llm_usage=[],
        processing_times={"total": 1.5},
        status="complete",
    )


@pytest.fixture(scope="session")
def mock_llm_gateway():
    """Create mock LLM gateway for testing."""

    class MockLLMGateway:
        def __init__(self):
            self.call_count = 0

        def verify_ocr(self, image_crop, ocr_text, drawing_id, region_type):
            self.call_count += 1
            from src.drawing_intelligence.llm.llm_gateway import OCRVerification

            return OCRVerification(
                corrected_text=ocr_text, corrections_made=[], confidence=0.95
            )

        def extract_entities_llm(self, text, context, entity_types, drawing_id):
            self.call_count += 1
            return []

    return MockLLMGateway()


@pytest.fixture(autouse=True)
def reset_test_environment(temp_dir):
    """Reset test environment before each test."""
    # Create test directories
    Path(temp_dir).mkdir(exist_ok=True)

    yield

    # Cleanup after test
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


# Markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line(
        "markers", "integration: Integration tests for multiple components"
    )
    config.addinivalue_line("markers", "slow: Tests that take significant time")
    config.addinivalue_line("markers", "llm: Tests requiring LLM API access")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
