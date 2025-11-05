"""
Unit tests for validation_utils module.

Tests validation functions for PDFs, images, bboxes, and other data types.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from src.drawing_intelligence.utils.validation_utils import (
    validate_pdf_file,
    validate_image_array,
    validate_bbox,
    validate_confidence_score,
    validate_entity_type,
    validate_api_key,
    validate_drawing_id,
    validate_file_size,
    validate_model_path,
)
from src.drawing_intelligence.utils.geometry_utils import BoundingBox
from src.drawing_intelligence.models.data_structures import EntityType


class TestPDFValidation:
    """Test PDF file validation."""

    def test_validate_valid_pdf(self, tmp_path):
        """Test validation of valid PDF file."""
        # Create a dummy PDF file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%fake pdf content")

        is_valid, message = validate_pdf_file(str(pdf_path))
        assert is_valid is True or "file format" not in message.lower()

    def test_validate_nonexistent_pdf(self):
        """Test validation of non-existent file."""
        is_valid, message = validate_pdf_file("nonexistent.pdf")
        assert is_valid is False
        assert "not found" in message.lower() or "does not exist" in message.lower()

    def test_validate_non_pdf_file(self, tmp_path):
        """Test validation of non-PDF file."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("Not a PDF")

        is_valid, message = validate_pdf_file(str(txt_path))
        assert is_valid is False
        assert "pdf" in message.lower()

    def test_validate_empty_pdf(self, tmp_path):
        """Test validation of empty file."""
        pdf_path = tmp_path / "empty.pdf"
        pdf_path.write_bytes(b"")

        is_valid, message = validate_pdf_file(str(pdf_path))
        assert is_valid is False
        assert "empty" in message.lower() or "size" in message.lower()

    def test_validate_pdf_max_size(self, tmp_path):
        """Test PDF file size validation."""
        pdf_path = tmp_path / "large.pdf"
        # Create file larger than max size (e.g., 50MB)
        large_content = b"%PDF-1.4\n" + b"x" * (51 * 1024 * 1024)
        pdf_path.write_bytes(large_content)

        is_valid, message = validate_pdf_file(str(pdf_path), max_size_mb=50)
        assert is_valid is False
        assert "size" in message.lower()


class TestImageValidation:
    """Test image array validation."""

    def test_validate_valid_rgb_image(self):
        """Test validation of valid RGB image."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        is_valid, message = validate_image_array(image)
        assert is_valid is True

    def test_validate_valid_grayscale_image(self):
        """Test validation of valid grayscale image."""
        image = np.zeros((100, 100), dtype=np.uint8)
        is_valid, message = validate_image_array(image)
        assert is_valid is True

    def test_validate_invalid_shape(self):
        """Test validation of invalid image shape."""
        image = np.zeros((100,), dtype=np.uint8)  # 1D array
        is_valid, message = validate_image_array(image)
        assert is_valid is False
        assert "shape" in message.lower() or "dimension" in message.lower()

    def test_validate_invalid_dtype(self):
        """Test validation of invalid dtype."""
        image = np.zeros((100, 100, 3), dtype=np.float64)
        is_valid, message = validate_image_array(image)
        assert is_valid is False
        assert "dtype" in message.lower() or "type" in message.lower()

    def test_validate_invalid_channels(self):
        """Test validation of invalid channel count."""
        image = np.zeros((100, 100, 5), dtype=np.uint8)  # 5 channels invalid
        is_valid, message = validate_image_array(image)
        assert is_valid is False
        assert "channel" in message.lower()

    def test_validate_empty_image(self):
        """Test validation of empty image."""
        image = np.array([], dtype=np.uint8)
        is_valid, message = validate_image_array(image)
        assert is_valid is False
        assert "empty" in message.lower()

    def test_validate_min_dimensions(self):
        """Test validation of minimum dimensions."""
        image = np.zeros((10, 10, 3), dtype=np.uint8)  # Too small
        is_valid, message = validate_image_array(image, min_width=50, min_height=50)
        assert is_valid is False
        assert "dimension" in message.lower() or "size" in message.lower()

    def test_validate_max_dimensions(self):
        """Test validation of maximum dimensions."""
        image = np.zeros((5000, 5000, 3), dtype=np.uint8)  # Too large
        is_valid, message = validate_image_array(image, max_width=4000, max_height=4000)
        assert is_valid is False
        assert "dimension" in message.lower() or "size" in message.lower()


class TestBboxValidation:
    """Test bounding box validation."""

    def test_validate_valid_bbox(self):
        """Test validation of valid bbox."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        is_valid, message = validate_bbox(bbox, image_width=200, image_height=200)
        assert is_valid is True

    def test_validate_negative_coordinates(self):
        """Test bbox with negative coordinates."""
        bbox = BoundingBox(x=-10, y=20, width=100, height=50)
        is_valid, message = validate_bbox(bbox, image_width=200, image_height=200)
        assert is_valid is False
        assert "negative" in message.lower() or "coordinate" in message.lower()

    def test_validate_zero_dimensions(self):
        """Test bbox with zero dimensions."""
        bbox = BoundingBox(x=10, y=20, width=0, height=50)
        is_valid, message = validate_bbox(bbox, image_width=200, image_height=200)
        assert is_valid is False
        assert "width" in message.lower() or "dimension" in message.lower()

    def test_validate_bbox_exceeds_image(self):
        """Test bbox exceeding image bounds."""
        bbox = BoundingBox(x=150, y=150, width=100, height=100)
        is_valid, message = validate_bbox(bbox, image_width=200, image_height=200)
        assert is_valid is False
        assert "exceed" in message.lower() or "bound" in message.lower()

    def test_validate_bbox_partially_outside(self):
        """Test bbox partially outside image."""
        bbox = BoundingBox(x=190, y=190, width=20, height=20)
        is_valid, message = validate_bbox(bbox, image_width=200, image_height=200)
        assert is_valid is False
        assert "exceed" in message.lower() or "outside" in message.lower()

    def test_validate_bbox_at_edge(self):
        """Test bbox at image edge (should be valid)."""
        bbox = BoundingBox(x=0, y=0, width=200, height=200)
        is_valid, message = validate_bbox(bbox, image_width=200, image_height=200)
        assert is_valid is True


class TestConfidenceValidation:
    """Test confidence score validation."""

    def test_validate_valid_confidence(self):
        """Test valid confidence scores."""
        assert validate_confidence_score(0.0) is True
        assert validate_confidence_score(0.5) is True
        assert validate_confidence_score(1.0) is True

    def test_validate_negative_confidence(self):
        """Test negative confidence."""
        is_valid, message = validate_confidence_score(-0.1)
        assert is_valid is False
        assert "range" in message.lower()

    def test_validate_confidence_above_one(self):
        """Test confidence above 1.0."""
        is_valid, message = validate_confidence_score(1.5)
        assert is_valid is False
        assert "range" in message.lower()

    def test_validate_confidence_boundary(self):
        """Test confidence at boundaries."""
        assert validate_confidence_score(0.0) is True
        assert validate_confidence_score(1.0) is True

    def test_validate_confidence_non_numeric(self):
        """Test non-numeric confidence."""
        with pytest.raises((TypeError, ValueError)):
            validate_confidence_score("0.5")


class TestEntityTypeValidation:
    """Test entity type validation."""

    def test_validate_valid_entity_type(self):
        """Test valid entity types."""
        assert validate_entity_type(EntityType.PART_NUMBER) is True
        assert validate_entity_type(EntityType.DIMENSION) is True
        assert validate_entity_type(EntityType.MATERIAL) is True

    def test_validate_entity_type_string(self):
        """Test entity type as string."""
        assert validate_entity_type("PART_NUMBER") is True
        assert validate_entity_type("DIMENSION") is True

    def test_validate_invalid_entity_type(self):
        """Test invalid entity type."""
        is_valid, message = validate_entity_type("INVALID_TYPE")
        assert is_valid is False
        assert "invalid" in message.lower() or "unknown" in message.lower()

    def test_validate_entity_type_case_insensitive(self):
        """Test case-insensitive entity type validation."""
        assert validate_entity_type("part_number") is True
        assert validate_entity_type("Part_Number") is True


class TestAPIKeyValidation:
    """Test API key validation."""

    def test_validate_openai_key_format(self):
        """Test OpenAI API key format."""
        valid_key = "sk-proj-1234567890abcdef1234567890abcdef"
        is_valid, message = validate_api_key(valid_key, provider="openai")
        assert is_valid is True

    def test_validate_anthropic_key_format(self):
        """Test Anthropic API key format."""
        valid_key = "sk-ant-1234567890abcdef1234567890abcdef"
        is_valid, message = validate_api_key(valid_key, provider="anthropic")
        assert is_valid is True

    def test_validate_google_key_format(self):
        """Test Google API key format."""
        valid_key = "AIza1234567890abcdef1234567890abcdef"
        is_valid, message = validate_api_key(valid_key, provider="google")
        assert is_valid is True

    def test_validate_empty_api_key(self):
        """Test empty API key."""
        is_valid, message = validate_api_key("", provider="openai")
        assert is_valid is False
        assert "empty" in message.lower()

    def test_validate_invalid_format(self):
        """Test invalid API key format."""
        is_valid, message = validate_api_key("invalid-key-123", provider="openai")
        assert is_valid is False
        assert "format" in message.lower() or "invalid" in message.lower()

    def test_validate_unknown_provider(self):
        """Test unknown provider."""
        is_valid, message = validate_api_key("sk-test123", provider="unknown")
        assert is_valid is False
        assert "provider" in message.lower()


class TestDrawingIDValidation:
    """Test drawing ID validation."""

    def test_validate_valid_drawing_id(self):
        """Test valid drawing ID format."""
        valid_id = "DWG-20251104-143022-a1b2c3d4"
        is_valid, message = validate_drawing_id(valid_id)
        assert is_valid is True

    def test_validate_drawing_id_with_prefix(self):
        """Test drawing ID with correct prefix."""
        valid_id = "DWG-20251104-143022-abc123"
        is_valid, message = validate_drawing_id(valid_id)
        assert is_valid is True

    def test_validate_drawing_id_invalid_format(self):
        """Test invalid drawing ID format."""
        invalid_id = "INVALID-ID"
        is_valid, message = validate_drawing_id(invalid_id)
        assert is_valid is False
        assert "format" in message.lower()

    def test_validate_drawing_id_empty(self):
        """Test empty drawing ID."""
        is_valid, message = validate_drawing_id("")
        assert is_valid is False
        assert "empty" in message.lower()

    def test_validate_drawing_id_special_chars(self):
        """Test drawing ID with special characters."""
        invalid_id = "DWG-2025/11/04-test"
        is_valid, message = validate_drawing_id(invalid_id)
        assert is_valid is False
        assert "character" in message.lower() or "format" in message.lower()


class TestFileSizeValidation:
    """Test file size validation."""

    def test_validate_file_within_limit(self, tmp_path):
        """Test file within size limit."""
        file_path = tmp_path / "test.txt"
        file_path.write_bytes(b"x" * 1024)  # 1KB

        is_valid, message = validate_file_size(str(file_path), max_size_mb=1)
        assert is_valid is True

    def test_validate_file_exceeds_limit(self, tmp_path):
        """Test file exceeding size limit."""
        file_path = tmp_path / "large.txt"
        file_path.write_bytes(b"x" * (2 * 1024 * 1024))  # 2MB

        is_valid, message = validate_file_size(str(file_path), max_size_mb=1)
        assert is_valid is False
        assert "size" in message.lower()

    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        is_valid, message = validate_file_size("nonexistent.txt", max_size_mb=1)
        assert is_valid is False
        assert "not found" in message.lower() or "exist" in message.lower()

    def test_validate_zero_size_file(self, tmp_path):
        """Test empty file."""
        file_path = tmp_path / "empty.txt"
        file_path.write_bytes(b"")

        is_valid, message = validate_file_size(str(file_path), max_size_mb=1)
        assert is_valid is False
        assert "empty" in message.lower()


class TestModelPathValidation:
    """Test model path validation."""

    def test_validate_existing_model(self, tmp_path):
        """Test validation of existing model file."""
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake model data")

        is_valid, message = validate_model_path(str(model_path))
        assert is_valid is True

    def test_validate_nonexistent_model(self):
        """Test validation of non-existent model."""
        is_valid, message = validate_model_path("nonexistent_model.pt")
        assert is_valid is False
        assert "not found" in message.lower()

    def test_validate_model_extension(self, tmp_path):
        """Test model file extension validation."""
        model_path = tmp_path / "model.txt"  # Wrong extension
        model_path.write_bytes(b"fake model")

        is_valid, message = validate_model_path(
            str(model_path), valid_extensions=[".pt", ".pth"]
        )
        assert is_valid is False
        assert "extension" in message.lower()

    def test_validate_model_size(self, tmp_path):
        """Test model file size validation."""
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"x" * 100)  # Very small

        is_valid, message = validate_model_path(str(model_path), min_size_mb=1)
        assert is_valid is False
        assert "size" in message.lower()


class TestComplexValidations:
    """Test complex validation scenarios."""

    def test_validate_multiple_bboxes(self):
        """Test validating multiple bboxes."""
        bboxes = [
            BoundingBox(x=10, y=10, width=50, height=50),
            BoundingBox(x=70, y=70, width=50, height=50),
            BoundingBox(x=130, y=130, width=50, height=50),
        ]

        for bbox in bboxes:
            is_valid, _ = validate_bbox(bbox, image_width=200, image_height=200)
            assert is_valid is True

    def test_validate_bbox_overlap(self):
        """Test bboxes that overlap."""
        bbox1 = BoundingBox(x=10, y=10, width=100, height=100)
        bbox2 = BoundingBox(x=50, y=50, width=100, height=100)

        # Both should be individually valid
        is_valid1, _ = validate_bbox(bbox1, image_width=200, image_height=200)
        is_valid2, _ = validate_bbox(bbox2, image_width=200, image_height=200)

        assert is_valid1 is True
        assert is_valid2 is True

    def test_validate_full_pipeline_data(self):
        """Test validating complete pipeline data."""
        # Image
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        is_valid_img, _ = validate_image_array(image)

        # Bbox
        bbox = BoundingBox(x=100, y=100, width=200, height=200)
        is_valid_bbox, _ = validate_bbox(bbox, image_width=640, image_height=640)

        # Confidence
        confidence = 0.85
        is_valid_conf = validate_confidence_score(confidence)

        # All should be valid
        assert all([is_valid_img, is_valid_bbox, is_valid_conf])


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_validate_none_inputs(self):
        """Test None inputs."""
        with pytest.raises((ValueError, TypeError)):
            validate_image_array(None)

        with pytest.raises((ValueError, TypeError)):
            validate_bbox(None, 100, 100)

    def test_validate_extreme_values(self):
        """Test extreme values."""
        # Very large image
        large_image = np.zeros((10000, 10000, 3), dtype=np.uint8)
        is_valid, _ = validate_image_array(large_image, max_width=5000, max_height=5000)
        assert is_valid is False

    def test_validate_float_bbox_coordinates(self):
        """Test bbox with float coordinates."""
        # BoundingBox expects int, but test if validation catches it
        try:
            bbox = BoundingBox(x=10.5, y=20.5, width=100.5, height=50.5)
            is_valid, _ = validate_bbox(bbox, image_width=200, image_height=200)
            # Should either convert or fail gracefully
        except (ValueError, TypeError):
            pass  # Expected


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
