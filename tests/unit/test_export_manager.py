"""
Unit tests for ExportManager.
"""

import pytest
from pathlib import Path
import json
import csv

from src.drawing_intelligence.export import ExportManager, ExportConfig
from src.drawing_intelligence.database import DatabaseManager
from tests.fixtures.test_data_generator import TestDataGenerator


@pytest.fixture
def export_config():
    """Create export configuration."""
    return ExportConfig(
        json_format="pretty",
        csv_delimiter=",",
        include_intermediate_results=False,
        include_images=False,
    )


@pytest.fixture
def test_db(tmp_path):
    """Create test database."""
    db_path = tmp_path / "test_export.db"
    db = DatabaseManager(str(db_path))

    # Store test drawing
    result = TestDataGenerator.create_mock_processing_result()
    db.store_drawing(result)

    yield db, result.drawing_id
    db.close()


@pytest.fixture
def export_manager(test_db, export_config):
    """Create ExportManager instance."""
    db, drawing_id = test_db
    return ExportManager(db, export_config), drawing_id


class TestExportManager:
    """Test ExportManager class."""

    def test_initialization(self, export_manager):
        """Test export manager initialization."""
        manager, _ = export_manager
        assert manager.config.json_format == "pretty"
        assert manager.config.csv_delimiter == ","

    def test_export_drawing_json(self, export_manager, tmp_path):
        """Test JSON export."""
        manager, drawing_id = export_manager
        output_path = tmp_path / "export.json"

        result_path = manager.export_drawing_json(drawing_id, str(output_path))

        assert Path(result_path).exists()

        # Verify JSON content
        with open(result_path, "r") as f:
            data = json.load(f)

        assert "drawing_id" in data
        assert data["drawing_id"] == drawing_id
        assert "entities" in data
        assert "detections" in data

    def test_export_drawing_csv(self, export_manager, tmp_path):
        """Test CSV export."""
        manager, drawing_id = export_manager
        output_dir = tmp_path / "csv_export"

        files = manager.export_drawing_csv(drawing_id, str(output_dir))

        assert len(files) > 0

        # Verify at least summary CSV exists
        summary_file = output_dir / f"{drawing_id}_summary.csv"
        assert summary_file.exists()

        # Verify CSV can be read
        with open(summary_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) > 0

    def test_export_invalid_drawing(self, export_manager, tmp_path):
        """Test export with invalid drawing ID."""
        manager, _ = export_manager

        with pytest.raises(ValueError):
            manager.export_drawing_json("INVALID-ID", str(tmp_path / "out.json"))

    def test_export_batch_json(self, export_manager, tmp_path):
        """Test batch JSON export."""
        manager, drawing_id = export_manager
        output_path = tmp_path / "batch.json"

        result_path = manager.export_batch_json([drawing_id], str(output_path))

        assert Path(result_path).exists()

        with open(result_path, "r") as f:
            data = json.load(f)

        assert "total_drawings" in data
        assert "drawings" in data
        assert len(data["drawings"]) == 1
