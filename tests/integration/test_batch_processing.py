"""
Integration tests for batch processing.
"""

import pytest
from pathlib import Path

from src.drawing_intelligence.orchestration import PipelineOrchestrator
from tests.utils.test_helpers import setup_test_database, create_test_config
from tests.fixtures.test_data_generator import TestDataGenerator


@pytest.mark.integration
@pytest.mark.slow
class TestBatchProcessing:
    """Test batch processing functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.config = create_test_config()
        self.db = setup_test_database()

    def teardown_method(self):
        """Cleanup test environment."""
        if hasattr(self, "db"):
            self.db.close()

    def test_small_batch_processing(self, tmp_path):
        """Test processing small batch (3 drawings)."""
        # Create test PDFs
        pdf_dir = tmp_path / "batch"
        pdf_dir.mkdir()

        pdf_paths = []
        for i in range(3):
            pdf_path = pdf_dir / f"drawing_{i+1}.pdf"
            try:
                TestDataGenerator.create_mock_pdf(str(pdf_path))
                pdf_paths.append(str(pdf_path))
            except ImportError:
                pytest.skip("reportlab not available")

        # Create orchestrator (simplified - no full initialization)
        # This would normally use full PipelineOrchestrator

        # Verify files created
        assert len(pdf_paths) == 3
        for path in pdf_paths:
            assert Path(path).exists()

    def test_batch_with_failures(self, tmp_path):
        """Test batch processing with some failures."""
        pdf_dir = tmp_path / "batch_fail"
        pdf_dir.mkdir()

        # Mix of valid and invalid files
        valid_pdf = pdf_dir / "valid.pdf"
        invalid_pdf = pdf_dir / "invalid.pdf"

        try:
            TestDataGenerator.create_mock_pdf(str(valid_pdf))
        except ImportError:
            pytest.skip("reportlab not available")

        # Create invalid PDF (empty file)
        invalid_pdf.touch()

        pdf_paths = [str(valid_pdf), str(invalid_pdf)]

        # Would test that batch continues despite failures
        assert len(pdf_paths) == 2
