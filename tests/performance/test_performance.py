"""
Performance tests for drawing intelligence system.
"""

import pytest
import time
from pathlib import Path

from tests.fixtures.test_data_generator import TestDataGenerator


@pytest.mark.performance
class TestPerformance:
    """Performance tests."""

    @pytest.mark.slow
    def test_pdf_processing_speed(self, tmp_path):
        """Test PDF processing speed."""
        pdf_path = tmp_path / "perf_test.pdf"

        try:
            TestDataGenerator.create_mock_pdf(str(pdf_path), num_pages=5)
        except ImportError:
            pytest.skip("reportlab not available")
