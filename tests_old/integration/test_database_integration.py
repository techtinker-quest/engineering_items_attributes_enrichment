"""
Integration tests for database operations.
"""

import pytest
from datetime import datetime

from src.drawing_intelligence.database import DatabaseManager, QueryFilters
from tests.fixtures.test_data_generator import TestDataGenerator
from tests.utils.test_helpers import setup_test_database, cleanup_test_database


@pytest.mark.integration
class TestDatabaseIntegration:
    """Test database integration."""

    def setup_method(self):
        """Setup test database."""
        self.db = setup_test_database()

    def teardown_method(self):
        """Cleanup test database."""
        cleanup_test_database(self.db)

    def test_store_and_retrieve_complete_drawing(self):
        """Test storing and retrieving complete drawing."""
        # Create mock result
        result = TestDataGenerator.create_mock_processing_result()

        # Store in database
        drawing_id = self.db.store_drawing(result)

        assert drawing_id == result.drawing_id

        # Retrieve from database
        stored = self.db.get_drawing_by_id(drawing_id)

        assert stored is not None
        assert stored.drawing_id == drawing_id
        assert stored.source_file == result.source_file
        assert len(stored.entities) == len(result.entities)
        assert len(stored.detections) == len(result.detections)

    def test_query_with_filters(self):
        """Test querying with various filters."""
        # Store multiple drawings
        for i in range(5):
            result = TestDataGenerator.create_mock_processing_result()
            self.db.store_drawing(result)

        # Query all
        filters = QueryFilters(limit=10)
        drawings = self.db.query_drawings(filters)

        assert len(drawings) == 5

        # Query with confidence filter
        filters = QueryFilters(min_confidence=0.80, limit=10)
        drawings = self.db.query_drawings(filters)

        assert all(d.overall_confidence >= 0.80 for d in drawings)

    def test_search_entities(self):
        """Test full-text search on entities."""
        # Store drawing with known entities
        result = TestDataGenerator.create_mock_processing_result()
        self.db.store_drawing(result)

        # Search (if FTS is implemented)
        # results = self.db.search_entities("ABC-12345")
        # assert len(results) > 0

    def test_database_statistics(self):
        """Test getting database statistics."""
        # Store some drawings
        for i in range(3):
            result = TestDataGenerator.create_mock_processing_result()
            self.db.store_drawing(result)

        stats = self.db.get_database_stats()

        assert stats["total_drawings"] == 3
        assert stats["database_size_mb"] >= 0
