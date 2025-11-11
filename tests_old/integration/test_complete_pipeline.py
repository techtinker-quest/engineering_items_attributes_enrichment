"""
Integration Test: Complete Pipeline

Tests the entire processing pipeline end-to-end.
"""

import pytest
from pathlib import Path
import numpy as np

from src.drawing_intelligence.orchestration import PipelineOrchestrator
from src.drawing_intelligence.database import DatabaseManager
from src.drawing_intelligence.orchestration import CheckpointManager, RoutingEngine
from src.drawing_intelligence.llm.budget_controller import BudgetController
from ..fixtures.test_data_generator import TestDataGenerator
from ..utils.test_helpers import (
    setup_test_database,
    cleanup_test_database,
    create_test_config,
    assert_processing_result_valid,
)


@pytest.mark.integration
class TestCompletePipeline:
    """Test complete processing pipeline."""

    def setup_method(self):
        """Setup test environment."""
        self.config = create_test_config()
        self.db = setup_test_database()

        # Create checkpoint manager
        checkpoint_dir = Path("tests/test_data/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(str(checkpoint_dir))

        # Create budget controller
        self.budget_controller = BudgetController(
            daily_budget_usd=1.0, per_drawing_limit_usd=0.10, db_manager=self.db
        )

        # Create routing engine
        self.routing_engine = RoutingEngine(self.config, self.budget_controller)

    def teardown_method(self):
        """Cleanup test environment."""
        cleanup_test_database(self.db)

        # Cleanup checkpoint directory
        checkpoint_dir = Path("tests/test_data/checkpoints")
        if checkpoint_dir.exists():
            import shutil

            shutil.rmtree(checkpoint_dir, ignore_errors=True)

    @pytest.mark.slow
    def test_process_single_drawing_baseline(self):
        """Test processing single drawing with baseline pipeline."""
        # Create test PDF
        pdf_path = "tests/test_data/test_drawing.pdf"
        Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            TestDataGenerator.create_mock_pdf(pdf_path, num_pages=1)
        except ImportError:
            pytest.skip("reportlab not available")

        # Create orchestrator
        orchestrator = PipelineOrchestrator(
            config=self.config,
            db=self.db,
            checkpoint_manager=self.checkpoint_manager,
            routing_engine=self.routing_engine,
        )

        # Process drawing
        result = orchestrator.process_drawing(pdf_path=pdf_path, force_llm=False)

        # Validate result
        assert_processing_result_valid(result)
        assert result.status == "complete"
        assert result.drawing_id is not None
        assert result.source_file == pdf_path

        # Check that stages completed
        assert "total" in result.processing_times
        assert result.processing_times["total"] > 0

        # Verify stored in database
        stored_result = self.db.get_drawing_by_id(result.drawing_id)
        assert stored_result is not None
        assert stored_result.drawing_id == result.drawing_id

    @pytest.mark.slow
    def test_process_drawing_with_entities(self):
        """Test that entities are extracted."""
        pdf_path = "tests/test_data/test_drawing_entities.pdf"
        Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            TestDataGenerator.create_mock_pdf(pdf_path, num_pages=1, add_content=True)
        except ImportError:
            pytest.skip("reportlab not available")

        orchestrator = PipelineOrchestrator(
            config=self.config,
            db=self.db,
            checkpoint_manager=self.checkpoint_manager,
            routing_engine=self.routing_engine,
        )

        result = orchestrator.process_drawing(pdf_path)

        # Should have some entities (at least from embedded text)
        assert result.entities is not None
        assert isinstance(result.entities, list)

        # Check for expected entity types
        if len(result.entities) > 0:
            entity_types = [e.entity_type for e in result.entities]
            assert len(entity_types) > 0

    def test_process_drawing_error_handling(self):
        """Test error handling for invalid PDF."""
        pdf_path = "tests/test_data/nonexistent.pdf"

        orchestrator = PipelineOrchestrator(
            config=self.config,
            db=self.db,
            checkpoint_manager=self.checkpoint_manager,
            routing_engine=self.routing_engine,
        )

        # Should raise error for nonexistent file
        from src.drawing_intelligence.utils.error_handlers import DrawingProcessingError

        with pytest.raises(DrawingProcessingError):
            orchestrator.process_drawing(pdf_path)

    @pytest.mark.slow
    def test_quality_scoring_integration(self):
        """Test quality scoring is applied."""
        pdf_path = "tests/test_data/test_drawing_quality.pdf"
        Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            TestDataGenerator.create_mock_pdf(pdf_path)
        except ImportError:
            pytest.skip("reportlab not available")

        orchestrator = PipelineOrchestrator(
            config=self.config,
            db=self.db,
            checkpoint_manager=self.checkpoint_manager,
            routing_engine=self.routing_engine,
        )

        result = orchestrator.process_drawing(pdf_path)

        # Check quality metrics
        assert result.overall_confidence is not None
        assert 0.0 <= result.overall_confidence <= 1.0

        # Check review flags
        assert result.review_flags is not None
        assert isinstance(result.review_flags, list)

        # Check completeness score
        if result.completeness_score:
            assert 0.0 <= result.completeness_score.overall_score <= 1.0

    def test_audit_trail_creation(self):
        """Test that audit trail is created."""
        pdf_path = "tests/test_data/test_drawing_audit.pdf"
        Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            TestDataGenerator.create_mock_pdf(pdf_path)
        except ImportError:
            pytest.skip("reportlab not available")

        orchestrator = PipelineOrchestrator(
            config=self.config,
            db=self.db,
            checkpoint_manager=self.checkpoint_manager,
            routing_engine=self.routing_engine,
        )

        result = orchestrator.process_drawing(pdf_path)

        # Check audit trail
        audit_entries = self.db.get_audit_trail(result.drawing_id)
        assert audit_entries is not None
        assert len(audit_entries) > 0

        # Verify stages
        stages = [entry.stage for entry in audit_entries]
        assert "pdf_extraction" in stages or "complete" in stages


@pytest.mark.integration
class TestBatchProcessing:
    """Test batch processing functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.config = create_test_config()
        self.db = setup_test_database()

        checkpoint_dir = Path("tests/test_data/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(str(checkpoint_dir))

        self.budget_controller = BudgetController(
            daily_budget_usd=10.0, per_drawing_limit_usd=1.0, db_manager=self.db
        )

        self.routing_engine = RoutingEngine(self.config, self.budget_controller)

    def teardown_method(self):
        """Cleanup test environment."""
        cleanup_test_database(self.db)

        checkpoint_dir = Path("tests/test_data/checkpoints")
        if checkpoint_dir.exists():
            import shutil

            shutil.rmtree(checkpoint_dir, ignore_errors=True)

    @pytest.mark.slow
    def test_batch_processing(self):
        """Test batch processing multiple drawings."""
        # Create test PDFs
        pdf_dir = Path("tests/test_data/batch")
        pdf_dir.mkdir(parents=True, exist_ok=True)

        pdf_paths = []
        for i in range(3):
            pdf_path = pdf_dir / f"drawing_{i+1}.pdf"
            try:
                TestDataGenerator.create_mock_pdf(str(pdf_path))
                pdf_paths.append(str(pdf_path))
            except ImportError:
                pytest.skip("reportlab not available")

        # Create orchestrator
        orchestrator = PipelineOrchestrator(
            config=self.config,
            db=self.db,
            checkpoint_manager=self.checkpoint_manager,
            routing_engine=self.routing_engine,
        )

        # Process batch
        batch_result = orchestrator.process_batch(
            pdf_paths=pdf_paths, batch_id="TEST-BATCH-001", parallel_workers=2
        )

        # Validate batch result
        assert batch_result.batch_id == "TEST-BATCH-001"
        assert batch_result.total_drawings == 3
        assert batch_result.successful >= 0
        assert batch_result.failed >= 0
        assert batch_result.successful + batch_result.failed == 3

        # Check success rate
        assert 0.0 <= batch_result.success_rate <= 1.0

        # Check individual results
        assert len(batch_result.drawing_results) == 3
        for result in batch_result.drawing_results:
            assert result.drawing_id is not None

    @pytest.mark.slow
    def test_checkpoint_resume(self):
        """Test resuming batch from checkpoint."""
        pdf_dir = Path("tests/test_data/batch_resume")
        pdf_dir.mkdir(parents=True, exist_ok=True)

        pdf_paths = []
        for i in range(2):
            pdf_path = pdf_dir / f"drawing_{i+1}.pdf"
            try:
                TestDataGenerator.create_mock_pdf(str(pdf_path))
                pdf_paths.append(str(pdf_path))
            except ImportError:
                pytest.skip("reportlab not available")

        orchestrator = PipelineOrchestrator(
            config=self.config,
            db=self.db,
            checkpoint_manager=self.checkpoint_manager,
            routing_engine=self.routing_engine,
        )

        batch_id = "TEST-BATCH-RESUME"

        # Process first batch
        batch_result = orchestrator.process_batch(
            pdf_paths=pdf_paths, batch_id=batch_id, parallel_workers=1
        )

        # Verify checkpoint exists
        checkpoint_state = self.checkpoint_manager.get_checkpoint_state(batch_id)
        assert checkpoint_state is not None

        # All files should be complete
        for file_path in pdf_paths:
            status = checkpoint_state.file_status.get(file_path)
            assert status == "complete"


@pytest.mark.integration
class TestDatabaseIntegration:
    """Test database integration."""

    def setup_method(self):
        """Setup test database."""
        self.db = setup_test_database()

    def teardown_method(self):
        """Cleanup test database."""
        cleanup_test_database(self.db)

    def test_store_and_retrieve_drawing(self):
        """Test storing and retrieving drawing."""
        # Create mock result
        result = TestDataGenerator.create_mock_processing_result()

        # Store in database
        drawing_id = self.db.store_drawing(result)
        assert drawing_id == result.drawing_id

        # Retrieve from database
        stored_result = self.db.get_drawing_by_id(drawing_id)
        assert stored_result is not None
        assert stored_result.drawing_id == drawing_id
        assert stored_result.source_file == result.source_file

    def test_query_drawings(self):
        """Test querying drawings with filters."""
        # Store multiple drawings
        for i in range(5):
            result = TestDataGenerator.create_mock_processing_result()
            self.db.store_drawing(result)

        # Query all
        from src.drawing_intelligence.database import QueryFilters

        filters = QueryFilters(limit=10)
        drawings = self.db.query_drawings(filters)

        assert len(drawings) == 5

    def test_database_stats(self):
        """Test database statistics."""
        # Store some drawings
        for i in range(3):
            result = TestDataGenerator.create_mock_processing_result()
            self.db.store_drawing(result)

        # Get stats
        stats = self.db.get_database_stats()
        assert stats["total_drawings"] == 3
        assert stats["database_size_mb"] >= 0
