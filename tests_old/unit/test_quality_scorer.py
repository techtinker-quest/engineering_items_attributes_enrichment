"""
Unit tests for quality_scorer module.

Tests confidence calculation, review flag generation, and completeness scoring.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

from src.drawing_intelligence.quality.quality_scorer import QualityScorer, QualityConfig
from src.drawing_intelligence.models.data_structures import (
    ProcessingResult,
    OCRResult,
    EntityExtractionResult,
    DetectionResult,
    Entity,
    Detection,
    TextBlock,
    ValidationReport,
    ValidationIssue,
    EntityType,
    FlagType,
    Severity,
    PipelineType,
    BoundingBox,
    ReviewFlag,
    CompletenessScore,
    TitleBlock,
)


@pytest.fixture
def quality_config():
    """Create test quality configuration."""
    return QualityConfig(
        review_threshold=0.75,
        ocr_weight=0.30,
        detection_weight=0.40,
        entity_weight=0.30,
        flag_missing_critical_entities=True,
        critical_entities=[EntityType.PART_NUMBER],
    )


@pytest.fixture
def quality_scorer(quality_config):
    """Create quality scorer instance."""
    return QualityScorer(quality_config)


@pytest.fixture
def sample_processing_result():
    """Create sample processing result for testing."""
    # OCR Result
    text_blocks = [
        TextBlock(
            text_id="TXT-001",
            content="Part Number: ABC-123",
            bbox=BoundingBox(10, 20, 100, 30),
            confidence=0.90,
            ocr_engine="paddleocr",
            region_type="text",
        ),
        TextBlock(
            text_id="TXT-002",
            content="Material: Steel 304",
            bbox=BoundingBox(120, 20, 100, 30),
            confidence=0.85,
            ocr_engine="paddleocr",
            region_type="text",
        ),
    ]
    ocr_result = OCRResult(
        text_blocks=text_blocks,
        full_text="Part Number: ABC-123\nMaterial: Steel 304",
        average_confidence=0.875,
        language="en",
        layout_regions=[],
    )

    # Entities
    entities = [
        Entity(
            entity_id="ENT-001",
            entity_type=EntityType.PART_NUMBER,
            value="ABC-123",
            original_text="Part Number: ABC-123",
            normalized_value={"value": "ABC-123"},
            confidence=0.95,
            extraction_method="regex",
            source_text_id="TXT-001",
            bbox=BoundingBox(10, 20, 100, 30),
        ),
        Entity(
            entity_id="ENT-002",
            entity_type=EntityType.MATERIAL,
            value="Steel 304",
            original_text="Material: Steel 304",
            normalized_value={"value": "Steel 304"},
            confidence=0.90,
            extraction_method="regex",
            source_text_id="TXT-002",
            bbox=BoundingBox(120, 20, 100, 30),
        ),
    ]

    # Detections
    detections = [
        Detection(
            detection_id="DET-001",
            class_name="bolt",
            confidence=0.85,
            bbox=BoundingBox(100, 100, 50, 50),
            bbox_normalized=None,
        ),
        Detection(
            detection_id="DET-002",
            class_name="gear",
            confidence=0.80,
            bbox=BoundingBox(200, 200, 80, 80),
            bbox_normalized=None,
        ),
    ]

    # Validation Report
    validation_report = ValidationReport(
        is_valid=True, issues=[], confidence_adjustment=1.0, requires_human_review=False
    )

    # Title Block
    title_block = TitleBlock(
        part_number="ABC-123",
        oem="Test OEM",
        scale="1:1",
        date=None,
        revision="A",
        material="Steel 304",
        drafter=None,
    )

    return ProcessingResult(
        drawing_id="DWG-TEST-001",
        source_file="test_drawing.pdf",
        processing_timestamp=datetime.now(),
        pipeline_type=PipelineType.BASELINE_ONLY,
        pipeline_version="1.0.0",
        ocr_result=ocr_result,
        entities=entities,
        title_block=title_block,
        detections=detections,
        associations=[],
        hierarchy=None,
        validation_report=validation_report,
        overall_confidence=0.0,  # To be calculated
        review_flags=[],
        completeness_score=None,
        status="complete",
    )


class TestQualityScorerInitialization:
    """Test quality scorer initialization."""

    def test_init_with_config(self, quality_config):
        """Test initialization with configuration."""
        scorer = QualityScorer(quality_config)

        assert scorer.config == quality_config
        assert scorer.config.review_threshold == 0.75

    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        scorer = QualityScorer()

        assert scorer.config is not None
        assert scorer.config.review_threshold == 0.75
        assert scorer.config.ocr_weight == 0.30
        assert scorer.config.detection_weight == 0.40
        assert scorer.config.entity_weight == 0.30


class TestConfidenceCalculation:
    """Test overall confidence calculation."""

    def test_calculate_drawing_confidence(
        self, quality_scorer, sample_processing_result
    ):
        """Test basic confidence calculation."""
        confidence = quality_scorer.calculate_drawing_confidence(
            sample_processing_result
        )

        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably high for good result

    def test_weighted_confidence_formula(
        self, quality_scorer, sample_processing_result
    ):
        """Test weighted confidence calculation."""
        # OCR: 0.875, Detection avg: (0.85 + 0.80) / 2 = 0.825, Entity avg: (0.95 + 0.90) / 2 = 0.925
        # Expected: 0.30 * 0.875 + 0.40 * 0.825 + 0.30 * 0.925 = 0.8625

        confidence = quality_scorer.calculate_drawing_confidence(
            sample_processing_result
        )

        expected = 0.30 * 0.875 + 0.40 * 0.825 + 0.30 * 0.925
        assert abs(confidence - expected) < 0.01

    def test_confidence_with_validation_penalty(
        self, quality_scorer, sample_processing_result
    ):
        """Test confidence with validation penalty applied."""
        # Add validation issues
        sample_processing_result.validation_report.confidence_adjustment = (
            0.90  # 10% penalty
        )

        base_confidence = 0.30 * 0.875 + 0.40 * 0.825 + 0.30 * 0.925
        expected = base_confidence * 0.90

        confidence = quality_scorer.calculate_drawing_confidence(
            sample_processing_result
        )

        assert abs(confidence - expected) < 0.01

    def test_confidence_zero_detections(self, quality_scorer, sample_processing_result):
        """Test confidence with no detections."""
        sample_processing_result.detections = []

        confidence = quality_scorer.calculate_drawing_confidence(
            sample_processing_result
        )

        # Should still calculate but with zero detection contribution
        assert 0.0 <= confidence <= 1.0

    def test_confidence_zero_entities(self, quality_scorer, sample_processing_result):
        """Test confidence with no entities."""
        sample_processing_result.entities = []

        confidence = quality_scorer.calculate_drawing_confidence(
            sample_processing_result
        )

        assert 0.0 <= confidence <= 1.0

    def test_confidence_clamped_to_range(
        self, quality_scorer, sample_processing_result
    ):
        """Test confidence is clamped to [0, 1]."""
        # Force very high values
        sample_processing_result.ocr_result.average_confidence = (
            1.5  # Invalid but testing
        )

        confidence = quality_scorer.calculate_drawing_confidence(
            sample_processing_result
        )

        assert 0.0 <= confidence <= 1.0


class TestReviewFlagGeneration:
    """Test review flag generation."""

    def test_generate_review_flags_no_issues(
        self, quality_scorer, sample_processing_result
    ):
        """Test flag generation with no issues."""
        sample_processing_result.overall_confidence = 0.85

        flags = quality_scorer.generate_review_flags(sample_processing_result)

        # Should have no flags for good quality result
        assert isinstance(flags, list)
        assert len(flags) == 0

    def test_flag_low_confidence(self, quality_scorer, sample_processing_result):
        """Test low confidence flag generation."""
        sample_processing_result.overall_confidence = 0.65  # Below 0.75 threshold

        flags = quality_scorer.generate_review_flags(sample_processing_result)

        low_conf_flags = [f for f in flags if f.flag_type == FlagType.LOW_CONFIDENCE]
        assert len(low_conf_flags) > 0
        assert low_conf_flags[0].severity == Severity.MEDIUM

    def test_flag_critical_low_confidence(
        self, quality_scorer, sample_processing_result
    ):
        """Test critical low confidence flag."""
        sample_processing_result.overall_confidence = 0.45  # Very low

        flags = quality_scorer.generate_review_flags(sample_processing_result)

        low_conf_flags = [f for f in flags if f.flag_type == FlagType.LOW_CONFIDENCE]
        assert len(low_conf_flags) > 0
        assert low_conf_flags[0].severity == Severity.CRITICAL

    def test_flag_missing_entities(self, quality_scorer, sample_processing_result):
        """Test missing entities flag."""
        sample_processing_result.entities = []

        flags = quality_scorer.generate_review_flags(sample_processing_result)

        missing_flags = [f for f in flags if f.flag_type == FlagType.MISSING_DATA]
        assert len(missing_flags) > 0
        assert "entities" in missing_flags[0].reason.lower()

    def test_flag_missing_shapes(self, quality_scorer, sample_processing_result):
        """Test missing shapes flag."""
        sample_processing_result.detections = []

        flags = quality_scorer.generate_review_flags(sample_processing_result)

        missing_flags = [f for f in flags if f.flag_type == FlagType.MISSING_DATA]
        assert len(missing_flags) > 0
        assert (
            "shape" in missing_flags[0].reason.lower()
            or "detection" in missing_flags[0].reason.lower()
        )

    def test_flag_critical_field_missing(
        self, quality_scorer, sample_processing_result
    ):
        """Test critical field missing flag."""
        # Remove part number
        sample_processing_result.entities = [
            e
            for e in sample_processing_result.entities
            if e.entity_type != EntityType.PART_NUMBER
        ]
        sample_processing_result.title_block.part_number = None

        flags = quality_scorer.generate_review_flags(sample_processing_result)

        critical_flags = [
            f for f in flags if f.flag_type == FlagType.CRITICAL_FIELD_MISSING
        ]
        assert len(critical_flags) > 0
        assert critical_flags[0].severity == Severity.CRITICAL

    def test_flag_validation_issues(self, quality_scorer, sample_processing_result):
        """Test validation issues flag."""
        sample_processing_result.validation_report.issues = [
            ValidationIssue(
                severity=Severity.HIGH,
                type="TEST_ISSUE",
                message="Test validation issue",
                entity_id=None,
                shape_id=None,
            )
        ]
        sample_processing_result.validation_report.requires_human_review = True

        flags = quality_scorer.generate_review_flags(sample_processing_result)

        validation_flags = [
            f for f in flags if f.flag_type == FlagType.VALIDATION_ISSUE
        ]
        assert len(validation_flags) > 0

    def test_flag_processing_error(self, quality_scorer, sample_processing_result):
        """Test processing error flag."""
        sample_processing_result.status = "failed"
        sample_processing_result.error_message = "Test error"

        flags = quality_scorer.generate_review_flags(sample_processing_result)

        error_flags = [f for f in flags if f.flag_type == FlagType.PROCESSING_ERROR]
        assert len(error_flags) > 0
        assert error_flags[0].severity == Severity.CRITICAL

    def test_multiple_flags(self, quality_scorer, sample_processing_result):
        """Test generation of multiple flags."""
        sample_processing_result.overall_confidence = 0.65  # Low
        sample_processing_result.entities = []  # Missing entities

        flags = quality_scorer.generate_review_flags(sample_processing_result)

        # Should have at least 2 flags
        assert len(flags) >= 2
        flag_types = {f.flag_type for f in flags}
        assert FlagType.LOW_CONFIDENCE in flag_types
        assert FlagType.MISSING_DATA in flag_types


class TestCompletenessScore:
    """Test completeness scoring."""

    def test_assess_completeness(self, quality_scorer, sample_processing_result):
        """Test basic completeness assessment."""
        completeness = quality_scorer.assess_completeness(sample_processing_result)

        assert isinstance(completeness, CompletenessScore)
        assert 0.0 <= completeness.overall_score <= 1.0

    def test_completeness_has_part_number(
        self, quality_scorer, sample_processing_result
    ):
        """Test completeness recognizes part number."""
        completeness = quality_scorer.assess_completeness(sample_processing_result)

        assert completeness.has_part_number is True

    def test_completeness_missing_part_number(
        self, quality_scorer, sample_processing_result
    ):
        """Test completeness detects missing part number."""
        sample_processing_result.entities = [
            e
            for e in sample_processing_result.entities
            if e.entity_type != EntityType.PART_NUMBER
        ]
        sample_processing_result.title_block.part_number = None

        completeness = quality_scorer.assess_completeness(sample_processing_result)

        assert completeness.has_part_number is False
        assert EntityType.PART_NUMBER.value in completeness.missing_critical_fields

    def test_completeness_has_dimensions(
        self, quality_scorer, sample_processing_result
    ):
        """Test completeness recognizes dimensions."""
        # Add dimension entity
        dim_entity = Entity(
            entity_id="ENT-003",
            entity_type=EntityType.DIMENSION,
            value="25.4mm",
            original_text="Ø25.4mm",
            normalized_value={"value": 25.4, "unit": "mm"},
            confidence=0.90,
            extraction_method="regex",
            source_text_id="TXT-003",
            bbox=BoundingBox(50, 50, 80, 20),
        )
        sample_processing_result.entities.append(dim_entity)

        completeness = quality_scorer.assess_completeness(sample_processing_result)

        assert completeness.has_dimensions is True

    def test_completeness_has_shapes(self, quality_scorer, sample_processing_result):
        """Test completeness recognizes shapes."""
        completeness = quality_scorer.assess_completeness(sample_processing_result)

        assert completeness.has_shapes is True

    def test_completeness_has_title_block(
        self, quality_scorer, sample_processing_result
    ):
        """Test completeness recognizes title block."""
        completeness = quality_scorer.assess_completeness(sample_processing_result)

        assert completeness.has_title_block is True

    def test_completeness_by_category(self, quality_scorer, sample_processing_result):
        """Test completeness breakdown by category."""
        completeness = quality_scorer.assess_completeness(sample_processing_result)

        assert "entities" in completeness.completeness_by_category
        assert "shapes" in completeness.completeness_by_category
        assert "associations" in completeness.completeness_by_category
        assert "title_block" in completeness.completeness_by_category

    def test_completeness_weighted_score(
        self, quality_scorer, sample_processing_result
    ):
        """Test completeness uses weighted scoring."""
        completeness = quality_scorer.assess_completeness(sample_processing_result)

        # With 2 entities, 2 detections, title block
        # Should have reasonable overall score
        assert completeness.overall_score > 0.5

    def test_completeness_minimal_data(self, quality_scorer, sample_processing_result):
        """Test completeness with minimal data."""
        sample_processing_result.entities = []
        sample_processing_result.detections = []
        sample_processing_result.title_block = None

        completeness = quality_scorer.assess_completeness(sample_processing_result)

        assert completeness.overall_score < 0.5  # Low completeness


class TestReviewDecision:
    """Test review decision logic."""

    def test_needs_review_low_confidence(
        self, quality_scorer, sample_processing_result
    ):
        """Test needs review due to low confidence."""
        sample_processing_result.overall_confidence = 0.65
        flags = quality_scorer.generate_review_flags(sample_processing_result)
        sample_processing_result.review_flags = flags

        needs_review = sample_processing_result.needs_human_review()

        assert needs_review is True

    def test_needs_review_critical_flag(self, quality_scorer, sample_processing_result):
        """Test needs review due to critical flag."""
        sample_processing_result.overall_confidence = 0.85  # High confidence
        sample_processing_result.review_flags = [
            ReviewFlag(
                flag_id="FLAG-001",
                flag_type=FlagType.CRITICAL_FIELD_MISSING,
                severity=Severity.CRITICAL,
                reason="Missing part number",
                details={},
                suggested_action="Add part number",
            )
        ]

        needs_review = sample_processing_result.needs_human_review()

        assert needs_review is True

    def test_needs_review_high_severity(self, quality_scorer, sample_processing_result):
        """Test needs review due to high severity issue."""
        sample_processing_result.overall_confidence = 0.85
        sample_processing_result.review_flags = [
            ReviewFlag(
                flag_id="FLAG-001",
                flag_type=FlagType.MISSING_DATA,
                severity=Severity.HIGH,
                reason="No entities found",
                details={},
                suggested_action="Review extraction",
            )
        ]

        needs_review = sample_processing_result.needs_human_review()

        assert needs_review is True

    def test_no_review_needed(self, quality_scorer, sample_processing_result):
        """Test no review needed for good quality result."""
        sample_processing_result.overall_confidence = 0.85
        sample_processing_result.review_flags = []

        needs_review = sample_processing_result.needs_human_review()

        assert needs_review is False


class TestCriticalEntityChecks:
    """Test critical entity checking."""

    def test_check_critical_entities_present(
        self, quality_scorer, sample_processing_result
    ):
        """Test checking when critical entities present."""
        flags = quality_scorer.generate_review_flags(sample_processing_result)

        critical_flags = [
            f for f in flags if f.flag_type == FlagType.CRITICAL_FIELD_MISSING
        ]
        assert len(critical_flags) == 0

    def test_check_multiple_critical_entities(
        self, quality_scorer, sample_processing_result
    ):
        """Test checking multiple critical entity types."""
        config = QualityConfig(
            critical_entities=[
                EntityType.PART_NUMBER,
                EntityType.OEM,
                EntityType.DIMENSION,
            ]
        )
        scorer = QualityScorer(config)

        # Missing OEM and DIMENSION
        flags = scorer.generate_review_flags(sample_processing_result)

        # Should flag missing critical entities
        critical_flags = [
            f for f in flags if f.flag_type == FlagType.CRITICAL_FIELD_MISSING
        ]
        assert len(critical_flags) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_calculate_confidence_no_data(self, quality_scorer):
        """Test confidence calculation with no data."""
        result = ProcessingResult(
            drawing_id="DWG-TEST",
            source_file="test.pdf",
            processing_timestamp=datetime.now(),
            pipeline_type=PipelineType.BASELINE_ONLY,
            pipeline_version="1.0.0",
            ocr_result=OCRResult(
                text_blocks=[],
                full_text="",
                average_confidence=0.0,
                language="en",
                layout_regions=[],
            ),
            entities=[],
            title_block=None,
            detections=[],
            associations=[],
            hierarchy=None,
            validation_report=ValidationReport(True, [], 1.0, False),
            status="complete",
        )

        confidence = quality_scorer.calculate_drawing_confidence(result)

        assert confidence == 0.0

    def test_generate_flags_failed_result(self, quality_scorer):
        """Test flag generation for failed processing."""
        result = ProcessingResult(
            drawing_id="DWG-TEST",
            source_file="test.pdf",
            processing_timestamp=datetime.now(),
            pipeline_type=PipelineType.BASELINE_ONLY,
            pipeline_version="1.0.0",
            status="failed",
            error_message="Processing failed",
        )

        flags = quality_scorer.generate_review_flags(result)

        # Should have error flag
        error_flags = [f for f in flags if f.flag_type == FlagType.PROCESSING_ERROR]
        assert len(error_flags) > 0


class TestConfigurationOptions:
    """Test different configuration options."""

    def test_custom_review_threshold(self, sample_processing_result):
        """Test custom review threshold."""
        config = QualityConfig(review_threshold=0.85)
        scorer = QualityScorer(config)

        sample_processing_result.overall_confidence = 0.80  # Below custom threshold
        flags = scorer.generate_review_flags(sample_processing_result)

        # Should flag as low confidence
        low_conf_flags = [f for f in flags if f.flag_type == FlagType.LOW_CONFIDENCE]
        assert len(low_conf_flags) > 0

    def test_custom_weights(self, sample_processing_result):
        """Test custom confidence weights."""
        config = QualityConfig(
            ocr_weight=0.50, detection_weight=0.25, entity_weight=0.25
        )
        scorer = QualityScorer(config)

        confidence = scorer.calculate_drawing_confidence(sample_processing_result)

        # Should use custom weights
        assert 0.0 <= confidence <= 1.0

    def test_disable_critical_entity_flag(self, sample_processing_result):
        """Test disabling critical entity flagging."""
        config = QualityConfig(flag_missing_critical_entities=False)
        scorer = QualityScorer(config)

        # Remove part number
        sample_processing_result.entities = []
        sample_processing_result.title_block = None

        flags = scorer.generate_review_flags(sample_processing_result)

        # Should NOT flag missing critical entities
        critical_flags = [
            f for f in flags if f.flag_type == FlagType.CRITICAL_FIELD_MISSING
        ]
        assert len(critical_flags) == 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
