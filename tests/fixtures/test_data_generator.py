"""
Test Data Generator

Generate synthetic test data for unit and integration tests.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional
from datetime import datetime

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from ...models.data_structures import *


class TestDataGenerator:
    """Generate test data for unit tests."""

    @staticmethod
    def create_mock_image(
        width: int = 800,
        height: int = 600,
        add_text: bool = False,
        add_shapes: bool = False,
    ) -> np.ndarray:
        """
        Create synthetic test image.

        Args:
            width: Image width
            height: Image height
            add_text: Add text annotations
            add_shapes: Add geometric shapes

        Returns:
            Numpy array image
        """
        # Create white background
        if PIL_AVAILABLE:
            img = Image.new("L", (width, height), color=255)
            draw = ImageDraw.Draw(img)

            if add_shapes:
                # Rectangle
                draw.rectangle([100, 100, 300, 200], outline=0, width=2)
                # Circle
                draw.ellipse([500, 350, 650, 500], outline=0, width=2)
                # Line
                draw.line([100, 400, 700, 400], fill=0, width=2)

            if add_text:
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()

                draw.text((150, 250), "PART NO: ABC-12345", fill=0, font=font)
                draw.text((150, 280), "Ø25.4 ± 0.1 mm", fill=0, font=font)
                draw.text((150, 310), "MATERIAL: Steel 304", fill=0, font=font)

            return np.array(img)
        else:
            # Fallback: simple numpy array
            image = np.ones((height, width), dtype=np.uint8) * 255

            if add_shapes:
                # Rectangle
                image[100:200, 100:300] = 0
                # Circle approximation
                center_y, center_x = 425, 575
                y, x = np.ogrid[:height, :width]
                mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= 75**2
                image[mask] = 0

            return image

    @staticmethod
    def create_mock_pdf(
        output_path: str, num_pages: int = 1, add_content: bool = True
    ) -> str:
        """
        Create synthetic test PDF.

        Args:
            output_path: Output file path
            num_pages: Number of pages
            add_content: Add text content

        Returns:
            Output file path
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError("reportlab required for PDF generation")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        c = canvas.Canvas(str(output_path), pagesize=letter)
        width, height = letter

        for page in range(num_pages):
            if add_content:
                # Title
                c.setFont("Helvetica-Bold", 16)
                c.drawString(100, height - 100, f"Technical Drawing - Page {page + 1}")

                # Part information
                c.setFont("Helvetica", 12)
                y = height - 150
                c.drawString(100, y, "PART NUMBER: ABC-12345-REV2")
                y -= 20
                c.drawString(100, y, "MANUFACTURER: Test OEM Inc.")
                y -= 20
                c.drawString(100, y, "MATERIAL: Stainless Steel 304")
                y -= 20
                c.drawString(100, y, "WEIGHT: 2.5 kg")

                # Dimensions
                y -= 40
                c.setFont("Helvetica-Bold", 12)
                c.drawString(100, y, "DIMENSIONS:")
                c.setFont("Helvetica", 10)
                y -= 20
                c.drawString(120, y, "Ø25.4 ± 0.1 mm")
                y -= 15
                c.drawString(120, y, "Length: 100.0 ± 0.5 mm")
                y -= 15
                c.drawString(120, y, "Thread: M8 x 1.25")

                # Simple shapes
                c.rect(100, 200, 200, 100, stroke=1, fill=0)
                c.circle(500, 250, 50, stroke=1, fill=0)

            c.showPage()

        c.save()
        return str(output_path)

    @staticmethod
    def create_mock_text_blocks(count: int = 10) -> List[TextBlock]:
        """
        Create mock text blocks.

        Args:
            count: Number of text blocks

        Returns:
            List of TextBlock objects
        """
        text_blocks = []

        sample_texts = [
            "PART NO: ABC-12345",
            "Ø25.4 ± 0.1 mm",
            "MATERIAL: Steel 304",
            "WEIGHT: 2.5 kg",
            "SCALE 1:1",
            "REV: A",
            "DATE: 2025-01-15",
            "M8 x 1.25",
            "TOLERANCE: ±0.05",
            "FINISH: Ra 1.6",
        ]

        for i in range(count):
            text = sample_texts[i % len(sample_texts)]

            text_block = TextBlock(
                text_id=f"TXT-{i+1:03d}",
                content=text,
                bbox=BoundingBox(
                    x=100 + (i % 3) * 200, y=100 + (i // 3) * 50, width=150, height=30
                ),
                confidence=0.85 + np.random.random() * 0.14,
                ocr_engine="paddleocr",
                region_type="text",
            )

            text_blocks.append(text_block)

        return text_blocks

    @staticmethod
    def create_mock_entities(count: int = 5) -> List[Entity]:
        """
        Create mock entities.

        Args:
            count: Number of entities

        Returns:
            List of Entity objects
        """
        entities = []

        entity_specs = [
            (
                EntityType.PART_NUMBER,
                "ABC-12345",
                "PART NO: ABC-12345",
                {"value": "ABC-12345"},
                "regex",
            ),
            (
                EntityType.DIMENSION,
                "25.4",
                "Ø25.4 ± 0.1 mm",
                {"value": 25.4, "unit": "mm", "tolerance": 0.1},
                "regex",
            ),
            (
                EntityType.MATERIAL,
                "Steel 304",
                "MATERIAL: Steel 304",
                {"material": "Steel 304"},
                "regex",
            ),
            (
                EntityType.WEIGHT,
                "2.5",
                "WEIGHT: 2.5 kg",
                {"value": 2.5, "unit": "kg"},
                "regex",
            ),
            (
                EntityType.THREAD_SPEC,
                "M8x1.25",
                "M8 x 1.25",
                {"size": "M8", "pitch": 1.25},
                "regex",
            ),
        ]

        for i in range(min(count, len(entity_specs))):
            entity_type, value, original, normalized, method = entity_specs[i]

            entity = Entity(
                entity_id=f"ENT-{i+1:03d}",
                entity_type=entity_type,
                value=value,
                original_text=original,
                normalized_value=normalized,
                confidence=0.85 + np.random.random() * 0.14,
                extraction_method=method,
                source_text_id=f"TXT-{i+1:03d}",
                bbox=BoundingBox(x=100, y=100 + i * 50, width=150, height=30),
            )

            entities.append(entity)

        return entities

    @staticmethod
    def create_mock_detections(count: int = 3) -> List[Detection]:
        """
        Create mock shape detections.

        Args:
            count: Number of detections

        Returns:
            List of Detection objects
        """
        detections = []

        classes = ["bolt", "nut", "washer", "gear", "bearing"]

        for i in range(count):
            class_name = classes[i % len(classes)]

            x = 200 + i * 150
            y = 200
            w = 100
            h = 100

            detection = Detection(
                detection_id=f"DET-{i+1:03d}",
                class_name=class_name,
                confidence=0.75 + np.random.random() * 0.24,
                bbox=BoundingBox(x=x, y=y, width=w, height=h),
                bbox_normalized=NormalizedBBox(
                    x_center=(x + w / 2) / 800,
                    y_center=(y + h / 2) / 600,
                    width=w / 800,
                    height=h / 600,
                ),
            )

            detections.append(detection)

        return detections

    @staticmethod
    def create_mock_processing_result(
        drawing_id: Optional[str] = None, include_all: bool = True
    ) -> ProcessingResult:
        """
        Create complete mock processing result.

        Args:
            drawing_id: Drawing ID (generated if None)
            include_all: Include all optional fields

        Returns:
            ProcessingResult object
        """
        if drawing_id is None:
            from ...utils.file_utils import generate_unique_id

            drawing_id = generate_unique_id("DWG")

        # Create components
        text_blocks = TestDataGenerator.create_mock_text_blocks(10)
        entities = TestDataGenerator.create_mock_entities(5)
        detections = TestDataGenerator.create_mock_detections(3)

        # Create OCR result
        ocr_result = OCRResult(
            text_blocks=text_blocks,
            language_detected="en",
            layout_regions=[],
            average_confidence=0.88,
        )

        # Create associations
        associations = []
        if include_all and len(text_blocks) > 0 and len(detections) > 0:
            assoc = Association(
                association_id="ASSOC-001",
                text_id=text_blocks[0].text_id,
                shape_id=detections[0].detection_id,
                relationship_type="dimension",
                confidence=0.85,
                distance_pixels=120.5,
            )
            associations.append(assoc)

        # Create validation report
        validation_report = None
        if include_all:
            validation_report = ValidationReport(
                is_valid=True,
                issues=[],
                confidence_adjustment=1.0,
                requires_human_review=False,
            )

        # Create processing result
        result = ProcessingResult(
            drawing_id=drawing_id,
            source_file="test_drawing.pdf",
            processing_timestamp=datetime.now(),
            pipeline_type=PipelineType.BASELINE_ONLY,
            pipeline_version="1.0.0",
            pdf_pages=[],
            ocr_result=ocr_result,
            entities=entities,
            title_block=None,
            detections=detections,
            associations=associations,
            hierarchy=None,
            validation_report=validation_report,
            overall_confidence=0.85,
            confidence_scores=None,
            review_flags=[],
            completeness_score=None,
            llm_usage=[],
            processing_times={"total": 1.5},
            status="complete",
        )

        return result
