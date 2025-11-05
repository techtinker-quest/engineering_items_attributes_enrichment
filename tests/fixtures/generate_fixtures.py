"""
Test fixture generators for creating sample PDF and image files.

Run this script to generate test fixtures:
    python tests/fixtures/generate_fixtures.py
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

# Try to import reportlab for PDF generation
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: reportlab not installed. PDF generation will be limited.")
    print("Install with: pip install reportlab")


def generate_test_image(output_path: str = "test_image.png", size=(640, 640)):
    """
    Generate a test engineering drawing image with text and shapes.

    Args:
        output_path: Path to save the image
        size: Image size (width, height)
    """
    # Create white background
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)

    # Try to use a system font, fallback to default
    try:
        font_large = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24
        )
        font_medium = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
        )
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
        )
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Draw title block (bottom right)
    title_x, title_y = size[0] - 250, size[1] - 150
    draw.rectangle(
        [title_x, title_y, size[0] - 20, size[1] - 20], outline="black", width=2
    )

    # Title block content
    draw.text((title_x + 10, title_y + 10), "PART NO:", fill="black", font=font_small)
    draw.text((title_x + 80, title_y + 10), "ABC-12345", fill="black", font=font_medium)

    draw.text((title_x + 10, title_y + 35), "MATERIAL:", fill="black", font=font_small)
    draw.text((title_x + 80, title_y + 35), "Steel 304", fill="black", font=font_medium)

    draw.text((title_x + 10, title_y + 60), "SCALE:", fill="black", font=font_small)
    draw.text((title_x + 80, title_y + 60), "1:1", fill="black", font=font_medium)

    draw.text((title_x + 10, title_y + 85), "REV:", fill="black", font=font_small)
    draw.text((title_x + 80, title_y + 85), "A", fill="black", font=font_medium)

    # Draw some shapes (simulated components)
    # Rectangle (housing)
    draw.rectangle([100, 100, 300, 250], outline="black", width=2)
    draw.text((150, 260), "Housing", fill="black", font=font_medium)

    # Circle (bolt hole)
    draw.ellipse([180, 150, 220, 190], outline="black", width=2)
    draw.text((165, 200), "Ø25.4mm", fill="black", font=font_small)

    # Dimension lines
    draw.line([100, 280, 300, 280], fill="black", width=1)
    draw.line([100, 275, 100, 285], fill="black", width=1)  # Tick
    draw.line([300, 275, 300, 285], fill="black", width=1)  # Tick
    draw.text((180, 285), "200mm", fill="black", font=font_small)

    # Another shape (gear)
    gear_center = (450, 175)
    gear_radius = 50
    draw.ellipse(
        [
            gear_center[0] - gear_radius,
            gear_center[1] - gear_radius,
            gear_center[0] + gear_radius,
            gear_center[1] + gear_radius,
        ],
        outline="black",
        width=2,
    )
    # Gear teeth (simplified)
    for i in range(8):
        angle = i * 45
        import math

        x = gear_center[0] + gear_radius * math.cos(math.radians(angle))
        y = gear_center[1] + gear_radius * math.sin(math.radians(angle))
        draw.line([gear_center[0], gear_center[1], x, y], fill="black", width=1)
    draw.text(
        (gear_center[0] - 15, gear_center[1] + 60),
        "Gear",
        fill="black",
        font=font_medium,
    )

    # Save image
    img.save(output_path)
    print(f"✅ Generated test image: {output_path}")
    return output_path


def generate_test_pdf(output_path: str = "test_drawing.pdf"):
    """
    Generate a test engineering drawing PDF.

    Args:
        output_path: Path to save the PDF
    """
    if not REPORTLAB_AVAILABLE:
        print(
            "❌ Cannot generate PDF without reportlab. Install with: pip install reportlab"
        )
        return None

    # Create PDF
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(1 * inch, height - 1 * inch, "ENGINEERING DRAWING")

    # Title Block
    title_x = width - 4 * inch
    title_y = 1 * inch

    c.setFont("Helvetica", 10)
    c.rect(title_x, title_y, 3.5 * inch, 1.5 * inch, stroke=1, fill=0)

    # Title block content
    c.drawString(title_x + 0.1 * inch, title_y + 1.3 * inch, "PART NUMBER:")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(title_x + 1.2 * inch, title_y + 1.3 * inch, "ABC-12345-REV-A")

    c.setFont("Helvetica", 10)
    c.drawString(title_x + 0.1 * inch, title_y + 1.0 * inch, "MATERIAL:")
    c.setFont("Helvetica-Bold", 11)
    c.drawString(title_x + 1.2 * inch, title_y + 1.0 * inch, "Steel 304")

    c.setFont("Helvetica", 10)
    c.drawString(title_x + 0.1 * inch, title_y + 0.7 * inch, "SCALE:")
    c.setFont("Helvetica-Bold", 11)
    c.drawString(title_x + 1.2 * inch, title_y + 0.7 * inch, "1:1")

    c.setFont("Helvetica", 10)
    c.drawString(title_x + 0.1 * inch, title_y + 0.4 * inch, "REVISION:")
    c.setFont("Helvetica-Bold", 11)
    c.drawString(title_x + 1.2 * inch, title_y + 0.4 * inch, "A")

    c.drawString(title_x + 0.1 * inch, title_y + 0.1 * inch, "DRAFTER:")
    c.drawString(title_x + 1.2 * inch, title_y + 0.1 * inch, "J. Engineer")

    # Draw some shapes
    c.setFont("Helvetica", 10)

    # Rectangle (component)
    comp_x = 2 * inch
    comp_y = height - 4 * inch
    c.rect(comp_x, comp_y, 3 * inch, 2 * inch, stroke=1, fill=0)
    c.drawString(comp_x + 1.2 * inch, comp_y - 0.3 * inch, "Housing")

    # Dimension line
    c.line(comp_x, comp_y - 0.5 * inch, comp_x + 3 * inch, comp_y - 0.5 * inch)
    c.drawString(comp_x + 1.3 * inch, comp_y - 0.7 * inch, "76.2mm (3.0 in)")

    # Circle (bolt hole)
    hole_x = comp_x + 1.5 * inch
    hole_y = comp_y + 1 * inch
    c.circle(hole_x, hole_y, 0.3 * inch, stroke=1, fill=0)
    c.drawString(hole_x - 0.5 * inch, hole_y - 0.6 * inch, "Ø25.4 ± 0.1mm")

    # Notes
    c.setFont("Helvetica", 9)
    c.drawString(1 * inch, height - 5.5 * inch, "NOTES:")
    c.drawString(
        1 * inch, height - 5.8 * inch, "1. All dimensions in millimeters unless noted"
    )
    c.drawString(1 * inch, height - 6.1 * inch, "2. Material: ASTM A36 Steel")
    c.drawString(1 * inch, height - 6.4 * inch, "3. Surface finish: Ra 3.2")

    # Technical specifications
    c.setFont("Helvetica-Bold", 11)
    c.drawString(1 * inch, height - 7 * inch, "SPECIFICATIONS:")
    c.setFont("Helvetica", 9)
    c.drawString(1 * inch, height - 7.3 * inch, "Weight: 2.5 kg")
    c.drawString(1 * inch, height - 7.6 * inch, "Thread: M8 x 1.25")
    c.drawString(1 * inch, height - 7.9 * inch, "Tolerance: ±0.05mm standard")

    # Save PDF
    c.save()
    print(f"✅ Generated test PDF: {output_path}")
    return output_path


def generate_all_fixtures():
    """Generate all test fixtures."""
    # Get the directory where this script is located
    fixtures_dir = Path(__file__).parent
    sample_dir = fixtures_dir / "sample_drawings"

    # Create directories if they don't exist
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Generate fixtures
    print("Generating test fixtures...")
    print("-" * 60)

    image_path = sample_dir / "test_image.png"
    pdf_path = sample_dir / "test_drawing.pdf"

    generate_test_image(str(image_path))
    generate_test_pdf(str(pdf_path))

    print("-" * 60)
    print("✅ All fixtures generated successfully!")
    print(f"\nFiles created:")
    print(f"  - {image_path}")
    print(f"  - {pdf_path}")


if __name__ == "__main__":
    generate_all_fixtures()
