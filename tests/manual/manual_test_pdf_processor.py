"""
Manual testing script for PDF processor.

Location: tests/manual/manual_test_pdf_processor.py

This script provides simple manual tests that you can run to verify
the PDF processor works with real files.

Usage:
    python manual_test_pdf_processor.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from drawing_intelligence.processing.pdf_processor import (
    PDFConfig,
    PDFCorruptedError,
    PDFEncryptionError,
    PDFProcessor,
)


def create_test_pdf(output_path: Path) -> None:
    """Create a simple test PDF using reportlab if available."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(str(output_path), pagesize=letter)

        # Page 1
        c.drawString(100, 750, "ENGINEERING DRAWING")
        c.drawString(100, 730, "Part Number: ABC-123")
        c.drawString(100, 710, "Material: Steel")
        c.drawString(100, 690, "Dimensions: 100mm x 50mm")
        c.showPage()

        # Page 2
        c.drawString(100, 750, "BILL OF MATERIALS")
        c.drawString(100, 730, "Item 1: Bolt M6x20")
        c.drawString(100, 710, "Item 2: Nut M6")
        c.showPage()

        c.save()
        print(f"✓ Created test PDF: {output_path}")
        return True

    except ImportError:
        print("⚠ reportlab not installed. Install with: pip install reportlab")
        return False


def test_basic_functionality():
    """Test basic PDF processor functionality."""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Functionality")
    print("=" * 70)

    # Create test PDF
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    test_pdf = test_dir / "test_drawing.pdf"

    if not create_test_pdf(test_pdf):
        print("❌ Could not create test PDF. Skipping test.")
        return

    # Initialize processor
    config = PDFConfig(dpi=300, max_pages=10)
    processor = PDFProcessor(config)

    try:
        # Test metadata extraction
        print("\n1. Testing metadata extraction...")
        metadata = processor.get_pdf_metadata(test_pdf)
        print(f"   ✓ Pages: {metadata['num_pages']}")
        print(f"   ✓ Format: {metadata['format']}")

        # Test text extraction
        print("\n2. Testing text extraction...")
        text = processor.extract_embedded_text(test_pdf)
        if text:
            print(f"   ✓ Extracted {len(text)} characters")
            lines = text.split("\n")[:5]
            for line in lines:
                if line.strip():
                    print(f"   - {line.strip()}")
        else:
            print("   ⚠ No text found")

        # Test page extraction
        print("\n3. Testing page extraction...")
        pages = processor.extract_pages(test_pdf)
        print(f"   ✓ Extracted {len(pages)} pages")
        for i, page in enumerate(pages):
            print(
                f"   - Page {i + 1}: {page.dimensions}, "
                f"{len(page.embedded_text_blocks)} text blocks"
            )

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


def test_config_validation():
    """Test configuration validation."""
    print("\n" + "=" * 70)
    print("TEST 2: Configuration Validation")
    print("=" * 70)

    test_cases = [
        ("Valid DPI (300)", {"dpi": 300}, True),
        ("Valid DPI (600)", {"dpi": 600}, True),
        ("Invalid DPI (too low)", {"dpi": 50}, False),
        ("Invalid DPI (too high)", {"dpi": 2000}, False),
        ("Invalid max_pages", {"max_pages": 0}, False),
        ("Invalid max_file_size", {"max_file_size_mb": -10}, False),
    ]

    for test_name, kwargs, should_pass in test_cases:
        try:
            config = PDFConfig(**kwargs)
            if should_pass:
                print(f"   ✓ {test_name}: Passed")
            else:
                print(f"   ❌ {test_name}: Should have failed but didn't")
        except ValueError as e:
            if not should_pass:
                print(f"   ✓ {test_name}: Correctly rejected")
            else:
                print(f"   ❌ {test_name}: Incorrectly rejected - {e}")


def test_error_handling():
    """Test error handling."""
    print("\n" + "=" * 70)
    print("TEST 3: Error Handling")
    print("=" * 70)

    processor = PDFProcessor(PDFConfig())

    # Test non-existent file
    print("\n1. Testing non-existent file...")
    try:
        processor.extract_pages("/nonexistent/file.pdf")
        print("   ❌ Should have raised error")
    except Exception as e:
        print(f"   ✓ Correctly raised: {type(e).__name__}")

    # Test invalid file
    print("\n2. Testing invalid file (not PDF)...")
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    invalid_file = test_dir / "not_a_pdf.txt"
    invalid_file.write_text("This is not a PDF")

    try:
        processor.extract_pages(invalid_file)
        print("   ❌ Should have raised error")
    except Exception as e:
        print(f"   ✓ Correctly raised: {type(e).__name__}")


def test_progress_callback():
    """Test progress callback functionality."""
    print("\n" + "=" * 70)
    print("TEST 4: Progress Callback")
    print("=" * 70)

    test_dir = Path(__file__).parent / "test_data"
    test_pdf = test_dir / "test_drawing.pdf"

    if not test_pdf.exists():
        print("⚠ Test PDF not found. Run test 1 first.")
        return

    processor = PDFProcessor(PDFConfig())

    progress_updates = []

    def track_progress(current, total):
        progress_updates.append((current, total))
        print(f"   Progress: {current}/{total} pages")

    try:
        pages = processor.extract_pages(test_pdf, progress_callback=track_progress)
        print(f"\n   ✓ Progress tracked {len(progress_updates)} times")
        print(f"   ✓ Extracted {len(pages)} pages")
    except Exception as e:
        print(f"   ❌ Test failed: {e}")


def test_config_overrides():
    """Test configuration override functionality."""
    print("\n" + "=" * 70)
    print("TEST 5: Configuration Overrides")
    print("=" * 70)

    config = PDFConfig(dpi=300, max_pages=10)
    processor = PDFProcessor(config)

    test_dir = Path(__file__).parent / "test_data"
    test_pdf = test_dir / "test_drawing.pdf"

    if not test_pdf.exists():
        print("⚠ Test PDF not found. Run test 1 first.")
        return

    try:
        # Extract with override
        print("\n1. Extracting with DPI override (600)...")
        pages = processor.extract_pages(test_pdf, dpi=600)
        print(f"   ✓ Extracted {len(pages)} pages with overridden DPI")

        # Verify original config unchanged
        print("\n2. Verifying original config unchanged...")
        assert processor.config.dpi == 300
        print(f"   ✓ Original DPI still: {processor.config.dpi}")

    except Exception as e:
        print(f"   ❌ Test failed: {e}")


def test_generator_extraction():
    """Test generator-based extraction for memory efficiency."""
    print("\n" + "=" * 70)
    print("TEST 6: Generator-based Extraction")
    print("=" * 70)

    test_dir = Path(__file__).parent / "test_data"
    test_pdf = test_dir / "test_drawing.pdf"

    if not test_pdf.exists():
        print("⚠ Test PDF not found. Run test 1 first.")
        return

    processor = PDFProcessor(PDFConfig())

    try:
        print("\n   Extracting pages using generator...")
        page_count = 0
        for page in processor.extract_pages_iter(test_pdf):
            page_count += 1
            print(f"   - Page {page_count}: {page.dimensions}")

        print(f"\n   ✓ Successfully iterated through {page_count} pages")

    except Exception as e:
        print(f"   ❌ Test failed: {e}")


def main():
    """Run all manual tests."""
    print("\n" + "=" * 70)
    print("PDF PROCESSOR MANUAL TEST SUITE")
    print("=" * 70)

    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Config Validation", test_config_validation),
        ("Error Handling", test_error_handling),
        ("Progress Callback", test_progress_callback),
        ("Config Overrides", test_config_overrides),
        ("Generator Extraction", test_generator_extraction),
    ]

    print(f"\nRunning {len(tests)} test suites...\n")

    for test_name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("TEST SUITE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
