"""
Validate system installation and dependencies.
"""

import sys
import importlib
from pathlib import Path

REQUIRED_PACKAGES = [
    "numpy",
    "opencv-python",
    "pillow",
    "PyMuPDF",
    "paddleocr",
    "easyocr",
    "ultralytics",
    "anthropic",
    "openai",
    "google-generativeai",
    "pyyaml",
    "jinja2",
]


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9+ required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check required packages."""
    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing.append(package)
    return len(missing) == 0, missing


def check_structure():
    """Check directory structure."""
    required_dirs = ["config", "data", "src", "tests", "logs"]
    missing = []
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ (missing)")
            missing.append(dir_name)
    return len(missing) == 0, missing


def main():
    print("=" * 60)
    print("DRAWING INTELLIGENCE SYSTEM - INSTALLATION VALIDATION")
    print("=" * 60)

    all_valid = True

    print("\n📌 Checking Python version...")
    if not check_python_version():
        all_valid = False

    print("\n📦 Checking dependencies...")
    deps_valid, missing_deps = check_dependencies()
    if not deps_valid:
        all_valid = False
        print(f"\n⚠️  Install missing packages: pip install {' '.join(missing_deps)}")

    print("\n📁 Checking directory structure...")
    struct_valid, missing_dirs = check_structure()
    if not struct_valid:
        all_valid = False

    print("\n" + "=" * 60)
    if all_valid:
        print("✅ Installation is VALID - Ready to use!")
    else:
        print("❌ Installation has issues - Please fix and rerun")
    print("=" * 60)

    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
