"""
Environment Setup Script

Sets up the complete environment for the Drawing Intelligence System.
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil


def print_header(message: str):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(message)
    print("=" * 60 + "\n")


def check_python_version():
    """Check Python version."""
    print_header("Checking Python Version")

    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9+ required")
        return False

    print("✅ Python version OK")
    return True


def create_directories():
    """Create required directories."""
    print_header("Creating Directories")

    directories = [
        "data",
        "data/checkpoints",
        "models",
        "output",
        "logs",
        "temp",
        "config",
        "config/templates",
        "config/prompts",
    ]

    for dir_path in directories:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {dir_path}")
        else:
            print(f"   Exists: {dir_path}")


def install_dependencies():
    """Install Python dependencies."""
    print_header("Installing Dependencies")

    print("Installing core dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "deployment/requirements.txt"]
    )

    # Ask about dev dependencies
    response = input("\nInstall development dependencies? (y/n): ")
    if response.lower() == "y":
        print("Installing development dependencies...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                "deployment/requirements-dev.txt",
            ]
        )


def setup_configuration():
    """Setup configuration files."""
    print_header("Setting Up Configuration")

    config_file = Path("config/system_config.yaml")

    if config_file.exists():
        print(f"⚠️  Configuration file already exists: {config_file}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != "y":
            print("Skipping configuration setup")
            return

    # Create default configuration
    default_config = """# Drawing Intelligence System Configuration

paths:
  data_dir: "data"
  models_dir: "models"
  output_dir: "output"
  temp_dir: "temp"
  log_dir: "logs"

database:
  path: "data/drawings.db"
  backup_enabled: true
  backup_frequency_hours: 24
  enable_wal_mode: true

pdf_processing:
  dpi: 300
  max_file_size_mb: 50
  max_pages: 20
  convert_to_grayscale: true

image_preprocessing:
  skew_threshold_degrees: 0.5
  apply_clahe: true

ocr:
  primary_engine: 'paddleocr'
  fallback_engine: 'easyocr'
  confidence_threshold: 0.85
  languages: ['en']

entity_extraction:
  use_regex: true
  use_spacy: false
  use_llm: false
  confidence_threshold: 0.80
  normalize_units: true

shape_detection:
  model_path: "models/yolov8s_engineering_v1.0/weights/best.pt"
  confidence_threshold: 0.45
  nms_threshold: 0.45
  device: 'cuda'  # or 'cpu'
  batch_size: 8

data_association:
  label_distance_threshold: 200
  dimension_distance_threshold: 500
  min_association_confidence: 0.6
  enable_obstacle_detection: true

validation:
  enforce_part_number: true
  warn_on_orphaned_entities: true
  check_dimension_compatibility: true

quality_scoring:
  review_threshold: 0.75
  ocr_weight: 0.30
  detection_weight: 0.40
  entity_weight: 0.30
  flag_missing_critical_entities: true
  critical_entities: ['PART_NUMBER']

llm_integration:
  enabled: false  # Set to true when API keys configured
  primary_provider:
    name: 'anthropic'
    model: 'claude-3-sonnet-20240229'
    api_key_env: 'ANTHROPIC_API_KEY'
  fallback_provider:
    name: 'openai'
    model: 'gpt-4-turbo-2024-04-09'
    api_key_env: 'OPENAI_API_KEY'
  timeout_seconds: 30
  max_retries: 3
  cost_controls:
    daily_budget_usd: 50.00
    per_drawing_limit_usd: 0.30
    alert_threshold_usd: 40.00
    auto_disable_on_exceed: true

batch_processing:
  parallel_workers: 4
  checkpoint_frequency: 10
  continue_on_error: true
  max_retries_per_drawing: 3
  batch_checkpoint_dir: "data/checkpoints"

export:
  json_format: "pretty"
  csv_delimiter: ","
  include_intermediate_results: false
  include_images: false

logging:
  level: "INFO"
  log_dir: "logs"
  rotate_size_mb: 100
  rotate_count: 10
  log_to_console: true
"""

    with open(config_file, "w") as f:
        f.write(default_config)

    print(f"✅ Created configuration file: {config_file}")


def setup_environment_variables():
    """Guide user through environment variable setup."""
    print_header("Environment Variables Setup")

    print("The following environment variables can be configured:\n")

    env_vars = {
        "ANTHROPIC_API_KEY": "Anthropic (Claude) API key",
        "OPENAI_API_KEY": "OpenAI (GPT) API key",
        "GOOGLE_API_KEY": "Google (Gemini) API key (optional)",
    }

    for var, description in env_vars.items():
        current = os.getenv(var)
        if current:
            print(f"✅ {var}: Already set")
        else:
            print(f"⚠️  {var}: Not set ({description})")

    print("\nTo set environment variables:")
    print("  Linux/Mac: export VARIABLE_NAME='your-key-here'")
    print("  Windows: set VARIABLE_NAME=your-key-here")
    print("\nOr add them to your .env file")


def download_models_prompt():
    """Prompt user to download models."""
    print_header("Model Download")

    print("YOLO models need to be downloaded separately.")
    print("\nOptions:")
    print("  1. Download pre-trained models")
    print("  2. Train your own models")
    print("  3. Skip (download later)")

    response = input("\nChoice (1-3): ")

    if response == "1":
        print("\nRun: python scripts/download_models.py")
    elif response == "2":
        print("\nRefer to training documentation")
    else:
        print("\nSkipping model download")


def run_tests():
    """Run test suite."""
    print_header("Running Tests")

    response = input("Run test suite? (y/n): ")
    if response.lower() != "y":
        print("Skipping tests")
        return

    print("\nRunning unit tests...")
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/unit", "-v"])

    if result.returncode == 0:
        print("✅ Unit tests passed")
    else:
        print("⚠️  Some unit tests failed")


def print_summary():
    """Print setup summary."""
    print_header("Setup Complete!")

    print("Next steps:")
    print("\n1. Configure API keys (if using LLM features):")
    print("   export ANTHROPIC_API_KEY='your-key'")
    print("   export OPENAI_API_KEY='your-key'")

    print("\n2. Download YOLO models:")
    print("   python scripts/download_models.py")

    print("\n3. Validate installation:")
    print("   python scripts/validate_installation.py")

    print("\n4. Process your first drawing:")
    print("   drawing-intelligence process your_drawing.pdf")

    print("\n5. Or process a batch:")
    print("   drawing-intelligence batch drawings_folder/")

    print("\nDocumentation: README.md")
    print("Configuration: config/system_config.yaml")


def main():
    """Run complete setup."""
    print("=" * 60)
    print("DRAWING INTELLIGENCE SYSTEM - SETUP")
    print("=" * 60)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Create directories
    create_directories()

    # Install dependencies
    install_dependencies()

    # Setup configuration
    setup_configuration()

    # Environment variables
    setup_environment_variables()

    # Download models
    download_models_prompt()

    # Run tests
    run_tests()

    # Summary
    print_summary()


if __name__ == "__main__":
    main()
