"""
Script to concatenate Python files in logical batches for analysis.

Usage:
    python batch_concatenate.py

Creates multiple files: batch_1_models.txt, batch_2_processing.txt, etc.
"""

import os
from pathlib import Path


def get_file_content(py_file, root_path):
    """Get formatted content of a Python file."""
    try:
        rel_path = py_file.relative_to(root_path)

        lines = []
        lines.append("\n" + "=" * 80 + "\n")
        lines.append(f"# FILE: {rel_path}\n")
        lines.append("=" * 80 + "\n\n")

        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read()
            lines.append(content)

        lines.append("\n\n")
        return "".join(lines)
    except Exception as e:
        return f"# ERROR reading {py_file}: {str(e)}\n\n"


def concatenate_by_batches(root_dir="."):
    """Concatenate Python files in logical batches."""

    root_path = Path(root_dir).resolve()
    skip_dirs = {
        "__pycache__",
        ".git",
        "venv",
        "env",
        ".pytest_cache",
        "node_modules",
        ".egg-info",
    }

    # Define batches based on the project structure
    batches = {
        "batch_1_models": [
            "src/drawing_intelligence/models",
        ],
        "batch_2_processing": [
            "src/drawing_intelligence/processing",
        ],
        "batch_3_orchestration": [
            "src/drawing_intelligence/orchestration",
        ],
        "batch_4_database": [
            "src/drawing_intelligence/database",
        ],
        "batch_5_llm": [
            "src/drawing_intelligence/llm",
        ],
        "batch_6_export": [
            "src/drawing_intelligence/export",
        ],
        "batch_7_quality": [
            "src/drawing_intelligence/quality",
        ],
        "batch_8_utils": [
            "src/drawing_intelligence/utils",
        ],
        "batch_9_cli": [
            "src/drawing_intelligence/cli",
        ],
    }

    print("Creating batched files...\n")

    for batch_name, directories in batches.items():
        files_content = []
        file_count = 0

        for directory in directories:
            dir_path = root_path / directory

            if not dir_path.exists():
                continue

            # Find all Python files in this directory
            for py_file in dir_path.rglob("*.py"):
                # Skip if in excluded directory
                if any(skip_dir in py_file.parts for skip_dir in skip_dirs):
                    continue

                files_content.append(get_file_content(py_file, root_path))
                file_count += 1

        if file_count > 0:
            # Write batch file
            output_file = f"{batch_name}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"# Batch: {batch_name}\n")
                f.write(f"# Total files: {file_count}\n")
                f.write("=" * 80 + "\n\n")
                f.writelines(files_content)

            file_size = Path(output_file).stat().st_size / 1024  # Size in KB
            print(f"✓ Created {output_file} ({file_count} files, {file_size:.1f} KB)")

    print("\n✓ All batches created successfully!")
    print("\nRecommended order to paste:")
    print("1. batch_1_models.txt (data structures)")
    print("2. batch_2_processing.txt (core processing)")
    print("3. batch_3_orchestration.txt (workflow)")
    print("4. batch_4_database.txt (database)")
    print("5. batch_5_llm.txt (LLM integration)")
    print("6. batch_6_export.txt (export)")
    print("7. batch_7_quality.txt (quality)")
    print("8. batch_8_utils.txt (utilities)")
    print("9. batch_9_cli.txt (CLI)")


if __name__ == "__main__":
    import sys

    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    concatenate_by_batches(root_dir)
