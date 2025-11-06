"""
Script to concatenate all Python files from the project into a single file
for analysis purposes.

Usage:
    python concatenate_files.py > all_code.txt

Then paste the contents of all_code.txt into the chat.
"""

import os
from pathlib import Path


def concatenate_python_files(root_dir=".", output_file=None):
    """
    Concatenate all .py files in the project with clear delimiters.

    Args:
        root_dir: Root directory to start searching (default: current directory)
        output_file: If provided, write to file instead of stdout
    """

    # Get the absolute path of root directory
    root_path = Path(root_dir).resolve()

    # Directories to skip
    skip_dirs = {
        "__pycache__",
        ".git",
        "venv",
        "env",
        ".pytest_cache",
        "node_modules",
        ".egg-info",
    }

    # Collect all Python files
    python_files = []

    for py_file in root_path.rglob("*.py"):
        # Skip if in any excluded directory
        if any(skip_dir in py_file.parts for skip_dir in skip_dirs):
            continue
        python_files.append(py_file)

    # Sort files for consistent ordering
    python_files.sort()

    output_lines = []
    output_lines.append(f"# Total Python files found: {len(python_files)}\n")
    output_lines.append("=" * 80 + "\n\n")

    # Process each file
    for py_file in python_files:
        try:
            # Get relative path from root
            rel_path = py_file.relative_to(root_path)

            # Add file header
            output_lines.append("\n" + "=" * 80 + "\n")
            output_lines.append(f"# FILE: {rel_path}\n")
            output_lines.append("=" * 80 + "\n\n")

            # Read and add file content
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
                output_lines.append(content)

            output_lines.append("\n\n")

        except Exception as e:
            output_lines.append(f"# ERROR reading {rel_path}: {str(e)}\n\n")

    # Write output
    final_output = "".join(output_lines)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_output)
        print(f"Concatenated {len(python_files)} files to {output_file}")
    else:
        print(final_output)


if __name__ == "__main__":
    import sys

    # Check if root directory is provided as argument
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    # Write directly to file to avoid Windows console encoding issues
    output_filename = "concatenated_code.txt"
    concatenate_python_files(root_dir, output_file=output_filename)
    print(f"\n✓ Successfully created {output_filename}")
