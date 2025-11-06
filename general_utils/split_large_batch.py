"""
Script to split large batch files into smaller chunks.

Usage:
    python split_large_batch.py batch_2_processing.txt

Creates: batch_2a.txt, batch_2b.txt, etc.
"""

import sys
from pathlib import Path


def split_batch_file(input_file, max_size_kb=80):
    """Split a batch file into smaller chunks based on file boundaries."""

    input_path = Path(input_file)

    if not input_path.exists():
        print(f"Error: {input_file} not found")
        return

    # Read the file
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by file markers
    file_marker = "\n" + "=" * 80 + "\n# FILE:"
    sections = content.split(file_marker)

    # First section is the header
    header = sections[0]
    file_sections = sections[1:]

    print(f"Found {len(file_sections)} files in {input_file}")

    # Group files into chunks under max_size_kb
    max_size_bytes = max_size_kb * 1024
    chunks = []
    current_chunk = []
    current_size = len(header.encode("utf-8"))

    for section in file_sections:
        section_with_marker = file_marker + section
        section_size = len(section_with_marker.encode("utf-8"))

        if current_size + section_size > max_size_bytes and current_chunk:
            # Save current chunk and start new one
            chunks.append(current_chunk)
            current_chunk = []
            current_size = len(header.encode("utf-8"))

        current_chunk.append(section_with_marker)
        current_size += section_size

    # Add last chunk
    if current_chunk:
        chunks.append(current_chunk)

    # Write chunk files
    base_name = input_path.stem

    for i, chunk in enumerate(chunks, 1):
        output_file = f"{base_name}_{chr(96+i)}.txt"  # _a, _b, _c, etc.

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(f"\n# PART {i} of {len(chunks)}\n")
            f.write("=" * 80 + "\n\n")
            f.writelines(chunk)

        file_size = Path(output_file).stat().st_size / 1024
        print(f"✓ Created {output_file} ({len(chunk)} files, {file_size:.1f} KB)")

    print(f"\n✓ Split {input_file} into {len(chunks)} parts")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_large_batch.py <batch_file.txt>")
        sys.exit(1)

    input_file = sys.argv[1]
    split_batch_file(input_file)
