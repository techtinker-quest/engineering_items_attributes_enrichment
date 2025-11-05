#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
tree_generator.py
-----------------
Generate a clean, aligned, and visually neat directory tree, with command-line options.

Features:
 * Folder structure with aligned branches (├──, └──, │)
 * Optional display of size or item count
 * Marks [EMPTY], [EMPTY FOLDER], [ACCESS DENIED]
 * Skips hidden files except `.gitkeep`
 * Saves output to text file + prints to console
"""

import os
import argparse  # Re-added for command-line flexibility
from pathlib import Path
from datetime import datetime
from typing import Iterator, List

# --- Configuration Constant ---
# The column index where the metadata (size/count) should ideally start.
# This ensures alignment for the metadata on the right.
TARGET_METADATA_COLUMN = 60


# --- Helper Functions ---


def _human_size(num: int) -> str:
    """Return a human-readable size for file display."""
    if num == 0:
        return "[EMPTY]"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024:
            return f"{num:.1f} {unit}".replace(".0", "")
        num /= 1024
    return f"{num:.1f} PB"  # For extremely large files


def _get_metadata(path: Path, show_sizes: bool) -> str:
    """Return status metadata (size / item count / empty markers)."""
    try:
        if path.is_file():
            size = path.stat().st_size
            if not show_sizes:
                return "[EMPTY]" if size == 0 else ""
            return f"[{_human_size(size)}]"

        # Folder logic
        children = [
            p
            for p in path.iterdir()
            if not p.name.startswith(".") or p.name == ".gitkeep"
        ]

        if not children:
            return "[EMPTY FOLDER]"

        if show_sizes:
            count = len(children)
            return f"[{count} item{'s' if count != 1 else ''}]"

        return ""

    except FileNotFoundError:
        # Should not happen in traversal but good for robustness
        return "[FILE NOT FOUND]"
    except PermissionError:
        return "[ACCESS DENIED]"


# --- Core Traversal Logic (Generator) ---


def _tree_generator(
    path: Path,
    prefix: str = "",
    is_last: bool = True,
    show_sizes: bool = True,
    target_column: int = TARGET_METADATA_COLUMN,
) -> Iterator[str]:
    """Recursively yield aligned tree lines."""

    # 1. Prepare line components
    connector = "└── " if is_last else "├── "

    # Root path is special: it doesn't need a connector or prefix on its line
    if prefix == "":
        display_name = path.name
    else:
        display_name = path.name

    if path.is_dir():
        display_name += "/"

    metadata = _get_metadata(path, show_sizes)

    # 2. Format and Yield the current line
    line = f"{prefix}{connector}{display_name}"

    if metadata:
        # Configuration for step alignment
        MIN_SEPARATION = 6
        STEP_SIZE = 6

        current_length = len(line)

        # Calculate the earliest column the metadata can start while maintaining MIN_SEPARATION
        P_min_start = current_length + MIN_SEPARATION

        # Desired start columns are T, T+S, T+2S, ... (60, 66, 72, ...)
        T = target_column
        S = STEP_SIZE

        # 1. Start checking from the target column (T)
        P_start = T

        # 2. Iterate in steps of S until P_start is far enough away from the file name.
        # This guarantees alignment to multiples of 6 (starting at 60) AND minimum separation.
        while P_start < P_min_start:
            P_start += S

        # 3. Calculate the padding based on the chosen start column
        padding_length = P_start - current_length
        padding = " " * padding_length

        line += f"{padding}{metadata}"

    # Yield the line for the current file/folder
    yield line + "\n"

    # 3. Recurse for children (if it's a directory)
    if not path.is_dir():
        return

    try:
        # Get and sort children: folders first, then files, all alphabetically
        children = [
            p
            for p in path.iterdir()
            if not p.name.startswith(".") or p.name == ".gitkeep"
        ]

        # Sort directories before files, then by name (case-insensitive)
        children.sort(key=lambda p: (p.is_file(), p.name.lower()))

    except PermissionError:
        yield f"{prefix}    [ACCESS DENIED]\n"
        return
    except OSError as e:
        yield f"{prefix}    [Error reading directory: {e}]\n"
        return

    # Determine the prefix extension for the next level
    next_prefix = prefix + ("    " if is_last else "│   ")

    for idx, child in enumerate(children):
        is_child_last = idx == len(children) - 1

        # Recursively call the generator and yield its results
        yield from _tree_generator(
            path=child,
            prefix=next_prefix,
            is_last=is_child_last,
            show_sizes=show_sizes,
            target_column=target_column,  # Pass the target column down
        )


# --- Main Orchestration ---


def generate_clean_tree(
    project_dir: str, output_file: str = "tree_clean.txt", show_sizes: bool = True
) -> None:
    """Generate pretty directory tree and save to file."""
    root_path = Path(project_dir).resolve()
    if not root_path.is_dir():
        raise ValueError(f"Directory not found: {project_dir}")

    # Generate the header content
    header = (
        f"# Folder structure\n"
        f"# Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"# Root: {root_path}\n"
        f"# Folder: {root_path.name}/\n\n"
        f"{root_path.name}/\n"  # Print the root directory name once
    )

    lines: List[str] = [header]

    # Use the generator to build the main content lines
    # Start the generator from the children of the root path to handle the top-level correctly.
    try:
        root_children = [
            p
            for p in root_path.iterdir()
            if not p.name.startswith(".") or p.name == ".gitkeep"
        ]
        root_children.sort(key=lambda p: (p.is_file(), p.name.lower()))
    except Exception as e:
        lines.append(f"Error reading root directory: {e}\n")
        root_children = []

    for idx, child in enumerate(root_children):
        is_last_child = idx == len(root_children) - 1

        # Generator for the root's children, starting with an empty prefix
        # We manually handle the root itself in the header and the line above.
        for line in _tree_generator(
            path=child,
            prefix="",  # Prefix starts empty for children of the root
            is_last=is_last_child,
            show_sizes=show_sizes,
        ):
            lines.append(line)

    # Write to file
    out_path = Path(output_file)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("".join(lines), encoding="utf-8")
    except Exception as e:
        print(f"Error writing to file {out_path}: {e}")

    print("".join(lines))
    print(f"\nTree saved to: {out_path.resolve()}")


# --- Command Line Interface / Direct Execution ---


def main():
    """
    Main function. Parses command-line arguments, using defaults for direct execution.
    Defaults match the hardcoded values: dir='.', output='tree_clean_pretty.txt', show_sizes=True.
    """
    parser = argparse.ArgumentParser(
        description="Generate a clean, aligned, and visually neat directory tree."
    )

    # Argument 1: Directory (Defaults to current directory '.')
    parser.add_argument(
        "directory",
        type=str,
        nargs="?",
        default=".",
        help="The root directory to analyze (default: current directory).",
    )

    # Argument 2: Output File (Defaults to 'tree_clean_pretty.txt')
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="tree_clean_pretty.txt",
        help="Output filename for the tree (default: tree_clean_pretty.txt).",
    )

    # Argument 3: Show Sizes (Defaults to True, use --no-sizes to set to False)
    parser.add_argument(
        "-s",
        "--no-sizes",
        action="store_false",
        dest="show_sizes",
        help="Do not display file sizes or item counts in folders.",
    )

    args = parser.parse_args()

    try:
        generate_clean_tree(
            project_dir=args.directory,
            output_file=args.output,
            show_sizes=args.show_sizes,
        )
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
