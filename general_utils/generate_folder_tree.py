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
 * NEW: Optional --empty-only mode to show only empty files and folders.
"""

import os
import argparse  # Re-added for command-line flexibility
from pathlib import Path
from datetime import datetime
from typing import Iterator, List, Tuple

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


def _is_empty(path: Path) -> Tuple[bool, str]:
    """
    Checks if a path is considered empty and returns the emptiness status string.
    Returns: (is_empty: bool, metadata_string: str)
    """
    try:
        if path.is_file():
            size = path.stat().st_size
            return size == 0, "[EMPTY]" if size == 0 else ""

        # Folder logic
        # Check for children without iterating fully if possible (for performance)
        has_children = False
        empty_check_children = []
        for p in path.iterdir():
            if not p.name.startswith(".") or p.name == ".gitkeep":
                has_children = True
                empty_check_children.append(p)
                # Optimization: In 'empty-only' mode, we need to know the folder is empty *or* # that it contains at least one empty item. We'll rely on the main traversal
                # logic to figure out if it contains an empty item. Here, we only check for
                # absolute folder emptiness.
                break  # We only need to know if there's *any* non-hidden child

        if not has_children:
            return True, "[EMPTY FOLDER]"

        # If the path is a folder and has children, it's not absolutely empty.
        # The recursive call in _tree_generator will handle if its *only* empty items are shown.
        return False, ""

    except PermissionError:
        return False, "[ACCESS DENIED]"  # Cannot determine, but don't filter it out
    except FileNotFoundError:
        return False, "[FILE NOT FOUND]"
    except Exception:  # Catch any other os error
        return False, ""  # Treat as non-empty for filtering purposes


def _get_metadata(path: Path, show_sizes: bool) -> str:
    """Return status metadata (size / item count / empty markers)."""
    is_empty_state, metadata = _is_empty(path)

    # If it's a file and not empty, or a directory that's not absolutely empty, calculate size/count
    if not is_empty_state and path.is_dir() and show_sizes:
        try:
            # Count non-hidden children
            children = [
                p
                for p in path.iterdir()
                if not p.name.startswith(".") or p.name == ".gitkeep"
            ]
            count = len(children)
            return f"[{count} item{'s' if count != 1 else ''}]"
        except PermissionError:
            return "[ACCESS DENIED]"
        except Exception:
            return ""

    if path.is_file() and show_sizes:
        try:
            size = path.stat().st_size
            return f"[{_human_size(size)}]"
        except Exception:
            return ""

    # Return the simple [EMPTY] or [EMPTY FOLDER] if applicable, or ACCESS DENIED/etc.
    return metadata or ""


# --- Core Traversal Logic (Generator) ---


def _tree_generator(
    path: Path,
    prefix: str = "",
    is_last: bool = True,
    show_sizes: bool = True,
    empty_only: bool = False,  # NEW parameter
    target_column: int = TARGET_METADATA_COLUMN,
) -> Iterator[str]:
    """Recursively yield aligned tree lines."""

    # Check 1: Should we skip this specific item?
    is_current_empty, _ = _is_empty(path)

    if empty_only and path.is_file() and not is_current_empty:
        return  # Skip non-empty files in empty_only mode

    # If it's a non-empty folder, we still need to process it to see if it *contains* empty items.
    # The folder itself will be displayed if it is non-empty BUT it contains *any* item that is displayed (i.e. an empty file or empty folder).

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

    # Prepare children for recursion
    children_to_display: List[Path] = []
    children: List[Path] = []

    if path.is_dir():
        try:
            # Get and sort children: folders first, then files, all alphabetically
            children = [
                p
                for p in path.iterdir()
                if not p.name.startswith(".") or p.name == ".gitkeep"
            ]

            # Sort directories before files, then by name (case-insensitive)
            children.sort(key=lambda p: (p.is_file(), p.name.lower()))

            # Filter children based on 'empty_only' flag
            if empty_only:
                # In empty_only mode, we only keep children that are themselves empty files
                # or non-empty folders that might contain empty items (must be checked recursively)
                for child in children:
                    is_child_empty, _ = _is_empty(child)
                    if child.is_file():
                        if is_child_empty:
                            children_to_display.append(child)
                    else:  # Directory: always include it if we're in empty_only mode
                        # because we need to recurse to find empty files/folders inside.
                        children_to_display.append(child)
            else:
                children_to_display = children

        except PermissionError:
            # If access is denied, we still display the parent folder with the [ACCESS DENIED] metadata
            # and append an error line.
            metadata = "[ACCESS DENIED]"  # Override existing metadata
            yield f"{prefix}{connector}{display_name}    {metadata}\n"
            yield f"{prefix}     [ACCESS DENIED]\n"
            return
        except OSError as e:
            # Similar to PermissionError, display parent and error line.
            metadata = f"[Error reading: {e}]"  # Override existing metadata
            yield f"{prefix}{connector}{display_name}    {metadata}\n"
            yield f"{prefix}     [Error reading directory: {e}]\n"
            return

    # Now, check the current directory's line *again* before yielding it.
    # This is the crucial part for folders in 'empty_only' mode:
    # A folder is only yielded if it's **empty** (is_current_empty == True)
    # OR if it contains at least one **child** that will be yielded.

    # A list to store lines generated by the children's recursive calls.
    child_lines: List[str] = []

    # 3. Recurse for children
    if path.is_dir() and children_to_display:
        next_prefix = prefix + ("    " if is_last else "│   ")
        for idx, child in enumerate(children_to_display):
            is_child_last = idx == len(children_to_display) - 1

            # Recursively call the generator and collect its results
            # The children generator will only yield lines for empty items/folders that contain them.
            for line in _tree_generator(
                path=child,
                prefix=next_prefix,
                is_last=is_child_last,
                show_sizes=show_sizes,
                empty_only=empty_only,  # Pass the flag down
                target_column=target_column,  # Pass the target column down
            ):
                child_lines.append(line)

        # In empty_only mode, if the folder is NOT absolutely empty,
        # but no child lines were generated, it means the folder only contains non-empty items,
        # so we should skip the folder itself and its sub-tree.
        if empty_only and not is_current_empty and not child_lines:
            return

    # If we reach this point, we should display the current line
    # unless it was a non-empty folder in 'empty_only' mode with no empty contents.
    if empty_only:
        if path.is_dir() and not is_current_empty and not child_lines:
            return

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

    # Yield the collected child lines
    yield from child_lines

    # Note: If path.is_dir() was False, we returned earlier. If it was a directory,
    # child lines were already collected and yielded, and the recursion is done.


# --- Main Orchestration ---


def generate_clean_tree(
    project_dir: str,
    output_file: str = "tree_clean.txt",
    show_sizes: bool = True,
    empty_only: bool = False,  # NEW parameter
) -> None:
    """Generate pretty directory tree and save to file."""
    root_path = Path(project_dir).resolve()
    if not root_path.is_dir():
        raise ValueError(f"Directory not found: {project_dir}")

    # Check if the root itself should be shown (always yes, but its children are filtered)
    root_line_needed = True

    # Generate the header content
    mode_text = " (Empty Files/Folders Only)" if empty_only else ""
    header = (
        f"# Folder structure{mode_text}\n"
        f"# Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"# Root: {root_path}\n"
        f"# Folder: {root_path.name}/\n\n"
        f"{root_path.name}/\n"  # Print the root directory name once
    )

    lines: List[str] = [header]

    # Use the generator to build the main content lines
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
        for line in _tree_generator(
            path=child,
            prefix="",  # Prefix starts empty for children of the root
            is_last=is_last_child,
            show_sizes=show_sizes,
            empty_only=empty_only,  # Pass the flag
        ):
            lines.append(line)

    # If only the header and the root line exist, and we're in empty_only mode,
    # it means the root is not empty and contains no empty items, so we should filter it.
    if empty_only and len(lines) == 5:  # Only header lines + root line
        # Check if the root directory is absolutely empty
        is_root_empty, _ = _is_empty(root_path)
        if not is_root_empty:
            # If not empty and no child lines, clear the output.
            lines = [header]  # Keep only the header
            root_line_needed = False

    # Write to file
    out_path = Path(output_file)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Only write lines if there's more than just the header, or if the root is absolutely empty
        if root_line_needed or len(lines) > 5:
            out_path.write_text("".join(lines), encoding="utf-8")
        else:
            out_path.write_text(
                f"{header}No empty files or folders found.\n", encoding="utf-8"
            )
    except Exception as e:
        print(f"Error writing to file {out_path}: {e}")

    # Print to console
    print("".join(lines))
    if not root_line_needed and len(lines) <= 5:
        print("No empty files or folders found in the traversal.")

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

    # NEW Argument 4: Empty Only
    parser.add_argument(
        "-e",
        "--empty-only",
        action="store_true",
        dest="empty_only",
        help="Only display empty files ([EMPTY]) and empty folders ([EMPTY FOLDER]).",
    )

    args = parser.parse_args()

    try:
        generate_clean_tree(
            project_dir=args.directory,
            output_file=args.output,
            show_sizes=args.show_sizes,
            empty_only=args.empty_only,  # Pass the new argument
        )
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
