import argparse
import os
import requests
import ast
import json
import csv  # Still needed internally for DictWriter functionality in TXT mode
import time  # <-- NEW: Added for timing measurements
from urllib.parse import urlparse
from datetime import datetime
from pathlib import Path
import logging
import sys

# --- Configuration and Constants ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Environment Variable for GitHub PAT
GITHUB_PAT = os.environ.get("GITHUB_PAT")
# Default API settings
GITHUB_API_URL = "https://api.github.com"
API_TIMEOUT_SECONDS = 15
DEFAULT_BRANCH = "main"

# Safe default delimiter for TXT output (single character required by csv.DictWriter)
TXT_DELIMITER = "|"

# PROGRESS CONSTANT
PROGRESS_UPDATE_INTERVAL_SECONDS = 5  # <-- NEW: Time interval for progress logging

# CSV/TXT fieldnames (used as keys for JSON output as well)
CSV_FIELDNAMES = [
    "Source_Type",
    "File_Path",
    "File_Type",
    "Size_Bytes",
    "SHA",
    "Last_Commit_Date_ISO",
    "Last_Committer",
    "HTML_URL",
    "Raw_Content_URL",
    "Classes",
    "Functions",
    "Methods",
    "Notes",
]

# Supported code file types for AST analysis
CODE_FILE_TYPES = {".py"}

# --- AST (Code Introspection) Logic (Remains the same) ---


def analyze_python_file(file_content: str) -> tuple[str, str, str, str]:
    """
    Analyzes Python code content using ast.NodeVisitor to extract classes,
    module-level functions, and class methods.
    """
    classes, functions, methods = [], [], []
    current_class = None

    class Analyzer(ast.NodeVisitor):
        def visit_ClassDef(self, node):
            nonlocal current_class
            classes.append(node.name)
            old_class = current_class
            current_class = node.name
            self.generic_visit(node)
            current_class = old_class

        def visit_FunctionDef(self, node):
            if current_class:
                methods.append(f"{current_class}::{node.name}")
            else:
                functions.append(node.name)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            self.visit_FunctionDef(node)

    try:
        if not file_content.strip():
            return "", "", "", "AST Warning: File content is empty or only whitespace"

        tree = ast.parse(file_content)
        Analyzer().visit(tree)

        return (", ".join(classes), ", ".join(functions), ", ".join(methods), "")
    except SyntaxError:
        return "", "", "", "AST Error: Python SyntaxError"
    except Exception as e:
        return "", "", "", f"AST Error: {type(e).__name__} ({e})"


# --- GitHub API Mode Logic ---


def get_api_headers(pat: str | None) -> dict[str, str]:
    headers = {"Accept": "application/vnd.github.com+json"}
    if pat:
        headers["Authorization"] = f"token {pat}"
    return headers


def fetch_file_commit_details(
    owner: str, repo: str, file_path: str, pat: str | None
) -> tuple[str, str]:
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/commits"
    headers = get_api_headers(pat)
    try:
        params = {"path": file_path, "per_page": 1}
        response = requests.get(
            url, headers=headers, params=params, timeout=API_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        commits = response.json()
        if commits:
            commit = commits[0]
            date = commit["commit"]["author"]["date"]
            committer = (
                commit.get("committer", {}).get("login")
                or commit["commit"]["author"]["name"]
            )
            return date, committer
        return "N/A", "N/A"
    except requests.exceptions.HTTPError as e:
        return "API Error", f"API Error (HTTP {e.response.status_code})"
    except Exception as e:
        return "API Error", f"API Error: {e.__class__.__name__}"


def traverse_github_repo(
    url: str, branch: str, pat: str | None, exclude_types: list[str], target_dir: str
) -> list[dict]:
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.strip("/").split("/")
    if len(path_parts) < 2:
        raise ValueError(
            "Invalid GitHub URL format. Expected: https://github.com/owner/repo"
        )
    owner, repo = path_parts[0], path_parts[1]
    headers = get_api_headers(pat)
    logging.info(
        f"[{datetime.now().strftime('%H:%M:%S')}] Starting GitHub API traversal for {owner}/{repo}..."
    )

    ref_url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/branches/{branch}"
    sha = branch
    try:
        ref_response = requests.get(
            ref_url, headers=headers, timeout=API_TIMEOUT_SECONDS
        )
        ref_response.raise_for_status()
        sha = ref_response.json()["commit"]["sha"]
        logging.info(f"Target Branch '{branch}' resolved to SHA: {sha[:7]}")
    except Exception as e:
        logging.error(
            f"Error resolving branch/SHA: {e.__class__.__name__}. Using provided branch/sha directly: {branch}."
        )

    tree_url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/git/trees/{sha}?recursive=1"

    try:
        tree_response = requests.get(
            tree_url, headers=headers, timeout=API_TIMEOUT_SECONDS * 2
        )
        tree_response.raise_for_status()
        tree_data = tree_response.json().get("tree", [])
    except Exception as e:
        logging.error(f"FATAL API Error fetching repository tree: {e}")
        return []

    # --- PROGRESS ANNOUNCEMENT AND SETUP (GitHub) ---
    total_items = len(tree_data)
    logging.info("-" * 50)
    logging.info(
        f"📁 Total items found in Git Tree: **{total_items}** files/folders to process."
    )
    logging.info("-" * 50)

    results = []
    last_log_count = 0
    last_log_time = time.time()
    # -------------------------------------------------

    target_dir_normalized = target_dir.strip("/")
    if target_dir_normalized:
        target_dir_normalized += "/"

    for item in tree_data:
        path = item["path"]
        item_type = item["type"]

        if target_dir_normalized and not path.startswith(target_dir_normalized):
            continue

        if item_type == "tree":
            results.append(
                {
                    "Source_Type": "GitHub",
                    "File_Path": path,
                    "File_Type": "directory",
                    "Size_Bytes": "N/A",
                    "SHA": item["sha"],
                    "Notes": "Directory",
                }
            )

        elif item_type == "blob":
            extension = Path(path).suffix.lower().lstrip(".")

            if extension in exclude_types:
                results.append(
                    {
                        "Source_Type": "GitHub",
                        "File_Path": path,
                        "File_Type": extension,
                        "Size_Bytes": item["size"],
                        "SHA": item["sha"],
                        "Notes": "Skipped: Excluded Type",
                    }
                )
            else:
                commit_date, committer = fetch_file_commit_details(
                    owner, repo, path, pat
                )
                html_url = f"https://github.com/{owner}/{repo}/blob/{sha}/{path}"
                raw_url = (
                    f"https://raw.githubusercontent.com/{owner}/{repo}/{sha}/{path}"
                )
                classes, functions, methods, ast_notes = "", "", "", ""

                if Path(path).suffix.lower() in CODE_FILE_TYPES:
                    try:
                        content_response = requests.get(
                            raw_url, headers=headers, timeout=API_TIMEOUT_SECONDS
                        )
                        content_response.raise_for_status()
                        classes, functions, methods, ast_notes = analyze_python_file(
                            content_response.text
                        )
                    except Exception:
                        ast_notes = (
                            "AST failed (raw fetch error, timeout, or large file)"
                        )

                results.append(
                    {
                        "Source_Type": "GitHub",
                        "File_Path": path,
                        "File_Type": extension,
                        "Size_Bytes": item["size"],
                        "SHA": item["sha"],
                        "Last_Commit_Date_ISO": commit_date,
                        "Last_Committer": committer,
                        "HTML_URL": html_url,
                        "Raw_Content_URL": raw_url,
                        "Classes": classes,
                        "Functions": functions,
                        "Methods": methods,
                        "Notes": ast_notes,
                    }
                )

        # --- CONDITIONAL LOGGING CHECK (GitHub) ---
        current_time = time.time()
        current_count = len(results)

        if current_time - last_log_time >= PROGRESS_UPDATE_INTERVAL_SECONDS:
            if current_count > last_log_count:
                # Log the progress and reset the trackers
                logging.info(
                    f"⏳ Progress: Processed **{current_count}** of {total_items} items."
                )
                last_log_count = current_count

            # Reset the time tracker regardless of whether a message was sent
            last_log_time = current_time
        # ------------------------------------------

    return results


# --- Local File System Mode Logic ---


def traverse_local_dir(root_path: str, exclude_types: list[str]) -> list[dict]:

    # --- Local Traversal requires pre-counting for total items ---
    root_path_obj = Path(root_path).resolve()
    if not root_path_obj.is_dir():
        raise FileNotFoundError(f"Local path is not a valid directory: {root_path}")

    logging.info(
        f"[{datetime.now().strftime('%H:%M:%S')}] Starting Local File System traversal for {root_path_obj}..."
    )

    # Pre-count all items to set user expectation
    all_entries = list(root_path_obj.rglob("*"))
    total_items = len(all_entries)

    # --- PROGRESS ANNOUNCEMENT AND SETUP (Local) ---
    logging.info("-" * 50)
    logging.info(
        f"📁 Total items found locally: **{total_items}** files/folders to process."
    )
    logging.info("-" * 50)

    results = []
    last_log_count = 0
    last_log_time = time.time()
    # -------------------------------------------------

    for entry in all_entries:  # Iterate over the pre-counted list
        path = str(entry.relative_to(root_path_obj))

        if entry.is_dir():
            results.append(
                {
                    "Source_Type": "Local",
                    "File_Path": path,
                    "File_Type": "directory",
                    "Size_Bytes": "N/A",
                    "Notes": "Directory",
                }
            )

        elif entry.is_file():
            extension = entry.suffix.lower().lstrip(".")

            if extension in exclude_types:
                results.append(
                    {
                        "Source_Type": "Local",
                        "File_Path": path,
                        "File_Type": extension,
                        "Size_Bytes": entry.stat().st_size,
                        "Notes": "Skipped: Excluded Type",
                    }
                )
            else:
                try:
                    stat_result = entry.stat()
                    size = stat_result.st_size
                    last_mod_ts = datetime.fromtimestamp(
                        stat_result.st_mtime
                    ).isoformat()
                except OSError as e:
                    logging.warning(
                        f"Skipping file due to access/stat error: {path} ({e.__class__.__name__})"
                    )
                    continue

                classes, functions, methods, ast_notes = "", "", "", ""
                if entry.suffix.lower() in CODE_FILE_TYPES:
                    file_content = ""
                    try:
                        file_content = entry.read_text(
                            encoding="utf-8", errors="strict"
                        )
                        classes, functions, methods, ast_notes = analyze_python_file(
                            file_content
                        )
                    except UnicodeDecodeError:
                        try:
                            file_content = entry.read_text(
                                encoding="latin-1", errors="strict"
                            )
                            classes, functions, methods, ast_notes = (
                                analyze_python_file(file_content)
                            )
                        except Exception as e:
                            ast_notes = f"AST Error: Encoding is neither UTF-8 nor Latin-1 ({e.__class__.__name__})"
                    except Exception as e:
                        ast_notes = f"AST Error: General Read/Parse Error ({e.__class__.__name__})"

                results.append(
                    {
                        "Source_Type": "Local",
                        "File_Path": path,
                        "File_Type": extension,
                        "Size_Bytes": size,
                        "SHA": "N/A",
                        "Last_Commit_Date_ISO": last_mod_ts,
                        "Last_Committer": "N/A",
                        "HTML_URL": "N/A",
                        "Raw_Content_URL": "N/A",
                        "Classes": classes,
                        "Functions": functions,
                        "Methods": methods,
                        "Notes": ast_notes,
                    }
                )

        # --- CONDITIONAL LOGGING CHECK (Local) ---
        current_time = time.time()
        current_count = len(results)

        if current_time - last_log_time >= PROGRESS_UPDATE_INTERVAL_SECONDS:
            if current_count > last_log_count:
                # Log the progress and reset the trackers
                logging.info(
                    f"⏳ Progress: Processed **{current_count}** of {total_items} items."
                )
                last_log_count = current_count

            # Reset the time tracker regardless of whether a message was sent
            last_log_time = current_time
        # ------------------------------------------

    return results


# --- Output Writing Logic (Remains the same) ---


def write_structured_output(
    scan_results: list[dict], output_file: str, file_format: str
):
    """Writes results to the specified format (txt or json)."""

    if file_format == "json":
        output_file_name = f"{output_file.rsplit('.', 1)[0]}.json"
        try:
            with open(output_file_name, "w", encoding="utf-8") as f:
                json.dump(scan_results, f, indent=4)
            logging.info(
                f"✅ Successfully wrote {len(scan_results)} entries to JSON file: **{output_file_name}**."
            )
            return

        except Exception as e:
            raise IOError(f"Error writing JSON output: {e}")

    # Handle TXT (using csv.DictWriter with the safe pipe delimiter)
    output_file_name = f"{output_file.rsplit('.', 1)[0]}.txt"
    delimiter = TXT_DELIMITER

    try:
        with open(output_file_name, "w", newline="", encoding="utf-8") as f:
            # We use DictWriter here because it handles fieldnames, quotes, and newline formatting reliably.
            writer = csv.DictWriter(
                f,
                fieldnames=CSV_FIELDNAMES,
                delimiter=delimiter,
                extrasaction="ignore",
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writeheader()
            writer.writerows(scan_results)

        logging.info(
            f"✅ Successfully wrote {len(scan_results)} entries to **{output_file_name}** "
            f"using delimiter '{delimiter}'."
        )
    except Exception as e:
        raise IOError(f"Error writing TXT output with delimiter '{delimiter}': {e}")


# --- Main Execution ---


def main():
    """Parses command-line arguments and runs the appropriate traversal/output logic."""
    global GITHUB_PAT

    parser = argparse.ArgumentParser(
        description="Universal Repository File Link and Code Analyzer.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--source-type",
        required=True,
        choices=["github", "local"],
        help="The source type.",
    )
    parser.add_argument(
        "--path", required=True, help="The source path: GitHub URL or local directory."
    )
    parser.add_argument(
        "--output",
        default=f"repo_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Base name for the output file.",
    )
    # Removed 'csv' choice
    parser.add_argument(
        "--format",
        default="txt",
        choices=["txt", "json"],
        help="Output format: 'txt' (pipe-separated) or 'json'.",
    )
    # Removed --delimiter argument as it is no longer necessary (TXT_DELIMITER is fixed)
    parser.add_argument(
        "--pat", default=None, help="Optional GitHub Personal Access Token (PAT)."
    )
    parser.add_argument(
        "--branch",
        default=DEFAULT_BRANCH,
        help=f"Target branch/commit SHA (GitHub mode only). Defaults to '{DEFAULT_BRANCH}'.",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated list of file extensions to exclude.",
    )
    parser.add_argument(
        "--target-dir",
        default="",
        help="Relative path to start the scan from (GitHub mode only).",
    )

    args = parser.parse_args()

    if args.pat:
        GITHUB_PAT = args.pat

    exclude_list = [
        ext.strip().lower() for ext in args.exclude.split(",") if ext.strip()
    ]

    logging.info("-" * 50)
    # Simplified logging
    logging.info(
        f"Mode: **{args.source_type.upper()}** | Output Format: **{args.format.upper()}** | TXT Delimiter: '{TXT_DELIMITER}'"
    )
    logging.info("-" * 50)

    try:
        # --- TIMER START ---
        start_time = time.time()
        traversal_start = time.time()
        # -------------------

        if args.source_type == "github":
            if not GITHUB_PAT:
                logging.warning("⚠️ Warning: No PAT provided. Running unauthenticated.")
            scan_results = traverse_github_repo(
                args.path, args.branch, GITHUB_PAT, exclude_list, args.target_dir
            )
        elif args.source_type == "local":
            scan_results = traverse_local_dir(args.path, exclude_list)
        else:
            scan_results = []

        traversal_end = time.time()
        # --- TIMER END ---

        if not scan_results:
            logging.warning("❗ No files were found or processed.")
            return

        # Write output using the specialized function (delimiter is implicitly TXT_DELIMITER)
        write_structured_output(scan_results, args.output, args.format)

        end_time = time.time()

        # --- PERFORMANCE REPORT ---
        logging.info("-" * 50)
        logging.info(
            f"⏱️ Traversal/Fetch Time: {traversal_end - traversal_start:.2f} seconds"
        )
        logging.info(
            f"⏱️ Total Runtime (End-to-End): {end_time - start_time:.2f} seconds"
        )
        logging.info("-" * 50)
        # --------------------------

    except Exception as e:
        logging.critical(f"\n❌ A critical error occurred: {e.__class__.__name__}: {e}")
        logging.info("Please check path, connection, and permissions.")
        sys.exit(1)


if __name__ == "__main__":
    main()
