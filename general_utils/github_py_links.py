import argparse
import requests
import csv
from pathlib import Path
from urllib.parse import quote
from datetime import datetime

# Constants
GITHUB_API_URL = "https://api.github.com"
HEADERS = {"Accept": "application/vnd.github.v3+json"}


def fetch_tree(owner: str, repo: str, branch: str, pat: str = None) -> list:
    """Fetch recursive tree from GitHub API."""
    headers = HEADERS.copy()
    if pat:
        headers["Authorization"] = f"token {pat}"

    # Get branch SHA
    ref_url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/git/ref/heads/{branch}"
    resp = requests.get(ref_url, headers=headers, timeout=10)
    resp.raise_for_status()
    sha = resp.json()["object"]["sha"]

    # Get recursive tree
    tree_url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/git/trees/{sha}?recursive=1"
    resp = requests.get(tree_url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json().get("tree", [])


def build_urls(owner: str, repo: str, branch: str, path: str) -> tuple[str, str]:
    """Build HTML and raw URLs for a file."""
    encoded_path = quote(path, safe="/")
    html_url = f"https://github.com/{owner}/{repo}/blob/{branch}/{encoded_path}"
    raw_url = (
        f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{encoded_path}"
    )
    return html_url, raw_url


def write_markdown(py_files: list, output_path: Path):
    """Write results to Markdown file."""
    md_content = f"# .py Files in Repository\n\n"
    md_content += f"**Repository**: `{py_files[0]['owner']}/{py_files[0]['repo']}`  \n"
    md_content += f"**Branch**: `{py_files[0]['branch']}`  \n"
    md_content += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
    md_content += f"**Total Files**: {len(py_files)}\n\n"

    md_content += "| Path | Size | View | Raw |\n"
    md_content += "|------|------|------|-----|\n"

    for f in py_files:
        md_content += f"| `{f['path']}` | {f['size']} | [View]({f['html_url']}) | [Raw]({f['raw_url']}) |\n"

    output_path.write_text(md_content, encoding="utf-8")
    print(f"Markdown report saved: {output_path}")


def write_csv(py_files: list, output_path: Path):
    """Write results to CSV file."""
    fieldnames = ["Path", "Size (Bytes)", "HTML URL", "Raw URL"]
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for f in py_files:
            writer.writerow(
                {
                    "Path": f["path"],
                    "Size (Bytes)": f["size"],
                    "HTML URL": f["html_url"],
                    "Raw URL": f["raw_url"],
                }
            )
    print(f"CSV report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="List all .py files in a GitHub repo with direct links (MD or CSV)."
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Owner/repo e.g., techtinker-quest/engineering_items_attributes_enrichment",
    )
    parser.add_argument("--branch", default="main", help="Branch name (default: main)")
    parser.add_argument(
        "--pat", default=None, help="GitHub PAT (for private repos or rate limits)"
    )
    parser.add_argument(
        "--format",
        choices=["md", "csv"],
        default="md",
        help="Output format: md (Markdown) or csv (CSV). Default: md",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. Auto-generated if not provided.",
    )

    args = parser.parse_args()

    # Parse owner/repo
    if "/" not in args.repo:
        print("Error: --repo must be in format owner/repo")
        return
    owner, repo = args.repo.split("/", 1)

    # Auto-generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = "csv" if args.format == "csv" else "md"
    default_name = f"py_links_{repo}_{args.branch}_{timestamp}.{ext}"
    output_path = Path(args.output) if args.output else Path(default_name)

    print(f"Fetching .py files from {owner}/{repo} ({args.branch})...")

    try:
        tree = fetch_tree(owner, repo, args.branch, args.pat)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print("Repo not found or private (check PAT and access).")
        elif e.response.status_code == 401:
            print("Invalid PAT. Regenerate at https://github.com/settings/tokens")
        else:
            print(f"HTTP Error: {e.response.status_code}")
        return
    except Exception as e:
        print(f"Request failed: {e}")
        return

    py_files = []
    for item in tree:
        if item["type"] == "blob" and item["path"].endswith(".py"):
            html_url, raw_url = build_urls(owner, repo, args.branch, item["path"])
            py_files.append(
                {
                    "path": item["path"],
                    "size": item.get("size", "N/A"),
                    "html_url": html_url,
                    "raw_url": raw_url,
                    "owner": owner,
                    "repo": repo,
                    "branch": args.branch,
                }
            )

    if not py_files:
        print("No .py files found in the repository.")
        return

    # Sort by path
    py_files.sort(key=lambda x: x["path"])

    # Console preview
    print(f"\nFound {len(py_files)} .py file(s):")
    for f in py_files[:5]:
        print(f"  • {f['path']} ({f['size']} bytes)")
    if len(py_files) > 5:
        print(f"  ... and {len(py_files) - 5} more.")

    # Export
    if args.format == "csv":
        write_csv(py_files, output_path)
    else:
        write_markdown(py_files, output_path)


if __name__ == "__main__":
    main()
