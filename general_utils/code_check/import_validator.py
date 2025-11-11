"""Python Import Validator.

Validates all import statements in a Python project, checking if imported
symbols actually exist in their source modules.

Usage:
    python import_validator.py [PROJECT_PATH] [OPTIONS]

Examples:
    python import_validator.py .
    python import_validator.py . --symbols custom_symbols.json
    python import_validator.py . --output imports.csv --verbose
    python import_validator.py . --exclude tests --exclude docs
"""

import ast
import json
import csv
import sys
import argparse
import logging
import tokenize
import difflib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from importlib.util import resolve_name
from functools import lru_cache

__version__ = "1.1.0"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Default directories to exclude from scanning
DEFAULT_EXCLUDED_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "env",
    ".tox",
    ".pytest_cache",
    "node_modules",
    ".mypy_cache",
    ".eggs",
    "build",
    "dist",
    "*.egg-info",
    "general_utils",
    "not_purely_project_code_related",
    "tests_old",
    "tests",
}


class ImportStatus(str, Enum):
    """Status of a validated import."""

    EXISTS = "exists"
    MISSING = "missing"
    EXTERNAL = "external"
    UNCERTAIN = "uncertain"


@dataclass
class ImportInfo:
    """Information about an import statement."""

    source_file: str
    source_line: int
    import_type: str  # 'import' or 'from_import' or 'from_import_star'
    module: str
    imported_name: Optional[str]
    alias: Optional[str]
    status: str
    details: str


class ImportValidator(ast.NodeVisitor):
    """AST visitor that validates import statements."""

    def __init__(self, project_root: str, symbol_table: Dict[str, Any]):
        """Initialize the import validator.

        Args:
            project_root: Root directory of the project.
            symbol_table: Symbol table dictionary.
        """
        self.project_root = project_root
        self.symbol_table = symbol_table
        self.imports: List[ImportInfo] = []
        self.current_file: Optional[str] = None
        self.current_module: Optional[str] = None

        # Pre-build lookup sets for performance
        self._build_lookup_caches()

    def _build_lookup_caches(self) -> None:
        """Build lookup caches for faster symbol resolution."""
        self.module_functions: Dict[str, Set[str]] = {}
        self.module_classes: Dict[str, Set[str]] = {}

        for module_name, module_info in self.symbol_table.items():
            if "error" in module_info:
                continue

            # Cache function names
            func_names = {f["name"] for f in module_info.get("functions", [])}
            self.module_functions[module_name] = func_names

            # Cache class names
            class_names = set(module_info.get("classes", {}).keys())
            self.module_classes[module_name] = class_names

    @lru_cache(maxsize=512)
    def _resolve_module(self, module: str, level: int) -> str:
        """Resolve relative imports to absolute module paths.

        Args:
            module: The module name from the import statement.
            level: The relative import level (number of dots).

        Returns:
            The resolved absolute module path.
        """
        try:
            if level > 0:
                return resolve_name(f"{'.' * level}{module}", self.current_module or "")
            return module
        except (ValueError, ImportError) as e:
            logger.warning(
                f"Could not resolve relative import in {self.current_file}: "
                f"level={level}, module={module}, error={e}"
            )
            return module

    def _suggest_alternatives(self, name: str, module: str) -> str:
        """Suggest alternative names if import is missing.

        Args:
            name: The missing symbol name.
            module: The module it was imported from.

        Returns:
            Suggestion string or empty string.
        """
        if module not in self.symbol_table:
            return ""

        module_info = self.symbol_table[module]
        available = []

        # Collect all available names
        available.extend(self.module_functions.get(module, []))
        available.extend(self.module_classes.get(module, []))

        if not available:
            return ""

        # Find close matches
        matches = difflib.get_close_matches(name, available, n=3, cutoff=0.6)
        if matches:
            return f" (Did you mean: {', '.join(matches)}?)"
        return ""

    def _check_builtin_or_stdlib(self, module_name: str) -> Tuple[bool, str]:
        """Check if module is a built-in or stdlib module.

        Args:
            module_name: The module name to check.

        Returns:
            Tuple of (is_builtin, details).
        """
        base_module = module_name.split(".")[0]

        # Check built-in modules
        if base_module in sys.builtin_module_names:
            return True, f"Built-in module: {base_module}"

        # Common stdlib modules (not exhaustive but covers most common ones)
        stdlib_modules = {
            "os",
            "sys",
            "re",
            "json",
            "csv",
            "math",
            "random",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "pathlib",
            "typing",
            "dataclasses",
            "enum",
            "abc",
            "logging",
            "argparse",
            "subprocess",
            "io",
            "time",
            "pickle",
            "copy",
            "unittest",
            "pytest",
            "asyncio",
        }

        if base_module in stdlib_modules:
            return True, f"Standard library module: {base_module}"

        return False, ""

    def _handle_star_import(
        self, node: ast.ImportFrom, resolved_module: str
    ) -> ImportInfo:
        """Handle star import (from module import *).

        Args:
            node: The ImportFrom AST node.
            resolved_module: The resolved module path.

        Returns:
            ImportInfo for the star import.
        """
        status = ImportStatus.EXTERNAL
        details = "Star import from external library"

        if resolved_module in self.symbol_table:
            # Check if module defines __all__
            # Note: We'd need to extract __all__ in symbol_table_builder to fully support this
            status = ImportStatus.UNCERTAIN
            details = (
                "Star import from project module (individual symbols not validated)"
            )
        elif not resolved_module:
            status = ImportStatus.MISSING
            details = "Star import from unresolved module"

        return ImportInfo(
            source_file=self.current_file or "",
            source_line=node.lineno,
            import_type="from_import_star",
            module=resolved_module,
            imported_name="*",
            alias=None,
            status=status.value,
            details=details,
        )

    def _check_symbol_in_module(
        self, name: str, resolved_module: str
    ) -> Tuple[ImportStatus, str]:
        """Check if a symbol exists in a module.

        Args:
            name: The symbol name to check.
            resolved_module: The module to check in.

        Returns:
            Tuple of (status, details).
        """
        if resolved_module not in self.symbol_table:
            is_builtin, builtin_details = self._check_builtin_or_stdlib(resolved_module)
            if is_builtin:
                return ImportStatus.EXTERNAL, builtin_details

            # Check if it's in the project at all
            base_module = resolved_module.split(".")[0]
            project_modules = {m.split(".")[0] for m in self.symbol_table.keys()}

            if base_module not in project_modules:
                return ImportStatus.EXTERNAL, "External library or not in project"
            else:
                return (
                    ImportStatus.MISSING,
                    f"Module '{resolved_module}' not found in project",
                )

        module_info = self.symbol_table[resolved_module]

        # Check if it's a class
        if name in self.module_classes.get(resolved_module, set()):
            return (
                ImportStatus.EXISTS,
                f"Class '{name}' found in module '{resolved_module}'",
            )

        # Check if it's a function
        if name in self.module_functions.get(resolved_module, set()):
            return (
                ImportStatus.EXISTS,
                f"Function '{name}' found in module '{resolved_module}'",
            )

        # Check if it's a submodule
        potential_submodule = f"{resolved_module}.{name}"
        if potential_submodule in self.symbol_table:
            return (
                ImportStatus.EXISTS,
                f"Submodule '{potential_submodule}' found",
            )

        # Check if it's a method (unusual but possible)
        for class_name, class_info in module_info.get("classes", {}).items():
            for method in class_info.get("methods", []):
                if method["name"] == name:
                    return (
                        ImportStatus.UNCERTAIN,
                        f"'{name}' is a method of class '{class_name}' (unusual import)",
                    )

        # Not found - suggest alternatives
        suggestion = self._suggest_alternatives(name, resolved_module)
        return (
            ImportStatus.MISSING,
            f"Symbol '{name}' NOT FOUND in module '{resolved_module}'{suggestion}",
        )

    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statement.

        Args:
            node: The Import AST node.
        """
        for alias in node.names:
            module_name = alias.name
            asname = alias.asname

            # Check if module exists
            is_builtin, builtin_details = self._check_builtin_or_stdlib(module_name)

            if is_builtin:
                status = ImportStatus.EXTERNAL
                details = builtin_details
            elif module_name in self.symbol_table:
                status = ImportStatus.EXISTS
                details = "Module found in project"
            else:
                # Check if it's a submodule by checking parent modules
                parts = module_name.split(".")
                found = False
                for i in range(len(parts), 0, -1):
                    partial = ".".join(parts[:i])
                    if partial in self.symbol_table:
                        status = ImportStatus.EXISTS
                        details = f"Parent module '{partial}' found"
                        found = True
                        break

                if not found:
                    base_module = parts[0]
                    project_modules = {
                        m.split(".")[0] for m in self.symbol_table.keys()
                    }
                    if base_module not in project_modules:
                        status = ImportStatus.EXTERNAL
                        details = "External library"
                    else:
                        status = ImportStatus.MISSING
                        details = f"Module '{module_name}' not found in project"

            import_info = ImportInfo(
                source_file=self.current_file or "",
                source_line=node.lineno,
                import_type="import",
                module=module_name,
                imported_name=None,
                alias=asname,
                status=status.value,
                details=details,
            )
            self.imports.append(import_info)

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from-import statement.

        Args:
            node: The ImportFrom AST node.
        """
        module = node.module if node.module else ""
        resolved_module = self._resolve_module(module, node.level)

        for alias in node.names:
            name = alias.name
            asname = alias.asname

            # Handle star imports
            if name == "*":
                import_info = self._handle_star_import(node, resolved_module)
                self.imports.append(import_info)
                continue

            # Check if the imported symbol exists
            status, details = self._check_symbol_in_module(name, resolved_module)

            import_info = ImportInfo(
                source_file=self.current_file or "",
                source_line=node.lineno,
                import_type="from_import",
                module=resolved_module,
                imported_name=name,
                alias=asname,
                status=status.value,
                details=details,
            )
            self.imports.append(import_info)

        self.generic_visit(node)

    def analyze_file(self, file_path: str, module_name: str) -> None:
        """Analyze a single Python file for imports.

        Args:
            file_path: Path to the Python file.
            module_name: The module name for this file.
        """
        self.current_file = file_path
        self.current_module = module_name

        try:
            with tokenize.open(file_path) as f:
                source = f.read()

            tree = ast.parse(source, filename=file_path)
            self.visit(tree)

        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Error analyzing {file_path}: {e}")


def load_symbol_table(symbol_table_path: str) -> Dict[str, Any]:
    """Load the symbol table from JSON file.

    Args:
        symbol_table_path: Path to the symbol table JSON file.

    Returns:
        The modules dictionary from the symbol table.
    """
    path = Path(symbol_table_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Symbol table not found: {symbol_table_path}\n"
            f"Run symbol_table_builder.py first to generate it."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "modules" in data:
        return data["modules"]
    else:
        return data


def run_validation(
    project_root: str,
    symbol_table_path: str = "project_symbols.json",
    output_path: str = "import_validation.csv",
    output_format: str = "csv",
    status_filter: Optional[str] = None,
    verbose: bool = False,
    quiet: bool = False,
) -> List[ImportInfo]:
    """Run complete import validation on a Python project.

    Args:
        project_root: Root directory of the project.
        symbol_table_path: Path to the JSON symbol table file.
        output_path: Path where the output should be written.
        output_format: Output format: 'csv' or 'json'.
        status_filter: If provided, only output imports with this status.
        verbose: Whether to print detailed progress information.
        quiet: Whether to suppress all non-error output.

    Returns:
        List of import information.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    elif quiet:
        logger.setLevel(logging.ERROR)

    logger.info("📖 Loading symbol table...")
    symbol_table = load_symbol_table(symbol_table_path)
    validator = ImportValidator(project_root, symbol_table)

    logger.info("🔍 Validating imports...")

    file_count = 0
    for module_name, module_info in symbol_table.items():
        if "error" in module_info:
            continue

        file_path = module_info["file_path"]
        file_count += 1

        logger.debug(f"  📄 {module_name}")
        if not quiet and not verbose and file_count % 10 == 0:
            logger.info(f"  Processed {file_count} files...")

        validator.analyze_file(file_path, module_name)

    # Apply filter if specified
    imports = validator.imports
    if status_filter:
        imports = [i for i in imports if i.status == status_filter]
        logger.info(
            f"  Filtered to {len(imports)} imports with status='{status_filter}'"
        )

    # Write output
    logger.info(f"💾 Writing results to {output_path}...")

    if output_format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([asdict(i) for i in imports], f, indent=2)
    else:  # csv
        fieldnames = [
            "source_file",
            "source_line",
            "import_type",
            "module",
            "imported_name",
            "alias",
            "status",
            "details",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([asdict(i) for i in imports])

    # Print summary
    if not quiet:
        print_summary(validator.imports, status_filter is not None)

    return imports


def print_summary(imports: List[ImportInfo], is_filtered: bool = False) -> None:
    """Print validation summary statistics.

    Args:
        imports: List of all analyzed imports.
        is_filtered: Whether the imports list has been filtered.
    """
    total = len(imports)

    if total == 0:
        logger.warning("\n⚠️  No imports found to validate.")
        return

    missing = sum(1 for i in imports if i.status == ImportStatus.MISSING.value)
    uncertain = sum(1 for i in imports if i.status == ImportStatus.UNCERTAIN.value)
    external = sum(1 for i in imports if i.status == ImportStatus.EXTERNAL.value)
    exists = sum(1 for i in imports if i.status == ImportStatus.EXISTS.value)

    logger.info("\n" + "=" * 70)
    logger.info("📊 IMPORT VALIDATION SUMMARY")
    logger.info("=" * 70)

    if is_filtered:
        logger.info(f"Showing filtered results ({total} imports)")
    else:
        logger.info(f"Total imports analyzed: {total}")

    logger.info(f"✅ Verified exists:   {exists:6d} ({exists/total*100:5.1f}%)")
    logger.info(f"🔌 External library:  {external:6d} ({external/total*100:5.1f}%)")
    logger.info(f"❓ Uncertain:         {uncertain:6d} ({uncertain/total*100:5.1f}%)")
    logger.info(f"❌ MISSING:           {missing:6d} ({missing/total*100:5.1f}%)")
    logger.info("=" * 70)

    if missing > 0:
        logger.info(f"\n🚨 Found {missing} MISSING imports!")
        logger.info("\nMissing imports details:")
        missing_imports = [i for i in imports if i.status == ImportStatus.MISSING.value]

        for imp in missing_imports:
            logger.info(f"\n  File: {imp.source_file}")
            logger.info(f"  Line: {imp.source_line}")
            logger.info(f"  Import: from {imp.module} import {imp.imported_name}")
            logger.info(f"  Issue: {imp.details}")

    if uncertain > 0:
        logger.info(
            f"\n⚠️  Found {uncertain} uncertain imports (star imports or unusual patterns)"
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate Python project imports and identify missing references.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s .
  %(prog)s /path/to/project --symbols custom_symbols.json
  %(prog)s . --output imports.csv --format json
  %(prog)s . --filter missing --verbose
  %(prog)s . --exclude tests --exclude migrations
        """,
    )

    parser.add_argument(
        "project_root",
        nargs="?",
        default=".",
        help="Path to the project root directory (default: current directory)",
    )

    parser.add_argument(
        "-s",
        "--symbols",
        default="project_symbols.json",
        help="Path to symbol table JSON file (default: project_symbols.json)",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="import_validation.csv",
        help="Output file path (default: import_validation.csv)",
    )

    parser.add_argument(
        "-f",
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format: csv or json (default: csv)",
    )

    parser.add_argument(
        "--filter",
        choices=["exists", "missing", "external", "uncertain"],
        help="Only output imports with this status",
    )

    parser.add_argument(
        "-e",
        "--exclude",
        action="append",
        default=[],
        help="Additional directory names to exclude (can be used multiple times)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed progress information",
    )

    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress all non-error output"
    )

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for the import validator."""
    args = parse_arguments()

    try:
        imports = run_validation(
            project_root=args.project_root,
            symbol_table_path=args.symbols,
            output_path=args.output,
            output_format=args.format,
            status_filter=args.filter,
            verbose=args.verbose,
            quiet=args.quiet,
        )

        if not args.quiet:
            logger.info(f"\n✅ Validation complete! Results saved to {args.output}")

        # Exit with error code if missing imports found
        if not args.filter:
            missing_count = sum(
                1 for i in imports if i.status == ImportStatus.MISSING.value
            )
            if missing_count > 0:
                return 1

        return 0

    except FileNotFoundError as e:
        logger.error(f"❌ Error: {e}")
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in symbol table: {e}")
        return 1
    except KeyboardInterrupt:
        logger.error("\n⚠️  Validation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
