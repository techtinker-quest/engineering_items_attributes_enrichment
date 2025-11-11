"""Python Project Symbol Table Builder.

A CLI tool that scans Python projects and catalogs all classes, functions, and methods
into a comprehensive JSON symbol table for analysis and documentation.

Usage:
    python symbol_table_builder.py [PROJECT_PATH] [OPTIONS]

Examples:
    python symbol_table_builder.py .
    python symbol_table_builder.py /path/to/project --output symbols.json
    python symbol_table_builder.py . --exclude tests --exclude docs --verbose
    python symbol_table_builder.py . --include-private --quiet
"""

import ast
import json
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

__version__ = "2.0.0"

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


class SymbolExtractor(ast.NodeVisitor):
    """AST visitor that extracts symbols from Python source code.

    Traverses the AST and catalogs classes, functions, methods, docstrings,
    signatures, and inheritance information.
    """

    def __init__(self, include_private: bool = False):
        """Initialize the symbol extractor.

        Args:
            include_private: If True, include symbols starting with underscore.
        """
        self.include_private = include_private
        self.module_docstring: Optional[str] = None
        self.classes: Dict[str, Dict[str, Any]] = {}
        self.functions: List[Dict[str, Any]] = []
        self.current_class: Optional[str] = None
        self.class_stack: List[str] = []  # Track nested classes

    def should_include(self, name: str) -> bool:
        """Check if a symbol should be included based on privacy settings.

        Args:
            name: The symbol name to check.

        Returns:
            True if the symbol should be included, False otherwise.
        """
        if self.include_private:
            return True
        return not name.startswith("_") or name.startswith("__") and name.endswith("__")

    @staticmethod
    def get_docstring(node: ast.AST) -> Optional[str]:
        """Extract docstring from an AST node.

        Args:
            node: AST node to extract docstring from.

        Returns:
            The docstring text if present, None otherwise.
        """
        return ast.get_docstring(node)

    @staticmethod
    def get_function_signature(node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract function signature including parameters and return type.

        Args:
            node: Function definition AST node.

        Returns:
            Dictionary containing parameters list and return type annotation.
        """
        params = []
        for arg in node.args.args:
            param_info = {"name": arg.arg}
            if arg.annotation:
                try:
                    param_info["type"] = ast.unparse(arg.annotation)
                except Exception:
                    param_info["type"] = None
            params.append(param_info)

        return_type = None
        if node.returns:
            try:
                return_type = ast.unparse(node.returns)
            except Exception:
                pass

        return {"parameters": params, "return_type": return_type}

    @staticmethod
    def get_method_type(node: ast.FunctionDef) -> str:
        """Determine if a method is static, class, or instance method.

        Args:
            node: Function definition AST node.

        Returns:
            String: 'static', 'class', or 'instance'.
        """
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                if decorator.id == "staticmethod":
                    return "static"
                elif decorator.id == "classmethod":
                    return "class"
        return "instance"

    @staticmethod
    def get_base_classes(node: ast.ClassDef) -> List[str]:
        """Extract base class names from a class definition.

        Args:
            node: Class definition AST node.

        Returns:
            List of base class names as strings.
        """
        bases = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except Exception:
                bases.append("<unparseable>")
        return bases

    def visit_Module(self, node: ast.Module) -> None:
        """Visit module node and extract module-level docstring.

        Args:
            node: Module AST node.
        """
        self.module_docstring = self.get_docstring(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition and extract class information.

        Args:
            node: Class definition AST node.
        """
        if not self.should_include(node.name):
            return

        # Build full class name including nesting
        if self.class_stack:
            full_name = ".".join(self.class_stack + [node.name])
        else:
            full_name = node.name

        self.classes[full_name] = {
            "line": node.lineno,
            "docstring": self.get_docstring(node),
            "bases": self.get_base_classes(node),
            "methods": [],
            "nested_classes": [],
        }

        # Track nesting for inner classes
        previous_class = self.current_class
        self.current_class = full_name
        self.class_stack.append(node.name)

        # Visit class body
        self.generic_visit(node)

        # Restore previous context
        self.class_stack.pop()
        self.current_class = previous_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition and extract function/method information.

        Args:
            node: Function definition AST node.
        """
        if not self.should_include(node.name):
            return

        func_info = {
            "name": node.name,
            "line": node.lineno,
            "docstring": self.get_docstring(node),
            "signature": self.get_function_signature(node),
        }

        if self.current_class:
            # This is a method
            func_info["method_type"] = self.get_method_type(node)
            self.classes[self.current_class]["methods"].append(func_info)
        else:
            # This is a module-level function
            self.functions.append(func_info)

        # Don't visit nested functions (inner functions are typically not public API)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition.

        Args:
            node: Async function definition AST node.
        """
        # Treat async functions the same as regular functions
        self.visit_FunctionDef(node)


def get_module_path(base_path: Path, file_path: Path) -> str:
    """Convert a file system path to a Python module import path.

    Transforms absolute or relative file paths into dot-separated module
    notation suitable for Python imports. Handles both regular modules
    and __init__.py files correctly.

    Args:
        base_path: The root directory of the project.
        file_path: The full path to the Python file to convert.

    Returns:
        A dot-separated module path string (e.g., 'src.utils.helper').
        Returns empty string if the file is not within base_path.

    Examples:
        >>> get_module_path(Path('/project'), Path('/project/src/utils/helper.py'))
        'src.utils.helper'
        >>> get_module_path(Path('/project'), Path('/project/pkg/__init__.py'))
        'pkg'
    """
    try:
        relative = file_path.relative_to(base_path)
    except ValueError:
        return ""

    # Remove .py extension
    if relative.suffix == ".py":
        relative = relative.with_suffix("")

    # Convert path to module notation
    parts = list(relative.parts)

    # Remove __init__ from module path
    if parts and parts[-1] == "__init__":
        parts.pop()

    return ".".join(parts)


def process_file(
    file_path: Path, project_root: Path, include_private: bool, verbose: bool
) -> Optional[Dict[str, Any]]:
    """Process a single Python file and extract symbols.

    Args:
        file_path: Path to the Python file to process.
        project_root: Root directory of the project.
        include_private: Whether to include private symbols.
        verbose: Whether to print verbose output.

    Returns:
        Dictionary containing extracted symbols, or None if processing failed.
    """
    module_name = get_module_path(project_root, file_path)

    if not module_name:
        return None

    result = {
        "module": module_name,
        "file_path": str(file_path),
        "classes": {},
        "functions": [],
        "module_docstring": None,
    }

    try:
        # Try UTF-8 first
        try:
            source_code = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback to latin-1 which accepts all byte values
            if verbose:
                print(f"   ⚠️  UTF-8 decode failed for {file_path}, trying latin-1...")
            source_code = file_path.read_text(encoding="latin-1")

        tree = ast.parse(source_code, filename=str(file_path))

        # Extract symbols using visitor
        extractor = SymbolExtractor(include_private=include_private)
        extractor.visit(tree)

        result["module_docstring"] = extractor.module_docstring
        result["classes"] = extractor.classes
        result["functions"] = extractor.functions

        return result

    except SyntaxError as e:
        error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
        result["error"] = error_msg
        if verbose:
            print(f"   ⚠️  {file_path}: {error_msg}")
        return result

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        result["error"] = error_msg
        if verbose:
            print(f"   ⚠️  {file_path}: {error_msg}")
        return result


def build_symbol_table(
    project_root: str,
    excluded_dirs: set,
    include_private: bool,
    verbose: bool,
    quiet: bool,
) -> Dict[str, Any]:
    """Scan a Python project and catalog all top-level definitions.

    Args:
        project_root: Path to the root directory of the Python project.
        excluded_dirs: Set of directory names to exclude from scanning.
        include_private: Whether to include private symbols (starting with _).
        verbose: Whether to print detailed progress information.
        quiet: Whether to suppress all non-error output.

    Returns:
        Complete symbol table with metadata.

    Raises:
        OSError: If the project_root directory cannot be accessed.
    """
    root_path = Path(project_root).resolve()

    if not root_path.exists():
        raise OSError(f"Project root does not exist: {project_root}")
    if not root_path.is_dir():
        raise OSError(f"Project root is not a directory: {project_root}")

    modules: Dict[str, Dict[str, Any]] = {}
    error_count = 0
    file_count = 0

    if not quiet:
        print(f"🔍 Scanning {root_path}...")

    for py_file in root_path.rglob("*.py"):
        # Skip excluded directories
        if any(excluded in py_file.parts for excluded in excluded_dirs):
            continue

        file_count += 1
        if verbose:
            print(f"   Processing: {py_file.relative_to(root_path)}")

        file_result = process_file(py_file, root_path, include_private, verbose)

        if file_result:
            module_name = file_result["module"]
            modules[module_name] = {
                "file_path": file_result["file_path"],
                "module_docstring": file_result["module_docstring"],
                "classes": file_result["classes"],
                "functions": file_result["functions"],
            }

            if "error" in file_result:
                modules[module_name]["error"] = file_result["error"]
                error_count += 1

    # Build complete symbol table with metadata
    symbol_table = {
        "metadata": {
            "tool": "Python Symbol Table Builder",
            "version": __version__,
            "scan_time": datetime.now().isoformat(),
            "project_root": str(root_path),
            "files_scanned": file_count,
            "errors": error_count,
            "include_private": include_private,
        },
        "modules": modules,
    }

    return symbol_table


def print_summary(symbol_table: Dict[str, Any], quiet: bool) -> None:
    """Print a summary of the extracted symbols.

    Args:
        symbol_table: The complete symbol table dictionary.
        quiet: Whether to suppress output.
    """
    if quiet:
        return

    modules = symbol_table["modules"]
    metadata = symbol_table["metadata"]

    total_modules = len(modules)
    total_classes = sum(len(m["classes"]) for m in modules.values())
    total_functions = sum(len(m["functions"]) for m in modules.values())
    total_methods = sum(
        len(class_info["methods"])
        for module in modules.values()
        for class_info in module["classes"].values()
    )

    print(f"\n📊 Summary:")
    print(f"   • Modules: {total_modules}")
    print(f"   • Classes: {total_classes}")
    print(f"   • Functions: {total_functions}")
    print(f"   • Methods: {total_methods}")
    print(f"   • Files scanned: {metadata['files_scanned']}")

    if metadata["errors"] > 0:
        print(f"   ⚠️  Errors: {metadata['errors']}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Scan a Python project and generate a symbol table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s .
  %(prog)s /path/to/project --output my_symbols.json
  %(prog)s . --exclude tests --exclude docs --verbose
  %(prog)s . --include-private --quiet
        """,
    )

    parser.add_argument(
        "project_root",
        nargs="?",
        default=".",
        help="Path to the project root directory (default: current directory)",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="project_symbols.json",
        help="Output JSON file path (default: project_symbols.json)",
    )

    parser.add_argument(
        "-e",
        "--exclude",
        action="append",
        default=[],
        help="Additional directory names to exclude (can be used multiple times)",
    )

    parser.add_argument(
        "-p",
        "--include-private",
        action="store_true",
        help="Include private symbols (names starting with _)",
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
    """Main entry point for the symbol table builder.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_arguments()

    # Combine default and user-specified exclusions
    excluded_dirs = DEFAULT_EXCLUDED_DIRS.union(set(args.exclude))

    try:
        symbol_table = build_symbol_table(
            project_root=args.project_root,
            excluded_dirs=excluded_dirs,
            include_private=args.include_private,
            verbose=args.verbose,
            quiet=args.quiet,
        )

        # Write output
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(symbol_table, f, indent=2)

        if not args.quiet:
            print(f"✅ Symbol table saved to {output_path}")

        print_summary(symbol_table, args.quiet)

        return 0

    except OSError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n⚠️  Scan interrupted by user", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
