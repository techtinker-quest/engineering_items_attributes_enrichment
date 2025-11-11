"""Python Call Graph Analyzer.

Analyzes Python projects to build a comprehensive call graph, tracking all
function/method calls and identifying missing references, external dependencies,
and uncertain resolutions.

This tool works in conjunction with the Symbol Table Builder to provide
complete project analysis.

Usage:
    python call_graph_analyzer.py [PROJECT_PATH] [OPTIONS]

Examples:
    python call_graph_analyzer.py .
    python call_graph_analyzer.py . --symbols custom_symbols.json
    python call_graph_analyzer.py . --output calls.csv --format json
    python call_graph_analyzer.py . --filter missing --verbose
"""

import ast
import json
import csv
import argparse
import sys
import logging
import tokenize
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from typing_extensions import TypedDict
from collections import defaultdict
from enum import Enum
from importlib.util import resolve_name

__version__ = "3.0.0"


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class Confidence(str, Enum):
    """Confidence levels for call resolution."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNRESOLVABLE = "unresolvable"


class Status(str, Enum):
    """Status of a resolved call target."""

    EXISTS = "exists"
    MISSING = "missing"
    EXTERNAL = "external"
    UNCERTAIN = "uncertain"
    UNRESOLVABLE = "unresolvable"


class CallInfo(TypedDict):
    """Type definition for call information dictionary."""

    source_file: str
    source_line: int
    caller_module: str
    caller_name: str
    callee_path: str
    status: str
    confidence: str


class ImportResolver:
    """Handles import statement resolution and tracking.

    Manages both regular imports and from-imports, including relative imports,
    and provides methods to resolve imported names to their full module paths.

    Attributes:
        current_module: The module currently being analyzed.
        imports: Mapping of import aliases to full module paths.
        from_imports: Mapping of imported names to their full paths.
        star_imports: List of modules imported with 'from x import *'.
    """

    def __init__(self, current_module: Optional[str] = None):
        """Initialize the import resolver.

        Args:
            current_module: The module name for resolving relative imports.
        """
        self.current_module = current_module
        self.imports: Dict[str, str] = {}
        self.from_imports: Dict[str, str] = {}
        self.star_imports: List[str] = []

    def reset(self, current_module: Optional[str] = None) -> None:
        """Reset the resolver for a new file.

        Args:
            current_module: The new module name.
        """
        self.current_module = current_module
        self.imports.clear()
        self.from_imports.clear()
        self.star_imports.clear()

    def handle_import(self, node: ast.Import) -> None:
        """Process an import statement.

        Args:
            node: The Import AST node.
        """
        for alias in node.names:
            module_name = alias.name
            asname = alias.asname if alias.asname else module_name
            self.imports[asname] = module_name

    def handle_from_import(self, node: ast.ImportFrom) -> None:
        """Process a from-import statement with robust relative import resolution.

        Args:
            node: The ImportFrom AST node.
        """
        module = node.module if node.module else ""

        # Use importlib for robust relative import resolution
        try:
            if node.level > 0:
                # Relative import
                resolved_module = resolve_name(
                    f"{'.' * node.level}{module}", self.current_module or ""
                )
            else:
                # Absolute import
                resolved_module = module
        except (ValueError, ImportError):
            # Fallback if resolution fails
            resolved_module = module
            logger.warning(
                f"Could not resolve relative import: "
                f"level={node.level}, module={module}, "
                f"current={self.current_module}"
            )

        for alias in node.names:
            name = alias.name
            asname = alias.asname if alias.asname else name

            if name == "*":
                # Star import - ambiguous
                self.star_imports.append(resolved_module)
                logger.warning(
                    f"Star import detected: 'from {resolved_module} import *' "
                    f"in {self.current_module}. This may cause resolution issues."
                )
            elif resolved_module:
                self.from_imports[asname] = f"{resolved_module}.{name}"
            else:
                self.from_imports[asname] = name

    def resolve_name(self, name: str) -> Optional[str]:
        """Resolve an imported name to its full module path.

        Args:
            name: The name to resolve.

        Returns:
            The full module path, or None if not found.
        """
        if name in self.from_imports:
            return self.from_imports[name]
        if name in self.imports:
            return self.imports[name]
        return None


class ContextTracker:
    """Tracks the current analysis context (module, class, function).

    Maintains the state of what code element is currently being analyzed
    to provide proper context for call resolution.

    Attributes:
        current_module: The module currently being analyzed.
        current_file: Path to the file currently being analyzed.
        class_stack: Stack of class names for nested class tracking.
        current_function: Name of the function currently being visited.
    """

    def __init__(self):
        """Initialize the context tracker."""
        self.current_module: Optional[str] = None
        self.current_file: Optional[str] = None
        self.class_stack: List[str] = []
        self.current_function: Optional[str] = None

    def reset(
        self, current_module: Optional[str] = None, current_file: Optional[str] = None
    ) -> None:
        """Reset context for a new file.

        Args:
            current_module: The new module name.
            current_file: Path to the new file.
        """
        self.current_module = current_module
        self.current_file = current_file
        self.class_stack.clear()
        self.current_function = None

    def enter_class(self, class_name: str) -> None:
        """Enter a class definition.

        Args:
            class_name: Name of the class being entered.
        """
        self.class_stack.append(class_name)

    def exit_class(self) -> None:
        """Exit the current class definition."""
        if self.class_stack:
            self.class_stack.pop()

    def enter_function(self, function_name: str) -> None:
        """Enter a function definition.

        Args:
            function_name: Name of the function being entered.
        """
        self.current_function = function_name

    def exit_function(self) -> None:
        """Exit the current function definition."""
        self.current_function = None

    @property
    def current_class(self) -> Optional[str]:
        """Get the full current class path (e.g., 'Outer.Inner').

        Returns:
            Dot-separated class path, or None if not in a class.
        """
        return ".".join(self.class_stack) if self.class_stack else None

    @property
    def caller_name(self) -> str:
        """Get the current caller name for call tracking.

        Returns:
            Formatted caller name (e.g., 'Class.method' or '<module:name>').
        """
        if self.current_class and self.current_function:
            return f"{self.current_class}.{self.current_function}"
        elif self.current_function:
            return self.current_function
        else:
            return f"<module:{self.current_module}>"


class CallGraphAnalyzer(ast.NodeVisitor):
    """AST visitor that analyzes function/method calls in Python code.

    Tracks all function and method calls, resolves their targets using import
    information and a symbol table, and classifies each call as existing,
    missing, external, or uncertain.

    Attributes:
        project_root: Root directory of the project being analyzed.
        symbol_table: Dictionary mapping module names to their symbol information.
        calls: List of all discovered calls with resolution information.
        import_resolver: Handles import statement resolution.
        context: Tracks current module/class/function context.
    """

    def __init__(self, project_root: str, symbol_table: Dict[str, Any]):
        """Initialize the call graph analyzer.

        Args:
            project_root: Root directory of the project being analyzed.
            symbol_table: Symbol table dictionary (modules only).
        """
        self.project_root = project_root
        self.symbol_table = symbol_table
        self.calls: List[CallInfo] = []

        self.import_resolver = ImportResolver()
        self.context = ContextTracker()

    def _reset_context(self) -> None:
        """Reset analysis context when starting a new file."""
        self.import_resolver.reset()
        self.context.reset()

    def visit_Import(self, node: ast.Import) -> None:
        """Process import statements.

        Args:
            node: The Import AST node to process.
        """
        self.import_resolver.handle_import(node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Process from-import statements.

        Args:
            node: The ImportFrom AST node to process.
        """
        self.import_resolver.handle_from_import(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition and update context.

        Args:
            node: The ClassDef AST node to visit.
        """
        self.context.enter_class(node.name)
        self.generic_visit(node)
        self.context.exit_class()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition and update context.

        Args:
            node: The FunctionDef AST node to visit.
        """
        self.context.enter_function(node.name)
        self.generic_visit(node)
        self.context.exit_function()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition.

        Args:
            node: The AsyncFunctionDef AST node to visit.
        """
        self.context.enter_function(node.name)
        self.generic_visit(node)
        self.context.exit_function()

    def _resolve_chained_attribute(self, node: ast.Attribute) -> Tuple[str, Confidence]:
        """Resolve chained attribute access like module.submodule.func.

        Args:
            node: The Attribute AST node.

        Returns:
            Tuple of (resolved_path, confidence).
        """
        parts = []
        current = node

        # Walk up the attribute chain
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value

        # Base should be a Name
        if isinstance(current, ast.Name):
            parts.append(current.id)
            parts.reverse()

            # Check if the base is an imported module
            base_name = parts[0]
            resolved = self.import_resolver.resolve_name(base_name)

            if resolved:
                # Replace base with resolved module
                full_path = resolved + "." + ".".join(parts[1:])
                return full_path, Confidence.MEDIUM

            return ".".join(parts), Confidence.LOW

        return ".".join(reversed(parts)), Confidence.LOW

    def resolve_call_target(self, call_node: ast.Call) -> Tuple[str, Confidence]:
        """Resolve the target of a function call.

        Attempts to determine what function or method is being called by analyzing
        the call expression, checking imports, and consulting the symbol table.

        Args:
            call_node: The Call AST node representing the function call.

        Returns:
            A tuple of (resolved_path, confidence).

        Examples:
            >>> analyzer.resolve_call_target(call_node)
            ('mymodule.MyClass.method', Confidence.HIGH)
        """
        if isinstance(call_node.func, ast.Name):
            # Direct name: func_name()
            func_name = call_node.func.id

            # Check from imports
            resolved = self.import_resolver.resolve_name(func_name)
            if resolved:
                return resolved, Confidence.HIGH

            # Check current module functions
            if (
                self.context.current_module
                and self.context.current_module in self.symbol_table
            ):
                module_info = self.symbol_table[self.context.current_module]

                # Check module-level functions
                if any(
                    f["name"] == func_name for f in module_info.get("functions", [])
                ):
                    return f"{self.context.current_module}.{func_name}", Confidence.HIGH

                # Check current class methods
                if self.context.current_class:
                    class_info = module_info["classes"].get(
                        self.context.current_class, {}
                    )
                    if any(
                        m["name"] == func_name for m in class_info.get("methods", [])
                    ):
                        return (
                            f"{self.context.current_module}.{self.context.current_class}.{func_name}",
                            Confidence.HIGH,
                        )

            # Might be builtin
            if func_name in dir(__builtins__):
                return f"<builtin>.{func_name}", Confidence.HIGH

            return func_name, Confidence.LOW

        elif isinstance(call_node.func, ast.Attribute):
            # Attribute access: obj.method() or module.func()
            attr_name = call_node.func.attr
            value = call_node.func.value

            # Handle super() calls
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                if value.func.id == "super" and self.context.current_class:
                    return (
                        f"<super:{self.context.current_module}.{self.context.current_class}>.{attr_name}",
                        Confidence.MEDIUM,
                    )

            if isinstance(value, ast.Name):
                base_name = value.id

                # Check if base is imported module
                resolved = self.import_resolver.resolve_name(base_name)
                if resolved:
                    return f"{resolved}.{attr_name}", Confidence.MEDIUM

                # Check if base is class in current module
                if (
                    self.context.current_module
                    and self.context.current_module in self.symbol_table
                ):
                    if (
                        base_name
                        in self.symbol_table[self.context.current_module]["classes"]
                    ):
                        return (
                            f"{self.context.current_module}.{base_name}.{attr_name}",
                            Confidence.MEDIUM,
                        )

                # self.method() call
                if base_name == "self" and self.context.current_class:
                    return (
                        f"{self.context.current_module}.{self.context.current_class}.{attr_name}",
                        Confidence.HIGH,
                    )

                # Could be variable.method() - hard to resolve statically
                return f"{base_name}.{attr_name}", Confidence.LOW

            elif isinstance(value, ast.Attribute):
                # Chained attribute access: module.submodule.func()
                return self._resolve_chained_attribute(value.func)  # type: ignore

            # Complex expressions
            return f"<complex>.{attr_name}", Confidence.LOW

        return "<unresolvable>", Confidence.UNRESOLVABLE

    def check_existence(self, resolved_path: str, confidence: Confidence) -> Status:
        """Check if a resolved call target exists in the symbol table.

        Validates whether the resolved function/method/class actually exists
        in the project by consulting the symbol table.

        Args:
            resolved_path: The fully qualified path to check.
            confidence: The confidence level of the resolution.

        Returns:
            Status enum value.

        Examples:
            >>> analyzer.check_existence('mymodule.func', Confidence.HIGH)
            Status.EXISTS
        """
        if confidence == Confidence.UNRESOLVABLE:
            return Status.UNRESOLVABLE

        if resolved_path.startswith("<builtin>"):
            return Status.EXISTS

        if resolved_path.startswith("<super:"):
            # Super calls are tricky - mark as uncertain
            return Status.UNCERTAIN

        if resolved_path.startswith("<"):
            return Status.UNCERTAIN

        parts = resolved_path.split(".")

        # Try module.Class.method (3+ parts) - including nested classes
        if len(parts) >= 3:
            for i in range(len(parts) - 2, 0, -1):
                mod_path = ".".join(parts[:i])
                if mod_path in self.symbol_table:
                    # Check for nested class path (e.g., Outer.Inner)
                    class_path = ".".join(parts[i:-1])
                    method_name = parts[-1]

                    class_info = self.symbol_table[mod_path]["classes"].get(class_path)
                    if class_info and any(
                        m["name"] == method_name for m in class_info.get("methods", [])
                    ):
                        return Status.EXISTS

        # Try module.function or module.Class (2 parts)
        if len(parts) >= 2:
            mod_path = ".".join(parts[:-1])
            name = parts[-1]
            if mod_path in self.symbol_table:
                module_info = self.symbol_table[mod_path]
                # Check functions
                if any(f["name"] == name for f in module_info.get("functions", [])):
                    return Status.EXISTS
                # Check classes
                if name in module_info.get("classes", {}):
                    return Status.EXISTS

        # Check if it's external library (not in our project)
        if parts[0] not in [m.split(".")[0] for m in self.symbol_table.keys()]:
            return Status.EXTERNAL

        return Status.MISSING

    def visit_Call(self, node: ast.Call) -> None:
        """Record and analyze a function or method call.

        Main visitor method that processes each call site, resolves the target,
        checks its existence, and records the call information.

        Args:
            node: The Call AST node representing the function call.
        """
        resolved_path, confidence = self.resolve_call_target(node)
        status = self.check_existence(resolved_path, confidence)

        call_info: CallInfo = {
            "source_file": self.context.current_file or "",
            "source_line": node.lineno,
            "caller_module": self.context.current_module or "",
            "caller_name": self.context.caller_name,
            "callee_path": resolved_path,
            "status": status.value,
            "confidence": confidence.value,
        }

        self.calls.append(call_info)
        self.generic_visit(node)

    def analyze_file(self, file_path: str, module_name: str) -> None:
        """Analyze a single Python file for function calls.

        Parses the file's AST and visits all nodes to discover and record
        function calls. Handles syntax errors gracefully.

        Args:
            file_path: Path to the Python file to analyze.
            module_name: The module name (dot-separated path) for this file.
        """
        self.context.reset(current_module=module_name, current_file=file_path)
        self.import_resolver.reset(current_module=module_name)

        try:
            # Use tokenize.open for automatic encoding detection
            with tokenize.open(file_path) as f:
                source = f.read()

            tree = ast.parse(source, filename=file_path)
            self.visit(tree)

        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Error analyzing {file_path}: {e}")


def load_symbol_table(symbol_table_path: str) -> Dict[str, Any]:
    """Load and validate the symbol table from JSON file.

    Args:
        symbol_table_path: Path to the symbol table JSON file.

    Returns:
        The modules dictionary from the symbol table.

    Raises:
        FileNotFoundError: If the symbol table file doesn't exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the symbol table has an unexpected structure.
    """
    path = Path(symbol_table_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Symbol table not found: {symbol_table_path}\n"
            f"Run symbol_table_builder.py first to generate it."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both old format (flat dict) and new format (with metadata)
    if "modules" in data:
        return data["modules"]
    else:
        # Assume old format
        return data


def run_analysis(
    project_root: str,
    symbol_table_path: str = "project_symbols.json",
    output_path: str = "call_graph.csv",
    output_format: str = "csv",
    status_filter: Optional[str] = None,
    verbose: bool = False,
    quiet: bool = False,
) -> List[CallInfo]:
    """Run complete call graph analysis on a Python project.

    Loads the symbol table, analyzes all project files, and generates
    a comprehensive report of all function calls including their resolution
    status and confidence levels.

    Args:
        project_root: Root directory of the project to analyze.
        symbol_table_path: Path to the JSON symbol table file.
        output_path: Path where the output should be written.
        output_format: Output format: 'csv' or 'json'.
        status_filter: If provided, only output calls with this status.
        verbose: Whether to print detailed progress information.
        quiet: Whether to suppress all non-error output.

    Returns:
        List of call dictionaries with resolution information.

    Raises:
        FileNotFoundError: If symbol table file doesn't exist.
        json.JSONDecodeError: If symbol table is not valid JSON.
        ValueError: If symbol table has unexpected structure.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    elif quiet:
        logger.setLevel(logging.ERROR)

    logger.info("📖 Loading symbol table...")
    symbol_table = load_symbol_table(symbol_table_path)
    analyzer = CallGraphAnalyzer(project_root, symbol_table)

    logger.info("🔍 Analyzing calls...")

    file_count = 0
    for module_name, module_info in symbol_table.items():
        if "error" in module_info:
            continue

        file_path = module_info["file_path"]
        file_count += 1

        logger.debug(f"  📄 {module_name}")
        if not quiet and not verbose and file_count % 10 == 0:
            logger.info(f"  Processed {file_count} files...")

        analyzer.analyze_file(file_path, module_name)

    # Apply filter if specified
    calls = analyzer.calls
    if status_filter:
        calls = [c for c in calls if c["status"] == status_filter]
        logger.info(f"  Filtered to {len(calls)} calls with status='{status_filter}'")

    # Write output
    logger.info(f"💾 Writing results to {output_path}...")

    if output_format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(calls, f, indent=2)
    else:  # csv
        fieldnames = [
            "source_file",
            "source_line",
            "caller_module",
            "caller_name",
            "callee_path",
            "status",
            "confidence",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(calls)

    # Print summary
    if not quiet:
        print_summary(analyzer.calls, status_filter is not None)

    return calls


def print_summary(calls: List[CallInfo], is_filtered: bool = False) -> None:
    """Print analysis summary statistics.

    Args:
        calls: List of all analyzed calls.
        is_filtered: Whether the calls list has been filtered.
    """
    total = len(calls)

    if total == 0:
        logger.warning("\n⚠️  No calls found to analyze.")
        return

    missing = sum(1 for c in calls if c["status"] == Status.MISSING.value)
    uncertain = sum(1 for c in calls if c["status"] == Status.UNCERTAIN.value)
    external = sum(1 for c in calls if c["status"] == Status.EXTERNAL.value)
    exists = sum(1 for c in calls if c["status"] == Status.EXISTS.value)
    unresolvable = sum(1 for c in calls if c["status"] == Status.UNRESOLVABLE.value)

    print("\n" + "=" * 70)
    print("📊 CALL GRAPH ANALYSIS SUMMARY")
    print("=" * 70)

    if is_filtered:
        print(f"Showing filtered results ({total} calls)")
    else:
        print(f"Total calls analyzed: {total}")

    print(f"✅ Verified exists:   {exists:6d} ({exists/total*100:5.1f}%)")
    print(f"🔌 External library:  {external:6d} ({external/total*100:5.1f}%)")
    print(f"❓ Uncertain:         {uncertain:6d} ({uncertain/total*100:5.1f}%)")
    print(f"🔍 Unresolvable:      {unresolvable:6d} ({unresolvable/total*100:5.1f}%)")
    print(f"❌ MISSING:           {missing:6d} ({missing/total*100:5.1f}%)")
    print("=" * 70)

    if missing > 0:
        print(f"\n🚨 Found {missing} potentially missing references!")
        print("\nTop 10 most frequent missing calls:")
        missing_calls = [c for c in calls if c["status"] == Status.MISSING.value]

        # Count occurrences of each missing call
        missing_counts: Dict[str, int] = defaultdict(int)
        for call in missing_calls:
            key = f"{call['caller_name']} → {call['callee_path']}"
            missing_counts[key] += 1

        # Sort by frequency
        sorted_missing = sorted(
            missing_counts.items(), key=lambda x: x[1], reverse=True
        )

        for i, (call_desc, count) in enumerate(sorted_missing[:10], 1):
            print(f"  {i}. {call_desc} ({count} occurrences)")

    # Show module breakdown
    if not is_filtered:
        print("\n📦 Calls by module (top 10):")
        module_counts: Dict[str, int] = defaultdict(int)
        for call in calls:
            if call["caller_module"]:
                module_counts[call["caller_module"]] += 1

        sorted_modules = sorted(module_counts.items(), key=lambda x: x[1], reverse=True)
        for module, count in sorted_modules[:10]:
            print(f"  • {module}: {count} calls")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Analyze Python project call graph and identify missing references.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s .
  %(prog)s /path/to/project --symbols custom_symbols.json
  %(prog)s . --output calls.csv --format json
  %(prog)s . --filter missing --verbose
  %(prog)s . --symbols project_symbols.json --output results.json --format json
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
        default="call_graph.csv",
        help="Output file path (default: call_graph.csv)",
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
        choices=["exists", "missing", "external", "uncertain", "unresolvable"],
        help="Only output calls with this status",
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
    """Main entry point for the call graph analyzer.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_arguments()

    try:
        calls = run_analysis(
            project_root=args.project_root,
            symbol_table_path=args.symbols,
            output_path=args.output,
            output_format=args.format,
            status_filter=args.filter,
            verbose=args.verbose,
            quiet=args.quiet,
        )

        if not args.quiet:
            logger.info(f"\n✅ Analysis complete! Results saved to {args.output}")

        # Exit with error code if missing references found (unless filtered)
        if not args.filter:
            missing_count = sum(1 for c in calls if c["status"] == Status.MISSING.value)
            if missing_count > 0:
                return 1

        return 0

    except FileNotFoundError as e:
        logger.error(f"❌ Error: {e}")
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in symbol table: {e}")
        return 1
    except ValueError as e:
        logger.error(f"❌ Error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.error("\n⚠️  Analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
