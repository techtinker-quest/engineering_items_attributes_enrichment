"""All-in-One Python Project Analyzer.

A complete static analysis tool that builds symbol tables, validates imports,
and analyzes call graphs to identify missing references in Python projects.

Usage:
    python project_analyzer.py [PROJECT_PATH] [OPTIONS]

Examples:
    python project_analyzer.py .
    python project_analyzer.py . --output-dir analysis --verbose
    python project_analyzer.py . --filter missing --format json
    python project_analyzer.py . --save-symbols --exclude tests
"""

import ast
import json
import csv
import sys
import argparse
import logging
import tokenize
import difflib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from importlib.util import resolve_name
from functools import lru_cache
from collections import defaultdict

__version__ = "2.0.0"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Default directories to exclude
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


class Status(str, Enum):
    """Status of validation results."""

    EXISTS = "exists"
    MISSING = "missing"
    EXTERNAL = "external"
    UNCERTAIN = "uncertain"
    UNRESOLVABLE = "unresolvable"


class Confidence(str, Enum):
    """Confidence level for call resolution."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNRESOLVABLE = "unresolvable"


@dataclass
class ImportInfo:
    """Information about an import statement."""

    source_file: str
    source_line: int
    import_type: str
    module: str
    imported_name: Optional[str]
    alias: Optional[str]
    status: str
    details: str


@dataclass
class CallInfo:
    """Information about a function call."""

    source_file: str
    source_line: int
    caller_module: str
    caller_name: str
    callee_path: str
    status: str
    confidence: str


# ============================================================================
# SYMBOL TABLE BUILDER
# ============================================================================


class SymbolExtractor(ast.NodeVisitor):
    """AST visitor that extracts symbols from Python source code."""

    def __init__(self, include_private: bool = False):
        self.include_private = include_private
        self.module_docstring: Optional[str] = None
        self.classes: Dict[str, Dict[str, Any]] = {}
        self.functions: List[Dict[str, Any]] = []
        self.current_class: Optional[str] = None
        self.class_stack: List[str] = []

    def should_include(self, name: str) -> bool:
        """Check if a symbol should be included."""
        if self.include_private:
            return True
        return not name.startswith("_") or (
            name.startswith("__") and name.endswith("__")
        )

    @staticmethod
    def get_docstring(node: ast.AST) -> Optional[str]:
        """Extract docstring from an AST node."""
        return ast.get_docstring(node)

    @staticmethod
    def get_function_signature(node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract function signature."""
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
        """Determine method type."""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                if decorator.id == "staticmethod":
                    return "static"
                elif decorator.id == "classmethod":
                    return "class"
        return "instance"

    @staticmethod
    def get_base_classes(node: ast.ClassDef) -> List[str]:
        """Extract base class names."""
        bases = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except Exception:
                bases.append("<unparseable>")
        return bases

    def visit_Module(self, node: ast.Module) -> None:
        """Visit module node."""
        self.module_docstring = self.get_docstring(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        if not self.should_include(node.name):
            return

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

        previous_class = self.current_class
        self.current_class = full_name
        self.class_stack.append(node.name)

        self.generic_visit(node)

        self.class_stack.pop()
        self.current_class = previous_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition."""
        if not self.should_include(node.name):
            return

        func_info = {
            "name": node.name,
            "line": node.lineno,
            "docstring": self.get_docstring(node),
            "signature": self.get_function_signature(node),
        }

        if self.current_class:
            func_info["method_type"] = self.get_method_type(node)
            self.classes[self.current_class]["methods"].append(func_info)
        else:
            self.functions.append(func_info)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition."""
        self.visit_FunctionDef(node)  # type: ignore


class SymbolTableBuilder:
    """Builds a symbol table from a Python project."""

    def __init__(
        self,
        project_root: str,
        excluded_dirs: Set[str],
        include_private: bool = False,
    ):
        self.project_root = Path(project_root).resolve()
        self.excluded_dirs = excluded_dirs
        self.include_private = include_private

    @staticmethod
    def get_module_path(base_path: Path, file_path: Path) -> str:
        """Convert file path to Python module path."""
        try:
            relative = file_path.relative_to(base_path)
        except ValueError:
            return ""

        if relative.suffix == ".py":
            relative = relative.with_suffix("")

        parts = list(relative.parts)

        if parts and parts[-1] == "__init__":
            parts.pop()

        return ".".join(parts)

    def process_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Process a single Python file."""
        module_name = self.get_module_path(self.project_root, file_path)

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
            try:
                source_code = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                source_code = file_path.read_text(encoding="latin-1")

            tree = ast.parse(source_code, filename=str(file_path))

            extractor = SymbolExtractor(include_private=self.include_private)
            extractor.visit(tree)

            result["module_docstring"] = extractor.module_docstring
            result["classes"] = extractor.classes
            result["functions"] = extractor.functions

            return result

        except SyntaxError as e:
            result["error"] = f"Syntax error at line {e.lineno}: {e.msg}"
            return result
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {str(e)}"
            return result

    def build(self) -> Dict[str, Any]:
        """Build the complete symbol table."""
        if not self.project_root.exists():
            raise OSError(f"Project root does not exist: {self.project_root}")
        if not self.project_root.is_dir():
            raise OSError(f"Project root is not a directory: {self.project_root}")

        modules: Dict[str, Dict[str, Any]] = {}
        error_count = 0
        file_count = 0

        logger.info(f"🔎 Scanning {self.project_root}...")

        for py_file in self.project_root.rglob("*.py"):
            if any(excluded in py_file.parts for excluded in self.excluded_dirs):
                continue

            file_count += 1
            logger.debug(f"   Processing: {py_file.relative_to(self.project_root)}")

            file_result = self.process_file(py_file)

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

        symbol_table = {
            "metadata": {
                "tool": "All-in-One Python Project Analyzer",
                "version": __version__,
                "scan_time": datetime.now().isoformat(),
                "project_root": str(self.project_root),
                "files_scanned": file_count,
                "errors": error_count,
                "include_private": self.include_private,
            },
            "modules": modules,
        }

        logger.info(
            f"✅ Built symbol table: {file_count} files, {len(modules)} modules"
        )
        if error_count > 0:
            logger.warning(f"⚠️  {error_count} files had errors")

        return symbol_table


# ============================================================================
# IMPORT RESOLVER & CONTEXT TRACKER
# ============================================================================


class ImportResolver:
    """Handles import statement resolution and tracking."""

    def __init__(self, current_module: Optional[str] = None):
        self.current_module = current_module
        self.imports: Dict[str, str] = {}
        self.from_imports: Dict[str, str] = {}
        self.star_imports: List[str] = []

    def reset(self, current_module: Optional[str] = None) -> None:
        self.current_module = current_module
        self.imports.clear()
        self.from_imports.clear()
        self.star_imports.clear()

    def handle_import(self, node: ast.Import) -> None:
        for alias in node.names:
            module_name = alias.name
            asname = alias.asname if alias.asname else module_name
            self.imports[asname] = module_name

    def handle_from_import(self, node: ast.ImportFrom) -> None:
        module = node.module if node.module else ""

        try:
            if node.level > 0:
                resolved_module = resolve_name(
                    f"{'.' * node.level}{module}", self.current_module or ""
                )
            else:
                resolved_module = module
        except (ValueError, ImportError):
            resolved_module = module

        for alias in node.names:
            name = alias.name
            asname = alias.asname if alias.asname else name

            if name == "*":
                self.star_imports.append(resolved_module)
            elif resolved_module:
                self.from_imports[asname] = f"{resolved_module}.{name}"
            else:
                self.from_imports[asname] = name

    def resolve_name(self, name: str) -> Optional[str]:
        if name in self.from_imports:
            return self.from_imports[name]
        if name in self.imports:
            return self.imports[name]
        return None


class ContextTracker:
    """Tracks analysis context (module, class, function)."""

    def __init__(self):
        self.current_module: Optional[str] = None
        self.current_file: Optional[str] = None
        self.class_stack: List[str] = []
        self.current_function: Optional[str] = None

    def reset(
        self, current_module: Optional[str] = None, current_file: Optional[str] = None
    ) -> None:
        self.current_module = current_module
        self.current_file = current_file
        self.class_stack.clear()
        self.current_function = None

    def enter_class(self, class_name: str) -> None:
        self.class_stack.append(class_name)

    def exit_class(self) -> None:
        if self.class_stack:
            self.class_stack.pop()

    def enter_function(self, function_name: str) -> None:
        self.current_function = function_name

    def exit_function(self) -> None:
        self.current_function = None

    @property
    def current_class(self) -> Optional[str]:
        return ".".join(self.class_stack) if self.class_stack else None

    @property
    def caller_name(self) -> str:
        if self.current_class and self.current_function:
            return f"{self.current_class}.{self.current_function}"
        elif self.current_function:
            return self.current_function
        else:
            return f"<module:{self.current_module}>"


# ============================================================================
# UNIFIED ANALYZER
# ============================================================================


class UnifiedAnalyzer(ast.NodeVisitor):
    """AST visitor that validates imports and analyzes function calls."""

    def __init__(self, project_root: str, symbol_table: Dict[str, Any]):
        self.project_root = project_root
        self.symbol_table = symbol_table
        self.imports: List[ImportInfo] = []
        self.calls: List[CallInfo] = []

        self.import_resolver = ImportResolver()
        self.context = ContextTracker()

        self._build_lookup_caches()

    def _build_lookup_caches(self) -> None:
        """Build lookup caches for performance."""
        self.module_functions: Dict[str, Set[str]] = {}
        self.module_classes: Dict[str, Set[str]] = {}

        for module_name, module_info in self.symbol_table.items():
            if "error" in module_info:
                continue

            func_names = {f["name"] for f in module_info.get("functions", [])}
            self.module_functions[module_name] = func_names

            class_names = set(module_info.get("classes", {}).keys())
            self.module_classes[module_name] = class_names

    @lru_cache(maxsize=512)
    def _resolve_module(self, module: str, level: int) -> str:
        """Resolve relative imports."""
        try:
            if level > 0:
                return resolve_name(
                    f"{'.' * level}{module}", self.context.current_module or ""
                )
            return module
        except (ValueError, ImportError):
            return module

    def _suggest_alternatives(self, name: str, module: str) -> str:
        """Suggest alternative names if symbol is missing."""
        if module not in self.symbol_table:
            return ""

        available = []
        available.extend(self.module_functions.get(module, []))
        available.extend(self.module_classes.get(module, []))

        if not available:
            return ""

        matches = difflib.get_close_matches(name, available, n=3, cutoff=0.6)
        if matches:
            return f" (Did you mean: {', '.join(matches)}?)"
        return ""

    def _check_builtin_or_stdlib(self, module_name: str) -> Tuple[bool, str]:
        """Check if module is built-in or stdlib."""
        base_module = module_name.split(".")[0]

        if base_module in sys.builtin_module_names:
            return True, f"Built-in module: {base_module}"

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
            return True, f"Standard library: {base_module}"

        return False, ""

    # Import Validation Methods
    def _validate_import_statement(self, node: ast.Import) -> None:
        """Validate import statement."""
        for alias in node.names:
            module_name = alias.name
            asname = alias.asname

            is_builtin, builtin_details = self._check_builtin_or_stdlib(module_name)

            if is_builtin:
                status = Status.EXTERNAL
                details = builtin_details
            elif module_name in self.symbol_table:
                status = Status.EXISTS
                details = "Module found in project"
            else:
                parts = module_name.split(".")
                found = False
                for i in range(len(parts), 0, -1):
                    partial = ".".join(parts[:i])
                    if partial in self.symbol_table:
                        status = Status.EXISTS
                        details = f"Parent module '{partial}' found"
                        found = True
                        break

                if not found:
                    base_module = parts[0]
                    project_modules = {
                        m.split(".")[0] for m in self.symbol_table.keys()
                    }
                    if base_module not in project_modules:
                        status = Status.EXTERNAL
                        details = "External library"
                    else:
                        status = Status.MISSING
                        details = f"Module '{module_name}' not found"

            self.imports.append(
                ImportInfo(
                    source_file=self.context.current_file or "",
                    source_line=node.lineno,
                    import_type="import",
                    module=module_name,
                    imported_name=None,
                    alias=asname,
                    status=status.value,
                    details=details,
                )
            )

    def _validate_from_import_statement(self, node: ast.ImportFrom) -> None:
        """Validate from-import statement."""
        module = node.module if node.module else ""
        resolved_module = self._resolve_module(module, node.level)

        for alias in node.names:
            name = alias.name
            asname = alias.asname

            if name == "*":
                status = Status.EXTERNAL
                details = "Star import from external library"

                if resolved_module in self.symbol_table:
                    status = Status.UNCERTAIN
                    details = "Star import (symbols not validated)"
                elif not resolved_module:
                    status = Status.MISSING
                    details = "Star import from unresolved module"

                self.imports.append(
                    ImportInfo(
                        source_file=self.context.current_file or "",
                        source_line=node.lineno,
                        import_type="from_import_star",
                        module=resolved_module,
                        imported_name="*",
                        alias=None,
                        status=status.value,
                        details=details,
                    )
                )
                continue

            status, details = self._check_symbol_in_module(name, resolved_module)

            self.imports.append(
                ImportInfo(
                    source_file=self.context.current_file or "",
                    source_line=node.lineno,
                    import_type="from_import",
                    module=resolved_module,
                    imported_name=name,
                    alias=asname,
                    status=status.value,
                    details=details,
                )
            )

    def _check_symbol_in_module(
        self, name: str, resolved_module: str
    ) -> Tuple[Status, str]:
        """Check if symbol exists in module."""
        if resolved_module not in self.symbol_table:
            is_builtin, builtin_details = self._check_builtin_or_stdlib(resolved_module)
            if is_builtin:
                return Status.EXTERNAL, builtin_details

            base_module = resolved_module.split(".")[0]
            project_modules = {m.split(".")[0] for m in self.symbol_table.keys()}

            if base_module not in project_modules:
                return Status.EXTERNAL, "External library"
            else:
                return Status.MISSING, f"Module '{resolved_module}' not found"

        if name in self.module_classes.get(resolved_module, set()):
            return Status.EXISTS, f"Class '{name}' found"

        if name in self.module_functions.get(resolved_module, set()):
            return Status.EXISTS, f"Function '{name}' found"

        potential_submodule = f"{resolved_module}.{name}"
        if potential_submodule in self.symbol_table:
            return Status.EXISTS, f"Submodule found"

        module_info = self.symbol_table[resolved_module]
        for class_name, class_info in module_info.get("classes", {}).items():
            for method in class_info.get("methods", []):
                if method["name"] == name:
                    return Status.UNCERTAIN, f"'{name}' is a method (unusual)"

        suggestion = self._suggest_alternatives(name, resolved_module)
        return Status.MISSING, f"Symbol '{name}' NOT FOUND{suggestion}"

    # Call Graph Analysis Methods
    def _resolve_call_target(self, call_node: ast.Call) -> Tuple[str, Confidence]:
        """Resolve function call target."""
        if isinstance(call_node.func, ast.Name):
            func_name = call_node.func.id

            resolved = self.import_resolver.resolve_name(func_name)
            if resolved:
                return resolved, Confidence.HIGH

            if (
                self.context.current_module
                and self.context.current_module in self.symbol_table
            ):
                module_info = self.symbol_table[self.context.current_module]

                if any(
                    f["name"] == func_name for f in module_info.get("functions", [])
                ):
                    return f"{self.context.current_module}.{func_name}", Confidence.HIGH

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

            if func_name in dir(__builtins__):
                return f"<builtin>.{func_name}", Confidence.HIGH

            return func_name, Confidence.LOW

        elif isinstance(call_node.func, ast.Attribute):
            attr_name = call_node.func.attr
            value = call_node.func.value

            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                if value.func.id == "super" and self.context.current_class:
                    return (
                        f"<super:{self.context.current_module}.{self.context.current_class}>.{attr_name}",
                        Confidence.MEDIUM,
                    )

            if isinstance(value, ast.Name):
                base_name = value.id

                resolved = self.import_resolver.resolve_name(base_name)
                if resolved:
                    return f"{resolved}.{attr_name}", Confidence.MEDIUM

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

                if base_name == "self" and self.context.current_class:
                    return (
                        f"{self.context.current_module}.{self.context.current_class}.{attr_name}",
                        Confidence.HIGH,
                    )

                return f"{base_name}.{attr_name}", Confidence.LOW

            return f"<complex>.{attr_name}", Confidence.LOW

        return "<unresolvable>", Confidence.UNRESOLVABLE

    def _check_call_existence(
        self, resolved_path: str, confidence: Confidence
    ) -> Status:
        """Check if call target exists."""
        if confidence == Confidence.UNRESOLVABLE:
            return Status.UNRESOLVABLE

        if resolved_path.startswith("<builtin>"):
            return Status.EXISTS

        if resolved_path.startswith("<super:"):
            return Status.UNCERTAIN

        if resolved_path.startswith("<"):
            return Status.UNCERTAIN

        parts = resolved_path.split(".")

        if len(parts) >= 3:
            for i in range(len(parts) - 2, 0, -1):
                mod_path = ".".join(parts[:i])
                if mod_path in self.symbol_table:
                    class_path = ".".join(parts[i:-1])
                    method_name = parts[-1]

                    class_info = self.symbol_table[mod_path]["classes"].get(class_path)
                    if class_info and any(
                        m["name"] == method_name for m in class_info.get("methods", [])
                    ):
                        return Status.EXISTS

        if len(parts) >= 2:
            mod_path = ".".join(parts[:-1])
            name = parts[-1]
            if mod_path in self.symbol_table:
                module_info = self.symbol_table[mod_path]
                if any(f["name"] == name for f in module_info.get("functions", [])):
                    return Status.EXISTS
                if name in module_info.get("classes", {}):
                    return Status.EXISTS

        if parts[0] not in [m.split(".")[0] for m in self.symbol_table.keys()]:
            return Status.EXTERNAL

        return Status.MISSING

    # AST Visitor Methods
    def visit_Import(self, node: ast.Import) -> None:
        self.import_resolver.handle_import(node)
        self._validate_import_statement(node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.import_resolver.handle_from_import(node)
        self._validate_from_import_statement(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.context.enter_class(node.name)
        self.generic_visit(node)
        self.context.exit_class()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.context.enter_function(node.name)
        self.generic_visit(node)
        self.context.exit_function()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.context.enter_function(node.name)
        self.generic_visit(node)
        self.context.exit_function()

    def visit_Call(self, node: ast.Call) -> None:
        resolved_path, confidence = self._resolve_call_target(node)
        status = self._check_call_existence(resolved_path, confidence)

        self.calls.append(
            CallInfo(
                source_file=self.context.current_file or "",
                source_line=node.lineno,
                caller_module=self.context.current_module or "",
                caller_name=self.context.caller_name,
                callee_path=resolved_path,
                status=status.value,
                confidence=confidence.value,
            )
        )
        self.generic_visit(node)

    def analyze_file(self, file_path: str, module_name: str) -> None:
        """Analyze a single file."""
        self.context.reset(current_module=module_name, current_file=file_path)
        self.import_resolver.reset(current_module=module_name)

        try:
            with tokenize.open(file_path) as f:
                source = f.read()

            tree = ast.parse(source, filename=file_path)
            self.visit(tree)

        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Error analyzing {file_path}: {e}")


# ============================================================================
# OUTPUT & REPORTING
# ============================================================================


def write_results(
    symbol_table: Dict[str, Any],
    imports: List[ImportInfo],
    calls: List[CallInfo],
    output_dir: str,
    output_format: str,
    import_filter: Optional[str],
    call_filter: Optional[str],
    save_symbols: bool,
) -> None:
    """Write analysis results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Filter if needed
    filtered_imports = imports
    if import_filter:
        filtered_imports = [i for i in imports if i.status == import_filter]

    filtered_calls = calls
    if call_filter:
        filtered_calls = [c for c in calls if c.status == call_filter]

    # Write symbol table if requested
    if save_symbols:
        symbols_file = output_path / "symbols.json"
        with open(symbols_file, "w", encoding="utf-8") as f:
            json.dump(symbol_table, f, indent=2)
        logger.info(f"💾 Symbol table: {symbols_file}")

    # Write imports
    if output_format == "json":
        import_file = output_path / "imports.json"
        with open(import_file, "w", encoding="utf-8") as f:
            json.dump([asdict(i) for i in filtered_imports], f, indent=2)
        logger.info(f"💾 Imports: {import_file}")
    else:
        import_file = output_path / "imports.csv"
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
        with open(import_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([asdict(i) for i in filtered_imports])
        logger.info(f"💾 Imports: {import_file}")

    # Write calls
    if output_format == "json":
        call_file = output_path / "calls.json"
        with open(call_file, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in filtered_calls], f, indent=2)
        logger.info(f"💾 Calls: {call_file}")
    else:
        call_file = output_path / "calls.csv"
        fieldnames = [
            "source_file",
            "source_line",
            "caller_module",
            "caller_name",
            "callee_path",
            "status",
            "confidence",
        ]
        with open(call_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([asdict(c) for c in filtered_calls])
        logger.info(f"💾 Calls: {call_file}")


def print_summary(
    symbol_table: Dict[str, Any],
    imports: List[ImportInfo],
    calls: List[CallInfo],
    import_filter: bool,
    call_filter: bool,
) -> None:
    """Print analysis summary."""
    metadata = symbol_table.get("metadata", {})

    logger.info("\n" + "=" * 70)
    logger.info("📊 PYTHON PROJECT ANALYSIS SUMMARY")
    logger.info("=" * 70)

    # Symbol table stats
    modules = symbol_table.get("modules", {})
    total_classes = sum(len(m.get("classes", {})) for m in modules.values())
    total_functions = sum(len(m.get("functions", [])) for m in modules.values())
    total_methods = sum(
        len(class_info.get("methods", []))
        for module in modules.values()
        for class_info in module.get("classes", {}).values()
    )

    logger.info(f"\n📚 PROJECT STRUCTURE:")
    logger.info(f"   • Modules: {len(modules)}")
    logger.info(f"   • Classes: {total_classes}")
    logger.info(f"   • Functions: {total_functions}")
    logger.info(f"   • Methods: {total_methods}")
    logger.info(f"   • Files scanned: {metadata.get('files_scanned', 0)}")
    if metadata.get("errors", 0) > 0:
        logger.info(f"   ⚠️  Parse errors: {metadata['errors']}")

    # Import summary
    total_imports = len(imports)
    if total_imports > 0:
        missing_imports = sum(1 for i in imports if i.status == Status.MISSING.value)
        uncertain_imports = sum(
            1 for i in imports if i.status == Status.UNCERTAIN.value
        )
        external_imports = sum(1 for i in imports if i.status == Status.EXTERNAL.value)
        exists_imports = sum(1 for i in imports if i.status == Status.EXISTS.value)

        logger.info(
            f"\n📥 IMPORTS: {total_imports} analyzed"
            + (" (filtered)" if import_filter else "")
        )
        logger.info(
            f"   ✅ Exists:    {exists_imports:6d} ({exists_imports/total_imports*100:5.1f}%)"
        )
        logger.info(
            f"   🔌 External:  {external_imports:6d} ({external_imports/total_imports*100:5.1f}%)"
        )
        logger.info(
            f"   ❓ Uncertain: {uncertain_imports:6d} ({uncertain_imports/total_imports*100:5.1f}%)"
        )
        logger.info(
            f"   ❌ MISSING:   {missing_imports:6d} ({missing_imports/total_imports*100:5.1f}%)"
        )

        if missing_imports > 0:
            logger.info(f"\n🚨 MISSING IMPORTS ({missing_imports} found):")
            for imp in [i for i in imports if i.status == Status.MISSING.value][:10]:
                logger.info(f"   📍 {imp.source_file}:{imp.source_line}")
                if imp.imported_name:
                    logger.info(f"      from {imp.module} import {imp.imported_name}")
                else:
                    logger.info(f"      import {imp.module}")
                logger.info(f"      ⚠️  {imp.details}")
                logger.info("")
            if missing_imports > 10:
                logger.info(f"   ... and {missing_imports - 10} more (see imports.csv)")

    # Call summary
    total_calls = len(calls)
    if total_calls > 0:
        missing_calls = sum(1 for c in calls if c.status == Status.MISSING.value)
        uncertain_calls = sum(1 for c in calls if c.status == Status.UNCERTAIN.value)
        external_calls = sum(1 for c in calls if c.status == Status.EXTERNAL.value)
        exists_calls = sum(1 for c in calls if c.status == Status.EXISTS.value)
        unresolvable_calls = sum(
            1 for c in calls if c.status == Status.UNRESOLVABLE.value
        )

        logger.info(
            f"\n📞 CALLS: {total_calls} analyzed"
            + (" (filtered)" if call_filter else "")
        )
        logger.info(
            f"   ✅ Exists:       {exists_calls:6d} ({exists_calls/total_calls*100:5.1f}%)"
        )
        logger.info(
            f"   🔌 External:     {external_calls:6d} ({external_calls/total_calls*100:5.1f}%)"
        )
        logger.info(
            f"   ❓ Uncertain:    {uncertain_calls:6d} ({uncertain_calls/total_calls*100:5.1f}%)"
        )
        logger.info(
            f"   🔍 Unresolvable: {unresolvable_calls:6d} ({unresolvable_calls/total_calls*100:5.1f}%)"
        )
        logger.info(
            f"   ❌ MISSING:      {missing_calls:6d} ({missing_calls/total_calls*100:5.1f}%)"
        )

        if missing_calls > 0:
            logger.info(f"\n🚨 TOP MISSING CALLS:")
            missing_call_list = [c for c in calls if c.status == Status.MISSING.value]
            call_counts: Dict[str, int] = defaultdict(int)
            for call in missing_call_list:
                key = f"{call.caller_name} → {call.callee_path}"
                call_counts[key] += 1

            for i, (call_desc, count) in enumerate(
                sorted(call_counts.items(), key=lambda x: x[1], reverse=True)[:10], 1
            ):
                logger.info(f"   {i}. {call_desc} ({count}x)")

    logger.info("=" * 70)

    # Overall health
    total_issues = sum(1 for i in imports if i.status == Status.MISSING.value) + sum(
        1 for c in calls if c.status == Status.MISSING.value
    )

    if total_issues == 0:
        logger.info("\n✅ No missing references found! Project looks healthy.")
    else:
        logger.info(f"\n⚠️  Total missing references: {total_issues}")
        logger.info(f"   💡 Review the output files for details and suggested fixes.")


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================


def run_analysis(
    project_root: str,
    excluded_dirs: Set[str],
    output_dir: str = "analysis_results",
    output_format: str = "csv",
    import_filter: Optional[str] = None,
    call_filter: Optional[str] = None,
    include_private: bool = False,
    save_symbols: bool = False,
    verbose: bool = False,
    quiet: bool = False,
) -> Tuple[Dict[str, Any], List[ImportInfo], List[CallInfo]]:
    """Run complete project analysis."""
    if verbose:
        logger.setLevel(logging.DEBUG)
    elif quiet:
        logger.setLevel(logging.ERROR)

    # Step 1: Build symbol table
    logger.info("=" * 70)
    logger.info("🔧 STEP 1: Building Symbol Table")
    logger.info("=" * 70)

    builder = SymbolTableBuilder(project_root, excluded_dirs, include_private)
    symbol_table = builder.build()
    modules = symbol_table.get("modules", {})

    # Step 2: Analyze imports and calls
    logger.info("\n" + "=" * 70)
    logger.info("🔧 STEP 2: Analyzing Imports & Calls")
    logger.info("=" * 70)

    analyzer = UnifiedAnalyzer(project_root, modules)

    file_count = 0
    for module_name, module_info in modules.items():
        if "error" in module_info:
            continue

        file_path = module_info["file_path"]
        file_count += 1

        logger.debug(f"  📄 {module_name}")
        if not quiet and not verbose and file_count % 10 == 0:
            logger.info(f"  Processed {file_count} files...")

        analyzer.analyze_file(file_path, module_name)

    logger.info(f"✅ Analyzed {file_count} files")

    # Step 3: Write results
    logger.info("\n" + "=" * 70)
    logger.info("🔧 STEP 3: Writing Results")
    logger.info("=" * 70)

    write_results(
        symbol_table,
        analyzer.imports,
        analyzer.calls,
        output_dir,
        output_format,
        import_filter,
        call_filter,
        save_symbols,
    )

    # Print summary
    if not quiet:
        print_summary(
            symbol_table,
            analyzer.imports,
            analyzer.calls,
            import_filter is not None,
            call_filter is not None,
        )

    return symbol_table, analyzer.imports, analyzer.calls


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="All-in-one Python project analyzer for symbols, imports, and calls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s .
  %(prog)s /path/to/project --output-dir results
  %(prog)s . --format json --save-symbols
  %(prog)s . --filter missing --verbose
  %(prog)s . --exclude tests --exclude migrations
  %(prog)s . --include-private --import-filter missing
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
        "--output-dir",
        default="analysis_results",
        help="Output directory for results (default: analysis_results)",
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
        choices=["missing", "exists", "external", "uncertain"],
        help="Filter both imports and calls by status (shortcut)",
    )

    parser.add_argument(
        "--import-filter",
        choices=["exists", "missing", "external", "uncertain"],
        help="Filter imports by status",
    )

    parser.add_argument(
        "--call-filter",
        choices=["exists", "missing", "external", "uncertain", "unresolvable"],
        help="Filter calls by status",
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
        "--save-symbols",
        action="store_true",
        help="Save the symbol table to symbols.json",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed progress information",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress all non-error output",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    # Combine default and user exclusions
    excluded_dirs = DEFAULT_EXCLUDED_DIRS.union(set(args.exclude))

    # Handle shortcut filter
    import_filter = args.import_filter or args.filter
    call_filter = args.call_filter or args.filter

    try:
        symbol_table, imports, calls = run_analysis(
            project_root=args.project_root,
            excluded_dirs=excluded_dirs,
            output_dir=args.output_dir,
            output_format=args.format,
            import_filter=import_filter,
            call_filter=call_filter,
            include_private=args.include_private,
            save_symbols=args.save_symbols,
            verbose=args.verbose,
            quiet=args.quiet,
        )

        if not args.quiet:
            logger.info(f"\n✅ Analysis complete! Results in: {args.output_dir}/")

        # Exit with error code if missing references found
        if not import_filter and not call_filter:
            missing_imports = sum(
                1 for i in imports if i.status == Status.MISSING.value
            )
            missing_calls = sum(1 for c in calls if c.status == Status.MISSING.value)

            if missing_imports > 0 or missing_calls > 0:
                return 1

        return 0

    except OSError as e:
        logger.error(f"❌ Error: {e}")
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON: {e}")
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
