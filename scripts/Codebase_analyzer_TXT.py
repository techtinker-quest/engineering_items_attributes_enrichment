import os
import ast
from pathlib import Path
from typing import List, Tuple, Any

# --- Configuration ---
ROOT_DIR = Path(os.getcwd())
SEPARATOR = "<~>"  # multi-character separator

# --- AST Helpers ---


def get_docstring_summary(node: Any) -> str:
    docstring = ast.get_docstring(node, clean=True)
    if docstring:
        summary = docstring.split("\n", 1)[0].strip()
        return summary if len(summary) < 100 else summary[:97] + "..."
    return "N/A (Missing Docstring)"


def get_function_arguments(node: ast.FunctionDef) -> str:
    args = []
    for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
        arg_name = arg.arg
        annotation = ast.unparse(arg.annotation) if arg.annotation else None
        args.append(f"{arg_name}: {annotation}" if annotation else arg_name)
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")
    return ", ".join(args)


def get_return_annotation(node: ast.FunctionDef) -> str:
    if node.returns:
        return ast.unparse(node.returns).replace("\n", " ")
    docstring = ast.get_docstring(node, clean=True)
    if docstring and ("return" in docstring.lower() or "yield" in docstring.lower()):
        return "Annotated (Check Docstring)"
    return "None (Inferred/Missing)"


def set_parents(node: ast.AST, parent: ast.AST = None):
    """Recursively set 'parent' attribute for all child nodes."""
    for child in ast.iter_child_nodes(node):
        setattr(child, "parent", node)
        set_parents(child, node)


# --- Extraction ---


def extract_info_from_file(
    file_path: Path,
) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    class_info, method_info, function_info = [], [], []

    try:
        content = file_path.read_text(encoding="utf-8-sig")  # strip BOM
        tree = ast.parse(content)
        set_parents(tree)

        relative_path = file_path.relative_to(ROOT_DIR)
        folder = (
            str(relative_path.parent) + "/"
            if str(relative_path.parent) != "."
            else "./"
        )
        file_name = relative_path.name

        for node in ast.walk(tree):
            # --- Classes ---
            if isinstance(node, ast.ClassDef):
                bases = [ast.unparse(b).replace("\n", " ") for b in node.bases]
                class_info.append(
                    [
                        folder,
                        file_name,
                        node.name,
                        f"Inherits: {', '.join(bases)}" if bases else "None",
                        get_docstring_summary(node),
                        "N/A (See Imports)",
                    ]
                )

                # --- Methods ---
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if (
                            item.name.startswith("__")
                            and get_docstring_summary(item) == "N/A (Missing Docstring)"
                        ):
                            continue
                        method_info.append(
                            [
                                folder,
                                file_name,
                                node.name,
                                item.name,
                                get_function_arguments(item),
                                get_return_annotation(item),
                                get_docstring_summary(item),
                            ]
                        )

            # --- Global functions ---
            elif isinstance(node, ast.FunctionDef):
                parent_is_class = hasattr(node, "parent") and isinstance(
                    node.parent, ast.ClassDef
                )
                if not parent_is_class:
                    if (
                        node.name == "main"
                        and get_docstring_summary(node) == "N/A (Missing Docstring)"
                    ):
                        continue
                    function_info.append(
                        [
                            folder,
                            file_name,
                            node.name,
                            get_function_arguments(node),
                            get_return_annotation(node),
                            get_docstring_summary(node),
                        ]
                    )

    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")

    return class_info, method_info, function_info


# --- Project Analysis ---


def analyze_project_structure(
    root_dir: Path,
) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    all_class_info, all_method_info, all_function_info = [], [], []
    print(f"Analyzing Python files in directory: {root_dir}")

    for file_path in root_dir.rglob("*.py"):
        if any(part.startswith(".") for part in file_path.parts) or any(
            part in ("venv", ".venv", "env", "node_modules") for part in file_path.parts
        ):
            continue
        class_data, method_data, function_data = extract_info_from_file(file_path)
        all_class_info.extend(class_data)
        all_method_info.extend(method_data)
        all_function_info.extend(function_data)

    return all_class_info, all_method_info, all_function_info


# --- TXT Writing ---


def write_data_to_txt(filepath: str, header: List[str], data: List[List[str]]):
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(SEPARATOR.join(header) + "\n")
            for row in data:
                f.write(SEPARATOR.join(row) + "\n")
        print(f"Successfully generated: {filepath}")
    except IOError as e:
        print(f"Error writing to file {filepath}: {e}")


def generate_documentation_txts():
    class_data, method_data, function_data = analyze_project_structure(ROOT_DIR)

    write_data_to_txt(
        "class_info.txt",
        [
            "Folder",
            "File Name",
            "Class Name",
            "Class Parameters (Inheritance)",
            "Class Responsibility / Purpose",
            "Dependencies (General)",
        ],
        class_data,
    )

    write_data_to_txt(
        "method_info.txt",
        [
            "Folder",
            "File Name",
            "Class Name",
            "Method Name",
            "Method Arguments",
            "Method Return",
            "Method Responsibility",
        ],
        method_data,
    )

    write_data_to_txt(
        "function_info.txt",
        [
            "Folder",
            "File Name",
            "Function Name",
            "Function Arguments",
            "Function Return",
            "Function Responsibility",
        ],
        function_data,
    )


if __name__ == "__main__":
    generate_documentation_txts()
