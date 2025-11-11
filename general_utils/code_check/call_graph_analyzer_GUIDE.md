# 📋 Command Prompt Usage Examples (Updated)


## Basic workflow
```bash
python symbol_table_builder.py .
python call_graph_analyzer.py .

# The analyzer handles complex cases better:
# ✓ super().method() calls are tracked
# ✓ Nested classes work correctly
# ✓ Chained module.submodule.func() calls resolve better
# ✓ Star imports are detected with warnings
```

## **Viewing Enhanced Warnings**
```bash
# Verbose mode now shows star import warnings
python call_graph_analyzer.py . --verbose

# Output will include warnings like:
# ⚠️  Star import detected: 'from module import *' in mypackage.utils
```

## **Better Error Handling**

```bash
# The tool now handles files with non-UTF-8 encoding automatically
python call_graph_analyzer.py . -v

# Files with special characters are handled gracefully
# Latin-1, UTF-16, etc. are detected automatically using tokenize.open()
```

## **All Previous Examples Still Work**

```bash
# Standard usage
python call_graph_analyzer.py .
python call_graph_analyzer.py . -s symbols.json -o calls.csv

# Filtering
python call_graph_analyzer.py . --filter missing --verbose

# JSON output
python call_graph_analyzer.py . -o analysis.json -f json

# Quiet mode for CI/CD
python call_graph_analyzer.py . -q
```

### **Complete Workflow with Both Tools**

```bash
# Step 1: Build symbol table with private symbols
python symbol_table_builder.py . --output project_symbols.json -v
python symbol_table_builder.py . --include-private --output symbols.json -v             #Or something like this

# Step 2: Analyze call graph with detailed output
python call_graph_analyzer.py . --symbols project_symbols.json --output calls.json --format csv -v

# Step 3: Check only for missing references
python call_graph_analyzer.py . --filter missing --output missing.csv
```

### **CI/CD Integration Example**

```bash
# Exit with error code if missing references found
python symbol_table_builder.py . -q
python call_graph_analyzer.py . -q

# Check exit code ($? on Unix, %ERRORLEVEL% on Windows)
# Returns 1 if missing references found, 0 otherwise
```