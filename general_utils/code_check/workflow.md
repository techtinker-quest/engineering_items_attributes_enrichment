# Step 1: Build the symbol table (include private symbols for completeness)
python symbol_table_builder.py . --output project_symbols.json --include-private -v

# Step 2: Validate all imports (THIS IS THE NEW TOOL - catches your ImportError)
python import_validator.py . --symbols project_symbols.json --output import_validation.csv --format csv -v

# Step 3: Analyze function calls (your existing tool)
python call_graph_analyzer.py . --symbols project_symbols.json --output call_graph.csv --format csv -v

# Optional: Filter to show only problems
python import_validator.py . --filter missing --output missing_imports.csv
python call_graph_analyzer.py . --filter missing --output missing_calls.csv