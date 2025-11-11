# Code Analyzer
Builds symbol tables, validates imports, and analyzes call graphs to identify missing references in Python projects.


#🎯 Usage Examples
## Basic analysis - default settings
python code_analyzer.py .

## Show only problems
python code_analyzer.py . --filter missing

## Save everything including symbol table
python code_analyzer.py . --save-symbols --format json

## Custom output location
python code_analyzer.py . --output-dir my_analysis

## Include private symbols and exclude directories
python code_analyzer.py . --include-private --exclude tests --exclude docs

## Verbose mode to see progress
python code_analyzer.py . --verbose

## Filter imports and calls separately
python code_analyzer.py . --import-filter missing --call-filter missing



## 📂 Output Structure
```
analysis_results/
├── imports.csv          # Import validation results
├── calls.csv           # Call graph analysis results
└── symbols.json        # (optional with --save-symbols)
```