# Basic Usage

## Scan current directory (simplest form)
python symbol_table_builder.py

## Scan current directory (explicit)
python symbol_table_builder.py .

## Scan a specific project folder
python symbol_table_builder.py C:\Users\YourName\Projects\MyProject

## Scan a specific project on Mac/Linux
python symbol_table_builder.py /home/username/projects/myproject


# Specifying Output File
## Save to a custom filename
python symbol_table_builder.py . --output my_symbols.json

## Save to a different directory
python symbol_table_builder.py . --output C:\Output\symbols.json

## Short form
python symbol_table_builder.py . -o symbols.json


# Excluding Directories

## Exclude one additional directory
python symbol_table_builder.py . --exclude tests

## Exclude multiple directories (use flag multiple times)
python symbol_table_builder.py . --exclude tests --exclude docs --exclude examples

## Short form
python symbol_table_builder.py . -e tests -e docs

# Including Private Symbols

## Include functions/classes starting with underscore
python symbol_table_builder.py . --include-private

## Short form
python symbol_table_builder.py . -p

# Controlling Output Verbosity

## Verbose mode - see each file being processed
python symbol_table_builder.py . --verbose

## Quiet mode - only show errors
python symbol_table_builder.py . --quiet

## Short forms
python symbol_table_builder.py . -v
python symbol_table_builder.py . -q


# Combining Multiple Options

## Full example with multiple options
python symbol_table_builder.py C:\MyProject --output symbols.json --exclude tests --exclude venv --include-private --verbose

## Another combination
python symbol_table_builder.py . -o output.json -e tests -e docs -p -v

# Getting Help

## Show all available options
python symbol_table_builder.py --help

## Show version
python symbol_table_builder.py --version