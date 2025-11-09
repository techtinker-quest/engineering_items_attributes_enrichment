"""
CLI Interface Module

Provides command-line interface for the Drawing Intelligence System, supporting
single drawing processing, batch operations, exports, cost reporting, and
system configuration validation.

This module serves as the primary entry point for all CLI operations and
coordinates between system components (database, orchestration, export, etc.).
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from contextlib import contextmanager

# Use absolute imports from src package
from src.drawing_intelligence.utils.config_loader import Config
from src.drawing_intelligence.database.database_manager import (
    DatabaseManager,
    QueryFilters,
)
from src.drawing_intelligence.orchestration.pipeline_orchestrator import (
    PipelineOrchestrator,
)
from src.drawing_intelligence.orchestration.routing_engine import RoutingEngine
from src.drawing_intelligence.orchestration.checkpoint_manager import (
    CheckpointManager,
)
from src.drawing_intelligence.llm.budget_controller import BudgetController
from src.drawing_intelligence.export.export_manager import (
    ExportManager,
    ExportConfig,
)
from src.drawing_intelligence.cli.config_validator import ConfigValidator
from src.drawing_intelligence.models.data_structures import ProcessingResult
from src.drawing_intelligence.utils.error_handlers import (
    DrawingProcessingError,
    ConfigurationError,
    DatabaseError,
)


logger = logging.getLogger(__name__)

# Constants
MAX_DISPLAY_FLAGS = 5
MAX_DRAWING_ID_DISPLAY_LENGTH = 30
DEFAULT_COST_REPORT_NAME = "cost_report.html"
DEFAULT_WORKER_COUNT = 4
DEFAULT_LIST_LIMIT = 50
DEFAULT_LIST_OFFSET = 0
MAX_WORKERS = 32
MIN_WORKERS = 1
SEPARATOR_WIDTH = 60
WIDE_SEPARATOR_WIDTH = 80

# Version
__version__ = "1.0.0"


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the CLI application.

    Sets up console logging with timestamp, logger name, level, and message
    format. Application output goes to stdout, logs go to stderr.

    Args:
        log_level: Logging level as string (DEBUG, INFO, WARNING, ERROR).
            Defaults to "INFO".
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


@contextmanager
def get_database_manager(config_path: str):
    """Context manager for database connections.

    Args:
        config_path: Path to system configuration file.

    Yields:
        tuple: (Config, DatabaseManager) objects.

    Raises:
        ConfigurationError: If configuration cannot be loaded.
        DatabaseError: If database connection fails.
    """
    config = None
    db = None
    try:
        config = Config.load(config_path)
        db = DatabaseManager(config.database["path"])
        yield config, db
    finally:
        if db is not None:
            db.close()


def initialize_components(config, db: DatabaseManager):
    """Initialize core system components.

    Args:
        config: Loaded system configuration.
        db: Database manager instance.

    Returns:
        tuple: (BudgetController, CheckpointManager, RoutingEngine,
                PipelineOrchestrator)
    """
    budget_controller = BudgetController(
        daily_budget_usd=config.llm_integration["cost_controls"]["daily_budget_usd"],
        per_drawing_limit_usd=config.llm_integration["cost_controls"][
            "per_drawing_limit_usd"
        ],
        db_manager=db,
    )

    checkpoint_manager = CheckpointManager(
        config.batch_processing["batch_checkpoint_dir"]
    )

    routing_engine = RoutingEngine(config, budget_controller)

    orchestrator = PipelineOrchestrator(
        config=config,
        db=db,
        checkpoint_manager=checkpoint_manager,
        routing_engine=routing_engine,
    )

    return budget_controller, checkpoint_manager, routing_engine, orchestrator


def validate_path(path: Path, must_exist: bool = True) -> None:
    """Validate file or directory path.

    Args:
        path: Path to validate.
        must_exist: Whether the path must exist.

    Raises:
        FileNotFoundError: If path doesn't exist and must_exist is True.
        ValueError: If path is invalid.
    """
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Basic path traversal check
    try:
        path.resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid path: {path}") from e


def ensure_output_directory(path: Path) -> None:
    """Create output directory if it doesn't exist.

    Args:
        path: Directory path to create.

    Raises:
        PermissionError: If directory cannot be created.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(f"Cannot create output directory: {path}") from e


def handle_error(context: str, error: Exception) -> int:
    """Centralized error handling for commands.

    Args:
        context: Description of the operation that failed.
        error: Exception that was raised.

    Returns:
        Exit code 1.
    """
    if isinstance(error, FileNotFoundError):
        logger.error(f"{context}: File not found - {error}")
    elif isinstance(error, PermissionError):
        logger.error(f"{context}: Permission denied - {error}")
    elif isinstance(error, ValueError):
        logger.error(f"{context}: Invalid value - {error}")
    elif isinstance(error, (ConfigurationError, DatabaseError)):
        logger.error(f"{context}: {error}")
    else:
        logger.error(f"{context}: {error}", exc_info=True)
    return 1


def main() -> int:
    """Execute the main CLI entry point.

    Parses command-line arguments, configures logging, and routes to the
    appropriate command handler. All exceptions are caught and logged.

    Returns:
        Exit code: 0 for success, non-zero for errors.

    Example:
        $ python main.py process drawing.pdf
        $ python main.py batch --input-dir ./drawings --workers 4
    """
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Handle version flag
    if hasattr(args, "version") and args.version:
        print(f"Drawing Intelligence System v{__version__}")
        return 0

    # Setup logging
    log_level = getattr(args, "log_level", "INFO")
    setup_logging(log_level)

    if not hasattr(args, "command") or args.command is None:
        parser.print_help()
        return 1

    try:
        # Route to appropriate command handler
        command_map = {
            "process": command_process,
            "batch": command_batch,
            "export": command_export,
            "cost-report": command_cost_report,
            "validate-config": command_validate_config,
            "list": command_list_drawings,
        }

        handler = command_map.get(args.command)
        if handler:
            return handler(args)
        else:
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130
    except Exception as e:
        return handle_error("Command execution failed", e)


def command_process(args: argparse.Namespace) -> int:
    """Process a single PDF drawing through the extraction pipeline.

    Initializes all required components (database, budget controller, routing
    engine, orchestrator) and processes the specified drawing. Optionally
    exports results in JSON or CSV format.

    Args:
        args: Parsed command-line arguments containing:
            - pdf_path: Path to the PDF file to process
            - force_llm: Whether to force LLM enhancement
            - export: Whether to export results
            - export_format: Format for export (json/csv)
            - output: Optional output path
            - config: Path to system configuration file

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    try:
        pdf_path = Path(args.pdf_path)
        validate_path(pdf_path)
    except (FileNotFoundError, ValueError) as e:
        return handle_error("Invalid input path", e)

    logger.info(f"Processing drawing: {pdf_path}")

    try:
        with get_database_manager(args.config) as (config, db):
            _, _, _, orchestrator = initialize_components(config, db)

            # Process drawing
            result = orchestrator.process_drawing(
                pdf_path=str(pdf_path), force_llm=args.force_llm
            )

            # Print summary
            print_result_summary(result)

            # Export if requested
            if args.export:
                output_path = handle_export(
                    db, result.drawing_id, args.export_format, args.output
                )
                print(f"✓ Exported to: {output_path}")

            return 0

    except Exception as e:
        return handle_error("Processing failed", e)


def command_batch(args: argparse.Namespace) -> int:
    """Process multiple PDF drawings in batch mode with parallel workers.

    Discovers PDF files in the input directory (optionally recursively),
    processes them using configurable parallel workers, and supports
    checkpoint-based resumption. Generates batch summary and optional exports.

    Args:
        args: Parsed command-line arguments containing:
            - input_dir: Directory containing PDF files
            - recursive: Whether to search subdirectories
            - workers: Number of parallel processing workers
            - batch_id: Optional batch identifier
            - resume: Whether to resume a previous batch
            - export: Whether to export results
            - export_format: Format for export (json/csv)
            - output: Optional output path
            - report: Optional HTML report path
            - config: Path to system configuration file

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    try:
        input_path = Path(args.input_dir)
        validate_path(input_path)
    except (FileNotFoundError, ValueError) as e:
        return handle_error("Invalid input directory", e)

    # Validate worker count
    if not MIN_WORKERS <= args.workers <= MAX_WORKERS:
        logger.error(f"Worker count must be between {MIN_WORKERS} and {MAX_WORKERS}")
        return 1

    logger.info(f"Processing batch: {input_path}")

    try:
        with get_database_manager(args.config) as (config, db):
            _, _, _, orchestrator = initialize_components(config, db)

            # Find PDF files
            if args.resume:
                if not args.batch_id:
                    logger.error("--batch-id is required when using --resume")
                    return 1
                logger.info(f"Resuming batch {args.batch_id}")
                batch_result = orchestrator.resume_batch(args.batch_id)
            else:
                pdf_files = sorted(
                    input_path.glob("**/*.pdf" if args.recursive else "*.pdf")
                )

                if not pdf_files:
                    logger.error(f"No PDF files found in {input_path}")
                    return 1

                logger.info(f"Found {len(pdf_files)} PDF files")

                # Generate batch_id if not provided
                batch_id = args.batch_id or (
                    f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )

                batch_result = orchestrator.process_batch(
                    pdf_paths=[str(f) for f in pdf_files],
                    batch_id=batch_id,
                    parallel_workers=args.workers,
                )

            # Print summary
            print_batch_summary(batch_result)

            # Export if requested
            if args.export:
                handle_batch_export(
                    db, batch_result, args.export_format, args.output, args.report
                )

            return 0

    except Exception as e:
        return handle_error("Batch processing failed", e)


def command_export(args: argparse.Namespace) -> int:
    """Export results for a previously processed drawing.

    Supports multiple export formats: JSON (structured data), CSV (tabular),
    and HTML report (human-readable summary).

    Args:
        args: Parsed command-line arguments containing:
            - drawing_id: Identifier of the drawing to export
            - format: Export format (json/csv/report)
            - output: Optional output path (defaults to drawing_id-based name)
            - config: Path to system configuration file

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    logger.info(f"Exporting drawing: {args.drawing_id}")

    try:
        with get_database_manager(args.config) as (config, db):
            output_path = handle_export(db, args.drawing_id, args.format, args.output)
            print(f"✓ Exported to: {output_path}")
            return 0

    except Exception as e:
        return handle_error("Export failed", e)


def command_cost_report(args: argparse.Namespace) -> int:
    """Generate LLM cost analysis report for a date range.

    Produces an HTML report summarizing LLM API usage and costs. Date range
    can be specified either as number of days from now or explicit start/end
    dates.

    Args:
        args: Parsed command-line arguments containing either:
            - days: Number of days to look back from now, OR
            - start_date: ISO format start date (YYYY-MM-DD)
            - end_date: ISO format end date (YYYY-MM-DD)
            - output: Optional output path (defaults to 'cost_report.html')
            - config: Path to system configuration file

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    logger.info("Generating cost report")

    try:
        # Parse dates
        if args.days:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=args.days)
        else:
            start_date = datetime.fromisoformat(args.start_date)
            end_date = datetime.fromisoformat(args.end_date)

        with get_database_manager(args.config) as (config, db):
            export_manager = ExportManager(db, ExportConfig())
            output_path = args.output or DEFAULT_COST_REPORT_NAME

            # Ensure output directory exists
            output_path_obj = Path(output_path)
            if output_path_obj.parent != Path("."):
                ensure_output_directory(output_path_obj.parent)

            export_manager.export_cost_report(
                start_date=start_date,
                end_date=end_date,
                output_path=output_path,
                format="html",
            )

            print(f"✓ Cost report generated: {output_path}")
            return 0

    except ValueError as e:
        return handle_error("Invalid date format", e)
    except Exception as e:
        return handle_error("Cost report generation failed", e)


def command_validate_config(args: argparse.Namespace) -> int:
    """Validate system configuration file for correctness and completeness.

    Checks all configuration sections (paths, database, models, LLM settings)
    and prints a detailed validation report with errors, warnings, and info
    messages.

    Args:
        args: Parsed command-line arguments containing:
            - config: Path to system configuration file to validate

    Returns:
        Exit code: 0 if configuration is valid, 1 if errors found.
    """
    logger.info(f"Validating configuration: {args.config}")

    try:
        config = Config.load(args.config)
        validator = ConfigValidator(config)
        result = validator.validate_all()

        # Print report
        validator.print_validation_report(result)

        return 0 if result.is_valid else 1

    except Exception as e:
        return handle_error("Configuration validation failed", e)


def command_list_drawings(args: argparse.Namespace) -> int:
    """Query and display processed drawings from the database.

    Lists drawings with filtering options (status, review flag) and pagination
    support. Displays drawing ID, confidence score, status, and review flag
    in a formatted table.

    Args:
        args: Parsed command-line arguments containing:
            - status: Optional status filter
            - needs_review: Whether to show only drawings needing review
            - limit: Maximum number of results (default: 50)
            - offset: Result offset for pagination (default: 0)
            - config: Path to system configuration file

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    logger.info("Listing drawings")

    # Validate pagination parameters
    if args.limit < 1:
        logger.error("--limit must be at least 1")
        return 1
    if args.offset < 0:
        logger.error("--offset must be non-negative")
        return 1

    try:
        with get_database_manager(args.config) as (config, db):
            # Build filters
            filters = QueryFilters(
                status=args.status,
                needs_review=args.needs_review,
                limit=args.limit,
                offset=args.offset,
            )

            # Query database
            drawings = db.query_drawings(filters)

            # Print results
            print("\n" + "=" * WIDE_SEPARATOR_WIDTH)
            print(f"Found {len(drawings)} drawing(s)")
            print("=" * WIDE_SEPARATOR_WIDTH)
            print(
                f"{'Drawing ID':<40} {'Confidence':<12} "
                f"{'Status':<15} {'Review':<10}"
            )
            print("-" * WIDE_SEPARATOR_WIDTH)

            for drawing in drawings:
                review_str = "Yes" if drawing.needs_review else "No"
                print(
                    f"{drawing.drawing_id:<40} "
                    f"{drawing.overall_confidence:<12.2f} "
                    f"{drawing.status:<15} {review_str:<10}"
                )

            print("=" * WIDE_SEPARATOR_WIDTH)

            return 0

    except Exception as e:
        return handle_error("List command failed", e)


def handle_export(
    db: DatabaseManager,
    drawing_id: str,
    export_format: str,
    output: Optional[str],
) -> str:
    """Handle export operations for a single drawing.

    Args:
        db: Database manager instance.
        drawing_id: Drawing identifier.
        export_format: Export format (json/csv/report).
        output: Optional output path.

    Returns:
        Path to exported file(s).

    Raises:
        ValueError: If export format is invalid.
    """
    export_manager = ExportManager(db, ExportConfig())

    if export_format == "json":
        output_path = output or f"{drawing_id}.json"
        output_path_obj = Path(output_path)
        if output_path_obj.parent != Path("."):
            ensure_output_directory(output_path_obj.parent)
        export_manager.export_drawing_json(drawing_id, output_path)
        return output_path

    elif export_format == "csv":
        output_dir = output or f"{drawing_id}_csv"
        ensure_output_directory(Path(output_dir))
        files = export_manager.export_drawing_csv(drawing_id, output_dir)
        return f"{output_dir} ({len(files)} files)"

    elif export_format == "report":
        output_path = output or f"{drawing_id}_report.html"
        output_path_obj = Path(output_path)
        if output_path_obj.parent != Path("."):
            ensure_output_directory(output_path_obj.parent)
        export_manager.export_drawing_report(drawing_id, output_path, format="html")
        return output_path

    else:
        raise ValueError(f"Invalid export format: {export_format}")


def handle_batch_export(
    db: DatabaseManager,
    batch_result,
    export_format: str,
    output: Optional[str],
    report_path: Optional[str],
) -> None:
    """Handle export operations for batch results.

    Args:
        db: Database manager instance.
        batch_result: Batch processing result object.
        export_format: Export format (json/csv).
        output: Optional output path.
        report_path: Optional HTML report path.
    """
    export_manager = ExportManager(db, ExportConfig())
    drawing_ids = [r.drawing_id for r in batch_result.drawing_results]

    if export_format == "json":
        output_path = output or f"{batch_result.batch_id}.json"
        output_path_obj = Path(output_path)
        if output_path_obj.parent != Path("."):
            ensure_output_directory(output_path_obj.parent)
        export_manager.export_batch_json(drawing_ids, output_path)
        print(f"✓ Exported to: {output_path}")

    elif export_format == "csv":
        output_dir = output or f"{batch_result.batch_id}_csv"
        ensure_output_directory(Path(output_dir))
        files = export_manager.export_batch_csv(drawing_ids, output_dir)
        print(f"✓ Exported {len(files)} CSV files to: {output_dir}")

    # Generate report if requested
    if report_path:
        report_path_obj = Path(report_path)
        if report_path_obj.parent != Path("."):
            ensure_output_directory(report_path_obj.parent)
        export_manager.generate_batch_report(
            batch_result.batch_id, report_path, format="html"
        )
        print(f"✓ Generated report: {report_path}")


def setup_argument_parser() -> argparse.ArgumentParser:
    """Configure the argument parser with all CLI commands and options.

    Creates a parser with subcommands for:
    - process: Single drawing processing
    - batch: Batch processing with parallel workers
    - export: Export drawing results
    - cost-report: Generate LLM cost reports
    - validate-config: Validate system configuration
    - list: Query processed drawings

    Returns:
        Configured ArgumentParser instance ready to parse sys.argv.

    Example:
        parser = setup_argument_parser()
        args = parser.parse_args()
    """
    parser = argparse.ArgumentParser(
        description="Drawing Intelligence System - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single drawing
  %(prog)s process drawing.pdf

  # Process batch with 8 workers
  %(prog)s batch ./drawings --workers 8 --recursive

  # Export drawing as JSON
  %(prog)s export DWG-001 --format json

  # Generate cost report for last 7 days
  %(prog)s cost-report --days 7

  # List drawings needing review
  %(prog)s list --needs-review
        """,
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )

    parser.add_argument(
        "--config",
        default="config/system_config.yaml",
        help="Path to configuration file (default: %(default)s)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: %(default)s)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command
    process_parser = subparsers.add_parser(
        "process",
        help="Process a single drawing",
        description="Process a single PDF drawing through the extraction pipeline.",
    )
    process_parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to PDF file",
    )
    process_parser.add_argument(
        "--force-llm",
        action="store_true",
        help="Force LLM enhancement even for high-confidence results",
    )
    process_parser.add_argument(
        "--export",
        action="store_true",
        help="Export results after processing",
    )
    process_parser.add_argument(
        "--export-format",
        choices=["json", "csv"],
        default="json",
        help="Export format (default: %(default)s)",
    )
    process_parser.add_argument(
        "--output",
        help="Output path for export",
    )

    # Batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Process multiple drawings",
        description="Process multiple PDF drawings in batch mode with parallel workers.",
    )
    batch_parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing PDF files",
    )
    batch_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively",
    )
    batch_parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKER_COUNT,
        help=f"Number of parallel workers "
        f"(default: %(default)s, max: {MAX_WORKERS})",
    )
    batch_parser.add_argument(
        "--batch-id",
        help="Custom batch identifier (auto-generated if not provided)",
    )
    batch_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume previous batch (requires --batch-id)",
    )
    batch_parser.add_argument(
        "--export",
        action="store_true",
        help="Export results after processing",
    )
    batch_parser.add_argument(
        "--export-format",
        choices=["json", "csv"],
        default="csv",
        help="Export format (default: %(default)s)",
    )
    batch_parser.add_argument(
        "--output",
        help="Output path for export",
    )
    batch_parser.add_argument(
        "--report",
        help="Generate HTML batch report at specified path",
    )

    # Export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export drawing results",
        description="Export results for a previously processed drawing.",
    )
    export_parser.add_argument(
        "drawing_id",
        help="Drawing ID to export",
    )
    export_parser.add_argument(
        "--format",
        choices=["json", "csv", "report"],
        required=True,
        help="Export format",
    )
    export_parser.add_argument(
        "--output",
        help="Output path (auto-generated if not provided)",
    )

    # Cost report command
    cost_parser = subparsers.add_parser(
        "cost-report",
        help="Generate cost report",
        description="Generate LLM cost analysis report for a date range.",
    )
    cost_group = cost_parser.add_mutually_exclusive_group(required=True)
    cost_group.add_argument(
        "--days",
        type=int,
        help="Report last N days",
    )
    cost_group.add_argument(
        "--date-range",
        nargs=2,
        metavar=("START", "END"),
        help="Date range in ISO format (YYYY-MM-DD)",
    )
    cost_parser.add_argument(
        "--output",
        help="Output path (default: cost_report.html)",
    )

    # Validate config command
    validate_parser = subparsers.add_parser(
        "validate-config",
        help="Validate system configuration",
        description="Validate system configuration file for correctness.",
    )

    # List drawings command
    list_parser = subparsers.add_parser(
        "list",
        help="List processed drawings",
        description="Query and display processed drawings from database.",
    )
    list_parser.add_argument(
        "--status",
        help="Filter by status (e.g., 'complete', 'failed')",
    )
    list_parser.add_argument(
        "--needs-review",
        action="store_true",
        help="Show only drawings needing review",
    )
    list_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIST_LIMIT,
        help=f"Maximum results (default: %(default)s)",
    )
    list_parser.add_argument(
        "--offset",
        type=int,
        default=DEFAULT_LIST_OFFSET,
        help=f"Result offset for pagination (default: %(default)s)",
    )

    return parser


def print_batch_summary(batch_result) -> None:
    """Print formatted summary of batch processing results.

    Args:
        batch_result: Batch processing result object.
    """
    print("\n" + "=" * SEPARATOR_WIDTH)
    print("BATCH PROCESSING SUMMARY")
    print("=" * SEPARATOR_WIDTH)
    print(f"Batch ID: {batch_result.batch_id}")
    print(f"Total Drawings: {batch_result.total_drawings}")
    print(f"Successful: {batch_result.successful}")
    print(f"Failed: {batch_result.failed}")
    print(f"Needs Review: {batch_result.needs_review}")
    print(f"Success Rate: {batch_result.success_rate * 100:.1f}%")
    print(f"Review Rate: {batch_result.review_rate * 100:.1f}%")
    print(f"Total LLM Cost: ${batch_result.total_llm_cost:.2f}")
    print(f"Avg Processing Time: " f"{batch_result.average_processing_time:.2f}s")
    print("=" * SEPARATOR_WIDTH)


def print_result_summary(result: ProcessingResult) -> None:
    """Print a formatted summary of drawing processing results to console.

    Displays key metrics including drawing ID, status, confidence score,
    entity/shape counts, associations, and review flags (first 5 shown).

    Args:
        result: ProcessingResult object containing all extraction results
            and metadata for a processed drawing.
    """
    print("\n" + "=" * SEPARATOR_WIDTH)
    print("PROCESSING RESULT SUMMARY")
    print("=" * SEPARATOR_WIDTH)
    print(f"Drawing ID: {result.drawing_id}")
    print(f"Source File: {result.source_file}")
    print(f"Status: {result.status}")
    print(f"Overall Confidence: {result.overall_confidence:.2f}")
    print(f"Needs Review: {result.needs_human_review()}")
    print(f"\nExtracted Entities: {len(result.entities)}")
    print(f"Detected Shapes: {len(result.detections)}")
    print(f"Text-Shape Associations: {len(result.associations)}")

    if result.review_flags:
        print(f"\nReview Flags ({len(result.review_flags)}):")
        for flag in result.review_flags[:MAX_DISPLAY_FLAGS]:
            print(
                f"  - [{flag.severity.value}] " f"{flag.flag_type.value}: {flag.reason}"
            )

    print("=" * SEPARATOR_WIDTH + "\n")


if __name__ == "__main__":
    sys.exit(main())
