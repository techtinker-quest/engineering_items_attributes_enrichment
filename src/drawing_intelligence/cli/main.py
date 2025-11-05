"""
CLI Interface Module

Command-line interface for the Drawing Intelligence System.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

# Import system components
from ..utils.config_loader import Config
from ..database.database_manager import DatabaseManager, QueryFilters
from ..orchestration.pipeline_orchestrator import PipelineOrchestrator
from ..orchestration.routing_engine import RoutingEngine
from ..orchestration.checkpoint_manager import CheckpointManager
from ..llm.budget_controller import BudgetController
from ..export.export_manager import ExportManager, ExportConfig
from .config_validator import ConfigValidator


logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> int:
    """
    Main CLI entry point.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level if hasattr(args, "log_level") else "INFO")

    try:
        # Route to appropriate command handler
        if args.command == "process":
            return command_process(args)
        elif args.command == "batch":
            return command_batch(args)
        elif args.command == "export":
            return command_export(args)
        elif args.command == "cost-report":
            return command_cost_report(args)
        elif args.command == "validate-config":
            return command_validate_config(args)
        elif args.command == "list":
            return command_list_drawings(args)
        else:
            parser.print_help()
            return 1

    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        return 1


def command_process(args: argparse.Namespace) -> int:
    """
    Handle 'process' command for single drawing.

    Args:
        args: Command-line arguments

    Returns:
        Exit code
    """
    logger.info(f"Processing drawing: {args.pdf_path}")

    # Load configuration
    config = Config.load(args.config)

    # Initialize components
    db = DatabaseManager(config.database.path)
    budget_controller = BudgetController(
        daily_budget_usd=config.llm_integration.cost_controls.daily_budget_usd,
        per_drawing_limit_usd=config.llm_integration.cost_controls.per_drawing_limit_usd,
        db_manager=db,
    )

    checkpoint_manager = CheckpointManager(config.batch_processing.batch_checkpoint_dir)

    routing_engine = RoutingEngine(config, budget_controller)

    orchestrator = PipelineOrchestrator(
        config=config,
        db=db,
        checkpoint_manager=checkpoint_manager,
        routing_engine=routing_engine,
    )

    # Process drawing
    try:
        result = orchestrator.process_drawing(
            pdf_path=args.pdf_path, force_llm=args.force_llm
        )

        # Print summary
        print_result_summary(result)

        # Export if requested
        if args.export:
            export_manager = ExportManager(db, ExportConfig())

            if args.export_format == "json":
                output_path = args.output or f"{result.drawing_id}.json"
                export_manager.export_drawing_json(result.drawing_id, output_path)
                logger.info(f"Exported to {output_path}")

            elif args.export_format == "csv":
                output_dir = args.output or f"{result.drawing_id}_csv"
                files = export_manager.export_drawing_csv(result.drawing_id, output_dir)
                logger.info(f"Exported {len(files)} CSV files to {output_dir}")

        return 0

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        return 1

    finally:
        db.close()


def command_batch(args: argparse.Namespace) -> int:
    """
    Handle 'batch' command for multiple drawings.

    Args:
        args: Command-line arguments

    Returns:
        Exit code
    """
    logger.info(f"Processing batch: {args.input_dir}")

    # Load configuration
    config = Config.load(args.config)

    # Initialize components
    db = DatabaseManager(config.database.path)
    budget_controller = BudgetController(
        daily_budget_usd=config.llm_integration.cost_controls.daily_budget_usd,
        per_drawing_limit_usd=config.llm_integration.cost_controls.per_drawing_limit_usd,
        db_manager=db,
    )

    checkpoint_manager = CheckpointManager(config.batch_processing.batch_checkpoint_dir)

    routing_engine = RoutingEngine(config, budget_controller)

    orchestrator = PipelineOrchestrator(
        config=config,
        db=db,
        checkpoint_manager=checkpoint_manager,
        routing_engine=routing_engine,
    )

    # Find PDF files
    input_path = Path(args.input_dir)
    pdf_files = list(input_path.glob("**/*.pdf" if args.recursive else "*.pdf"))

    if not pdf_files:
        logger.error(f"No PDF files found in {args.input_dir}")
        return 1

    logger.info(f"Found {len(pdf_files)} PDF files")

    try:
        # Process batch
        if args.resume:
            logger.info(f"Resuming batch {args.batch_id}")
            batch_result = orchestrator.resume_batch(args.batch_id)
        else:
            batch_result = orchestrator.process_batch(
                pdf_paths=[str(f) for f in pdf_files],
                batch_id=args.batch_id,
                parallel_workers=args.workers,
            )

        # Print summary
        print("\n" + "=" * 60)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Batch ID: {batch_result.batch_id}")
        print(f"Total Drawings: {batch_result.total_drawings}")
        print(f"Successful: {batch_result.successful}")
        print(f"Failed: {batch_result.failed}")
        print(f"Needs Review: {batch_result.needs_review}")
        print(f"Success Rate: {batch_result.success_rate * 100:.1f}%")
        print(f"Review Rate: {batch_result.review_rate * 100:.1f}%")
        print(f"Total LLM Cost: ${batch_result.total_llm_cost:.2f}")
        print(f"Avg Processing Time: {batch_result.average_processing_time:.2f}s")
        print("=" * 60)

        # Export if requested
        if args.export:
            export_manager = ExportManager(db, ExportConfig())
            drawing_ids = [r.drawing_id for r in batch_result.drawing_results]

            if args.export_format == "json":
                output_path = args.output or f"{batch_result.batch_id}.json"
                export_manager.export_batch_json(drawing_ids, output_path)
                logger.info(f"Exported to {output_path}")

            elif args.export_format == "csv":
                output_dir = args.output or f"{batch_result.batch_id}_csv"
                files = export_manager.export_batch_csv(drawing_ids, output_dir)
                logger.info(f"Exported {len(files)} CSV files to {output_dir}")

            # Generate report
            if args.report:
                report_path = args.report
                export_manager.generate_batch_report(
                    batch_result.batch_id, report_path, format="html"
                )
                logger.info(f"Generated report: {report_path}")

        return 0

    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        return 1

    finally:
        db.close()


def command_export(args: argparse.Namespace) -> int:
    """
    Handle 'export' command.

    Args:
        args: Command-line arguments

    Returns:
        Exit code
    """
    logger.info(f"Exporting drawing: {args.drawing_id}")

    # Load configuration
    config = Config.load(args.config)
    db = DatabaseManager(config.database.path)

    try:
        export_manager = ExportManager(db, ExportConfig())

        if args.format == "json":
            output_path = args.output or f"{args.drawing_id}.json"
            export_manager.export_drawing_json(args.drawing_id, output_path)
            logger.info(f"Exported to {output_path}")

        elif args.format == "csv":
            output_dir = args.output or f"{args.drawing_id}_csv"
            files = export_manager.export_drawing_csv(args.drawing_id, output_dir)
            logger.info(f"Exported {len(files)} CSV files")

        elif args.format == "report":
            output_path = args.output or f"{args.drawing_id}_report.html"
            export_manager.export_drawing_report(
                args.drawing_id, output_path, format="html"
            )
            logger.info(f"Generated report: {output_path}")

        return 0

    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        return 1

    finally:
        db.close()


def command_cost_report(args: argparse.Namespace) -> int:
    """
    Handle 'cost-report' command.

    Args:
        args: Command-line arguments

    Returns:
        Exit code
    """
    logger.info("Generating cost report")

    # Load configuration
    config = Config.load(args.config)
    db = DatabaseManager(config.database.path)

    try:
        # Parse dates
        if args.days:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=args.days)
        else:
            start_date = datetime.fromisoformat(args.start_date)
            end_date = datetime.fromisoformat(args.end_date)

        export_manager = ExportManager(db, ExportConfig())
        output_path = args.output or "cost_report.html"

        export_manager.export_cost_report(
            start_date=start_date,
            end_date=end_date,
            output_path=output_path,
            format="html",
        )

        logger.info(f"Generated cost report: {output_path}")
        return 0

    except Exception as e:
        logger.error(f"Cost report generation failed: {e}", exc_info=True)
        return 1

    finally:
        db.close()


def command_validate_config(args: argparse.Namespace) -> int:
    """
    Handle 'validate-config' command.

    Args:
        args: Command-line arguments

    Returns:
        Exit code
    """
    logger.info(f"Validating configuration: {args.config}")

    try:
        # Load configuration
        config = Config.load(args.config)

        # Validate
        validator = ConfigValidator(config)
        result = validator.validate_all()

        # Print report
        validator.print_validation_report(result)

        return 0 if result.is_valid else 1

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}", exc_info=True)
        return 1


def command_list_drawings(args: argparse.Namespace) -> int:
    """
    Handle 'list' command to query drawings.

    Args:
        args: Command-line arguments

    Returns:
        Exit code
    """
    logger.info("Listing drawings")

    # Load configuration
    config = Config.load(args.config)
    db = DatabaseManager(config.database.path)

    try:
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
        print("\n" + "=" * 80)
        print(f"Found {len(drawings)} drawings")
        print("=" * 80)
        print(f"{'Drawing ID':<40} {'Confidence':<12} {'Status':<15} {'Review':<10}")
        print("-" * 80)

        for drawing in drawings:
            review_str = "Yes" if drawing.needs_review else "No"
            print(
                f"{drawing.drawing_id:<40} {drawing.overall_confidence:<12.2f} {drawing.status:<15} {review_str:<10}"
            )

        print("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"List command failed: {e}", exc_info=True)
        return 1

    finally:
        db.close()


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Setup CLI argument parser.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Drawing Intelligence System - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        default="config/system_config.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process a single drawing")
    process_parser.add_argument("pdf_path", help="Path to PDF file")
    process_parser.add_argument(
        "--force-llm", action="store_true", help="Force LLM enhancement"
    )
    process_parser.add_argument("--export", action="store_true", help="Export results")
    process_parser.add_argument(
        "--export-format", choices=["json", "csv"], default="json", help="Export format"
    )
    process_parser.add_argument("--output", help="Output path")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Process multiple drawings")
    batch_parser.add_argument("input_dir", help="Input directory containing PDFs")
    batch_parser.add_argument(
        "--recursive", action="store_true", help="Search recursively"
    )
    batch_parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )
    batch_parser.add_argument("--batch-id", help="Batch identifier")
    batch_parser.add_argument(
        "--resume", action="store_true", help="Resume previous batch"
    )
    batch_parser.add_argument("--export", action="store_true", help="Export results")
    batch_parser.add_argument(
        "--export-format", choices=["json", "csv"], default="csv", help="Export format"
    )
    batch_parser.add_argument("--output", help="Output path")
    batch_parser.add_argument("--report", help="Generate HTML report")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export drawing results")
    export_parser.add_argument("drawing_id", help="Drawing ID to export")
    export_parser.add_argument(
        "--format",
        choices=["json", "csv", "report"],
        required=True,
        help="Export format",
    )
    export_parser.add_argument("--output", help="Output path")

    # Cost report command
    cost_parser = subparsers.add_parser("cost-report", help="Generate cost report")
    cost_parser.add_argument("--days", type=int, help="Report last N days")
    cost_parser.add_argument("--start-date", help="Start date (ISO format)")
    cost_parser.add_argument("--end-date", help="End date (ISO format)")
    cost_parser.add_argument("--output", help="Output path")

    # Validate config command
    validate_parser = subparsers.add_parser(
        "validate-config", help="Validate system configuration"
    )

    # List drawings command
    list_parser = subparsers.add_parser("list", help="List processed drawings")
    list_parser.add_argument("--status", help="Filter by status")
    list_parser.add_argument(
        "--needs-review", action="store_true", help="Show only drawings needing review"
    )
    list_parser.add_argument("--limit", type=int, default=50, help="Maximum results")
    list_parser.add_argument("--offset", type=int, default=0, help="Result offset")

    return parser


def display_progress(current: int, total: int, drawing_id: str) -> None:
    """
    Display progress bar for batch processing.

    Args:
        current: Current drawing number
        total: Total drawings
        drawing_id: Current drawing ID
    """
    percent = (current / total) * 100
    bar_length = 50
    filled = int(bar_length * current / total)
    bar = "█" * filled + "-" * (bar_length - filled)

    print(f"\r[{bar}] {percent:.1f}% ({current}/{total}) - {drawing_id[:30]}", end="")

    if current == total:
        print()  # New line when complete


def print_result_summary(result) -> None:
    """
    Print summary of processing result to console.

    Args:
        result: ProcessingResult object
    """
    print("\n" + "=" * 60)
    print("PROCESSING RESULT SUMMARY")
    print("=" * 60)
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
        for flag in result.review_flags[:5]:  # Show first 5
            print(f"  - [{flag.severity.value}] {flag.flag_type.value}: {flag.reason}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    sys.exit(main())
