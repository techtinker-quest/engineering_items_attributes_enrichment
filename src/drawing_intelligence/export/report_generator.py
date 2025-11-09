"""
Report Generator Module

Generates visual reports (HTML/PDF) for drawing processing results with
charts, statistics, and detailed analysis.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from datetime import datetime
import base64
from io import BytesIO
import tempfile

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from weasyprint import HTML as WeasyHTML

    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

# Type checking imports to avoid circular dependencies
if TYPE_CHECKING:
    from ..models.data_structures import (
        BatchResult,
        CostReport,
        DailyCost,
        DrawingRecord,
        EntityType,
    )

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Supported report output formats."""

    HTML = "html"
    PDF = "pdf"


class ReportGenerationError(Exception):
    """Base exception for report generation errors."""

    pass


class TemplateError(ReportGenerationError):
    """Exception raised when template rendering fails."""

    pass


class ChartGenerationError(ReportGenerationError):
    """Exception raised when chart generation fails."""

    pass


class MissingDependencyError(ReportGenerationError):
    """Exception raised when required dependency is missing."""

    pass


class ReportGenerator:
    """Generate visual reports from drawing processing results.

    This class creates comprehensive HTML and PDF reports for batch processing
    results, cost analysis, and individual drawing details. It uses Jinja2
    templates for HTML generation and optionally converts to PDF using
    WeasyPrint. Charts are generated using Matplotlib when available.

    Attributes:
        template_dir (Path): Directory containing Jinja2 templates.
        env (Optional[Environment]): Jinja2 environment for template rendering,
            None if jinja2 is not available.
        max_drawings_in_report (int): Maximum number of drawings to include
            in batch reports.
        chart_dpi (int): DPI resolution for generated charts.
        enable_charts (bool): Whether to generate charts even if matplotlib
            is available.

    Note:
        Requires optional dependencies: jinja2, matplotlib, weasyprint
        (for PDF).
    """

    def __init__(
        self,
        template_dir: Union[str, Path] = "config/templates",
        max_drawings_in_report: int = 20,
        chart_dpi: int = 100,
        enable_charts: bool = True,
    ) -> None:
        """Initialize report generator.

        Args:
            template_dir: Directory containing Jinja2 templates. Will be used
                as the template loader search path.
            max_drawings_in_report: Maximum number of drawings to include in
                batch report details.
            chart_dpi: DPI resolution for chart images.
            enable_charts: Whether to generate charts. Can be disabled for
                performance even if matplotlib is available.

        Raises:
            FileNotFoundError: If template_dir does not exist.
            MissingDependencyError: If jinja2 is not available.
        """
        self.template_dir = Path(template_dir)
        self.max_drawings_in_report = max_drawings_in_report
        self.chart_dpi = chart_dpi
        self.enable_charts = enable_charts

        if not JINJA2_AVAILABLE:
            logger.warning("jinja2 not available. Install with: pip install jinja2")

        if JINJA2_AVAILABLE:
            if not self.template_dir.exists():
                raise FileNotFoundError(
                    f"Template directory not found: {self.template_dir}"
                )

            self.env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                autoescape=select_autoescape(["html", "xml"]),
            )
        else:
            self.env = None

        logger.info(
            f"ReportGenerator initialized: template_dir={self.template_dir}, "
            f"max_drawings={self.max_drawings_in_report}, "
            f"chart_dpi={self.chart_dpi}, enable_charts={self.enable_charts}"
        )

    def generate_batch_report(
        self,
        batch_result: "BatchResult",
        drawing_records: List["DrawingRecord"],
        output_path: Union[str, Path],
        format: ReportFormat = ReportFormat.HTML,
    ) -> Path:
        """Generate comprehensive batch processing report.

        Creates a report summarizing batch processing results including success
        rates, review rates, costs, and individual drawing summaries. Includes
        confidence distribution and status charts when matplotlib is available.

        Args:
            batch_result: BatchResult object containing aggregate statistics.
            drawing_records: List of DrawingRecord objects for individual
                drawings.
            output_path: Output file path for the generated report.
            format: Report format (HTML or PDF). Defaults to HTML.

        Returns:
            Path to the generated report file.

        Raises:
            ValueError: If format is invalid.
            IOError: If file write operation fails.
            TemplateError: If template rendering fails.

        Note:
            PDF generation requires weasyprint. Falls back to HTML if
            unavailable.
        """
        start_time = datetime.now()
        output_path = Path(output_path)

        logger.info(
            f"Generating batch report: {output_path}",
            extra={
                "output_path": str(output_path),
                "format": format.value,
                "drawing_count": len(drawing_records),
            },
        )

        if not JINJA2_AVAILABLE:
            self._generate_simple_batch_report(batch_result, output_path)
            return output_path

        # Prepare context
        context = self._prepare_batch_context(batch_result, drawing_records)

        # Render template
        html_content = self._render_template("batch_report.html", context)

        # Write output
        result_path = self._write_report(html_content, output_path, format)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Successfully generated batch report in {elapsed:.2f}s: " f"{result_path}"
        )
        return result_path

    def generate_cost_report(
        self,
        cost_report: "CostReport",
        output_path: Union[str, Path],
        format: ReportFormat = ReportFormat.HTML,
    ) -> Path:
        """Generate cost analysis report with detailed breakdowns.

        Creates a report showing LLM API costs broken down by use case,
        provider, model, and daily trends. Includes bar charts and trend lines
        when matplotlib is available.

        Args:
            cost_report: CostReport object containing cost statistics and
                breakdowns.
            output_path: Output file path for the generated report.
            format: Report format (HTML or PDF). Defaults to HTML.

        Returns:
            Path to the generated report file.

        Raises:
            ValueError: If format is invalid.
            IOError: If file write operation fails.
            TemplateError: If template rendering fails.

        Note:
            Charts require matplotlib. PDF generation requires weasyprint.
        """
        start_time = datetime.now()
        output_path = Path(output_path)

        logger.info(
            f"Generating cost report: {output_path}",
            extra={"output_path": str(output_path), "format": format.value},
        )

        if not JINJA2_AVAILABLE:
            self._generate_simple_cost_report(cost_report, output_path)
            return output_path

        # Prepare context
        context = self._prepare_cost_context(cost_report)

        # Render template
        html_content = self._render_template("cost_report.html", context)

        # Write output
        result_path = self._write_report(html_content, output_path, format)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Successfully generated cost report in {elapsed:.2f}s: " f"{result_path}"
        )
        return result_path

    def generate_drawing_report(
        self,
        drawing_record: "DrawingRecord",
        output_path: Union[str, Path],
        format: ReportFormat = ReportFormat.HTML,
    ) -> Path:
        """Generate detailed report for a single drawing.

        Creates a comprehensive report for one drawing including all extracted
        entities, detected shapes, confidence scores, and review flags.

        Args:
            drawing_record: DrawingRecord object containing all drawing data.
            output_path: Output file path for the generated report.
            format: Report format (HTML or PDF). Defaults to HTML.

        Returns:
            Path to the generated report file.

        Raises:
            ValueError: If format is invalid.
            IOError: If file write operation fails.
            TemplateError: If template rendering fails.

        Note:
            PDF generation requires weasyprint. Falls back to HTML if
            unavailable.
        """
        start_time = datetime.now()
        output_path = Path(output_path)

        logger.info(
            f"Generating drawing report: {output_path}",
            extra={
                "output_path": str(output_path),
                "format": format.value,
                "drawing_id": drawing_record.drawing_id,
            },
        )

        if not JINJA2_AVAILABLE:
            self._generate_simple_drawing_report(drawing_record, output_path)
            return output_path

        # Prepare context
        context = self._prepare_drawing_context(drawing_record)

        # Render template
        html_content = self._render_template("drawing_report.html", context)

        # Write output
        result_path = self._write_report(html_content, output_path, format)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Successfully generated drawing report in {elapsed:.2f}s: "
            f"{result_path}"
        )
        return result_path

    def _prepare_batch_context(
        self, batch_result: "BatchResult", drawing_records: List["DrawingRecord"]
    ) -> Dict[str, Any]:
        """Prepare template context for batch report.

        Args:
            batch_result: BatchResult with aggregate statistics.
            drawing_records: List of DrawingRecord objects.

        Returns:
            Dictionary with template variables.
        """
        context = {
            "batch_id": batch_result.batch_id,
            "total_drawings": batch_result.total_drawings,
            "successful": batch_result.successful,
            "failed": batch_result.failed,
            "needs_review": batch_result.needs_review,
            "success_rate": batch_result.success_rate * 100,
            "review_rate": batch_result.review_rate * 100,
            "total_llm_cost": batch_result.total_llm_cost,
            "average_processing_time": batch_result.average_processing_time,
            "generation_time": datetime.now().isoformat(),
            "drawings": [],
        }

        # Add drawing summaries (limit to prevent memory issues)
        for record in drawing_records[: self.max_drawings_in_report]:
            context["drawings"].append(
                {
                    "drawing_id": record.drawing_id,
                    "source_file": Path(record.source_file).name,
                    "confidence": record.overall_confidence,
                    "needs_review": record.needs_review,
                    "entity_count": (len(record.entities) if record.entities else 0),
                    "detection_count": (
                        len(record.detections) if record.detections else 0
                    ),
                }
            )

        # Generate charts if available and enabled
        if MATPLOTLIB_AVAILABLE and self.enable_charts:
            confidences = [r.overall_confidence for r in drawing_records]
            context["confidence_chart"] = self._generate_confidence_chart(confidences)
            context["status_chart"] = self._generate_status_chart(batch_result)

        return context

    def _prepare_cost_context(self, cost_report: "CostReport") -> Dict[str, Any]:
        """Prepare template context for cost report.

        Args:
            cost_report: CostReport with cost statistics.

        Returns:
            Dictionary with template variables.
        """
        context = {
            "period": cost_report.period,
            "start_date": cost_report.start_date.strftime("%Y-%m-%d"),
            "end_date": cost_report.end_date.strftime("%Y-%m-%d"),
            "total_cost": cost_report.total_cost,
            "total_calls": cost_report.total_calls,
            "average_cost_per_drawing": cost_report.average_cost_per_drawing,
            "generation_time": datetime.now().isoformat(),
            "cost_by_use_case": cost_report.cost_by_use_case,
            "cost_by_provider": cost_report.cost_by_provider,
            "cost_by_model": cost_report.cost_by_model,
            "daily_costs": [
                {
                    "date": dc.date.strftime("%Y-%m-%d"),
                    "cost": dc.total_cost,
                    "calls": dc.call_count,
                }
                for dc in cost_report.daily_costs
            ],
            "top_drawings": cost_report.top_drawings_by_cost,
        }

        # Generate charts if available and enabled
        if MATPLOTLIB_AVAILABLE and self.enable_charts:
            context["use_case_chart"] = self._generate_cost_chart(
                cost_report.cost_by_use_case, title="Cost by Use Case"
            )
            context["provider_chart"] = self._generate_cost_chart(
                cost_report.cost_by_provider, title="Cost by Provider"
            )
            context["daily_trend_chart"] = self._generate_daily_trend_chart(
                cost_report.daily_costs
            )

        return context

    def _prepare_drawing_context(
        self, drawing_record: "DrawingRecord"
    ) -> Dict[str, Any]:
        """Prepare template context for drawing report.

        Args:
            drawing_record: DrawingRecord with drawing data.

        Returns:
            Dictionary with template variables.
        """
        from ..models.data_structures import EntityType

        context = {
            "drawing_id": drawing_record.drawing_id,
            "source_file": drawing_record.source_file,
            "processing_timestamp": (drawing_record.processing_timestamp.isoformat()),
            "overall_confidence": drawing_record.overall_confidence,
            "needs_review": drawing_record.needs_review,
            "status": drawing_record.status,
            "generation_time": datetime.now().isoformat(),
            "entities": [],
            "detections": [],
            "review_flags": [],
        }

        # Add entities
        if drawing_record.entities:
            entity_counts: Dict[str, int] = {}
            for entity in drawing_record.entities:
                entity_type = entity.entity_type.value
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

                context["entities"].append(
                    {
                        "type": entity_type,
                        "value": entity.value,
                        "confidence": entity.confidence,
                        "method": entity.extraction_method,
                    }
                )

            context["entity_counts"] = entity_counts

        # Add detections
        if drawing_record.detections:
            detection_counts: Dict[str, int] = {}
            for detection in drawing_record.detections:
                class_name = detection.class_name
                detection_counts[class_name] = detection_counts.get(class_name, 0) + 1

                context["detections"].append(
                    {"class": class_name, "confidence": detection.confidence}
                )

            context["detection_counts"] = detection_counts

        # Add review flags
        if drawing_record.review_flags:
            for flag in drawing_record.review_flags:
                context["review_flags"].append(
                    {
                        "type": flag.flag_type.value,
                        "severity": flag.severity.value,
                        "reason": flag.reason,
                        "action": flag.suggested_action,
                    }
                )

        return context

    def _write_report(
        self, html_content: str, output_path: Path, format: ReportFormat
    ) -> Path:
        """Write report to file in specified format using atomic write.

        Args:
            html_content: Rendered HTML content.
            output_path: Target file path.
            format: Output format (HTML or PDF).

        Returns:
            Path to the generated file.

        Raises:
            IOError: If file write fails.
            MissingDependencyError: If PDF format requested but weasyprint
                not available.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == ReportFormat.HTML:
            # Atomic write for HTML
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_path.parent,
                delete=False,
                suffix=".tmp",
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                tmp_file.write(html_content)

            tmp_path.replace(output_path)
            return output_path

        elif format == ReportFormat.PDF:
            return self._convert_html_to_pdf(html_content, output_path)

        else:
            raise ValueError(f"Unsupported format: {format}")

    def _render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render Jinja2 template with provided context.

        Args:
            template_name: Name of template file in template_dir.
            context: Dictionary of template variables.

        Returns:
            Rendered HTML string.

        Raises:
            TemplateError: If template rendering fails.
        """
        if self.env is None:
            raise TemplateError("Jinja2 environment not available")

        try:
            template = self.env.get_template(template_name)
            return template.render(**context)
        except Exception as e:
            logger.exception(f"Failed to render template {template_name}: {e}")
            raise TemplateError(
                f"Template rendering failed for {template_name}: {e}"
            ) from e

    def _fig_to_base64_uri(self, fig) -> str:
        """Convert matplotlib figure to base64 data URI.

        Args:
            fig: Matplotlib figure object.

        Returns:
            Base64-encoded PNG image as data URI string.

        Raises:
            ChartGenerationError: If conversion fails.
        """
        try:
            buffer = BytesIO()
            fig.savefig(buffer, format="png", dpi=self.chart_dpi, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            raise ChartGenerationError(
                f"Failed to convert figure to base64: {e}"
            ) from e

    def _generate_confidence_chart(self, confidences: List[float]) -> str:
        """Generate confidence score distribution histogram.

        Args:
            confidences: List of confidence scores (0.0 to 1.0).

        Returns:
            Base64-encoded PNG image as data URI string, or empty string
            if chart generation fails.
        """
        if not MATPLOTLIB_AVAILABLE or not confidences:
            return ""

        fig = None
        try:
            fig, ax = plt.subplots(figsize=(8, 4))

            # Histogram
            ax.hist(confidences, bins=20, edgecolor="black", alpha=0.7)
            ax.set_xlabel("Confidence Score")
            ax.set_ylabel("Count")
            ax.set_title("Confidence Score Distribution")
            ax.grid(True, alpha=0.3)

            return self._fig_to_base64_uri(fig)
        except Exception as e:
            logger.error(f"Failed to generate confidence chart: {e}")
            return ""
        finally:
            if fig is not None:
                plt.close(fig)

    def _generate_status_chart(self, batch_result: "BatchResult") -> str:
        """Generate batch status distribution pie chart.

        Args:
            batch_result: BatchResult with successful, needs_review, and
                failed counts.

        Returns:
            Base64-encoded PNG image as data URI string, or empty string
            if chart generation fails.
        """
        if not MATPLOTLIB_AVAILABLE:
            return ""

        fig = None
        try:
            fig, ax = plt.subplots(figsize=(6, 6))

            labels = ["Successful", "Needs Review", "Failed"]
            sizes = [
                batch_result.successful,
                batch_result.needs_review,
                batch_result.failed,
            ]
            colors = ["#28a745", "#ffc107", "#dc3545"]

            ax.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax.set_title("Batch Processing Status")

            return self._fig_to_base64_uri(fig)
        except Exception as e:
            logger.error(f"Failed to generate status chart: {e}")
            return ""
        finally:
            if fig is not None:
                plt.close(fig)

    def _generate_cost_chart(
        self, cost_data: Dict[str, float], title: str = "Cost Breakdown"
    ) -> str:
        """Generate horizontal bar chart for cost breakdown.

        Args:
            cost_data: Dictionary mapping categories to cost values.
            title: Chart title. Defaults to "Cost Breakdown".

        Returns:
            Base64-encoded PNG image as data URI string, or empty string
            if chart generation fails.
        """
        if not MATPLOTLIB_AVAILABLE or not cost_data:
            return ""

        fig = None
        try:
            fig, ax = plt.subplots(figsize=(8, 5))

            items = list(cost_data.keys())
            costs = list(cost_data.values())

            ax.barh(items, costs, color="steelblue")
            ax.set_xlabel("Cost (USD)")
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis="x")

            # Format x-axis as currency
            if MATPLOTLIB_AVAILABLE:
                formatter = FuncFormatter(lambda x, p: f"${x:.2f}")
                ax.xaxis.set_major_formatter(formatter)

            return self._fig_to_base64_uri(fig)
        except Exception as e:
            logger.error(f"Failed to generate cost chart: {e}")
            return ""
        finally:
            if fig is not None:
                plt.close(fig)

    def _generate_daily_trend_chart(self, daily_costs: List["DailyCost"]) -> str:
        """Generate line chart showing daily cost trends.

        Args:
            daily_costs: List of DailyCost objects with date and total_cost.

        Returns:
            Base64-encoded PNG image as data URI string, or empty string
            if chart generation fails.
        """
        if not MATPLOTLIB_AVAILABLE or not daily_costs:
            return ""

        fig = None
        try:
            fig, ax = plt.subplots(figsize=(10, 4))

            dates = [dc.date for dc in daily_costs]
            costs = [dc.total_cost for dc in daily_costs]

            ax.plot(dates, costs, marker="o", linewidth=2, markersize=6)
            ax.set_xlabel("Date")
            ax.set_ylabel("Cost (USD)")
            ax.set_title("Daily Cost Trend")
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

            # Format y-axis as currency
            if MATPLOTLIB_AVAILABLE:
                formatter = FuncFormatter(lambda y, p: f"${y:.2f}")
                ax.yaxis.set_major_formatter(formatter)

            return self._fig_to_base64_uri(fig)
        except Exception as e:
            logger.error(f"Failed to generate daily trend chart: {e}")
            return ""
        finally:
            if fig is not None:
                plt.close(fig)

    def _convert_html_to_pdf(self, html_content: str, pdf_path: Path) -> Path:
        """Convert HTML string to PDF file using WeasyPrint.

        Args:
            html_content: Complete HTML document string.
            pdf_path: Target PDF file path.

        Returns:
            Path to the generated PDF (or HTML if weasyprint unavailable).

        Raises:
            MissingDependencyError: If weasyprint is not available.
            IOError: If PDF conversion fails.
        """
        if not WEASYPRINT_AVAILABLE:
            logger.warning(
                "weasyprint not available. Install with: pip install weasyprint"
            )
            # Save as HTML instead
            html_path = pdf_path.with_suffix(".html")
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=html_path.parent,
                delete=False,
                suffix=".tmp",
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                tmp_file.write(html_content)

            tmp_path.replace(html_path)
            logger.info(f"Saved as HTML instead: {html_path}")
            return html_path

        try:
            # Atomic write for PDF
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=pdf_path.parent, delete=False, suffix=".tmp"
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                WeasyHTML(string=html_content).write_pdf(tmp_file)

            tmp_path.replace(pdf_path)
            logger.info(f"Successfully converted HTML to PDF: {pdf_path}")
            return pdf_path
        except Exception as e:
            logger.exception(f"Failed to convert to PDF: {e}")
            raise IOError(f"PDF conversion failed: {e}") from e

    def _generate_simple_batch_report(
        self, batch_result: "BatchResult", output_path: Path
    ) -> None:
        """Generate plain text batch report as fallback.

        Used when Jinja2 is not available. Creates simple formatted text
        file with key statistics.

        Args:
            batch_result: BatchResult object with statistics to report.
            output_path: Output text file path.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("BATCH PROCESSING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Batch ID: {batch_result.batch_id}\n")
            f.write(f"Total Drawings: {batch_result.total_drawings}\n")
            f.write(f"Successful: {batch_result.successful}\n")
            f.write(f"Failed: {batch_result.failed}\n")
            f.write(f"Needs Review: {batch_result.needs_review}\n")
            f.write(f"Success Rate: {batch_result.success_rate * 100:.1f}%\n")
            f.write(f"Review Rate: {batch_result.review_rate * 100:.1f}%\n")
            f.write(f"Total LLM Cost: ${batch_result.total_llm_cost:.2f}\n")
            f.write(
                f"Avg Processing Time: "
                f"{batch_result.average_processing_time:.2f}s\n"
            )

    def _generate_simple_cost_report(
        self, cost_report: "CostReport", output_path: Path
    ) -> None:
        """Generate plain text cost report as fallback.

        Used when Jinja2 is not available. Creates simple formatted text
        file with cost summary.

        Args:
            cost_report: CostReport object with cost data to report.
            output_path: Output text file path.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("COST REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Period: {cost_report.period}\n")
            f.write(f"Total Cost: ${cost_report.total_cost:.2f}\n")
            f.write(f"Total Calls: {cost_report.total_calls}\n")
            f.write(
                f"Avg per Drawing: " f"${cost_report.average_cost_per_drawing:.4f}\n"
            )

    def _generate_simple_drawing_report(
        self, drawing_record: "DrawingRecord", output_path: Path
    ) -> None:
        """Generate plain text drawing report as fallback.

        Used when Jinja2 is not available. Creates simple formatted text
        file with drawing metadata.

        Args:
            drawing_record: DrawingRecord object with drawing data.
            output_path: Output text file path.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("DRAWING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Drawing ID: {drawing_record.drawing_id}\n")
            f.write(f"Source File: {drawing_record.source_file}\n")
            f.write(f"Confidence: {drawing_record.overall_confidence:.2f}\n")
            f.write(f"Needs Review: {drawing_record.needs_review}\n")
            f.write(f"Status: {drawing_record.status}\n")
