"""
Report Generator Module

Generates visual reports (HTML/PDF) for drawing results.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import base64
from io import BytesIO

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("jinja2 not available, report generation will be limited")

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate visual reports (HTML/PDF)."""

    def __init__(self, template_dir: str = "config/templates"):
        """
        Initialize report generator.

        Args:
            template_dir: Directory containing Jinja2 templates
        """
        self.template_dir = template_dir

        if JINJA2_AVAILABLE:
            self.env = Environment(
                loader=FileSystemLoader(template_dir),
                autoescape=select_autoescape(["html", "xml"]),
            )
        else:
            self.env = None

        logger.info(f"ReportGenerator initialized with template_dir: {template_dir}")

    def generate_batch_report(
        self,
        batch_result,
        drawing_records: List,
        output_path: str,
        format: str = "html",
    ) -> None:
        """
        Generate comprehensive batch processing report.

        Args:
            batch_result: BatchResult object
            drawing_records: List of DrawingRecords
            output_path: Output file path
            format: Report format ('html' or 'pdf')

        Raises:
            ValueError: If invalid format
            IOError: If file write fails
        """
        logger.info(f"Generating batch report: {output_path}")

        if not JINJA2_AVAILABLE:
            self._generate_simple_batch_report(batch_result, output_path)
            return

        # Prepare context
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

        # Add drawing summaries
        for record in drawing_records[:20]:  # Limit to first 20
            context["drawings"].append(
                {
                    "drawing_id": record.drawing_id,
                    "source_file": Path(record.source_file).name,
                    "confidence": record.overall_confidence,
                    "needs_review": record.needs_review,
                    "entity_count": len(record.entities) if record.entities else 0,
                    "detection_count": (
                        len(record.detections) if record.detections else 0
                    ),
                }
            )

        # Generate charts
        if MATPLOTLIB_AVAILABLE:
            context["confidence_chart"] = self._generate_confidence_chart(
                [r.overall_confidence for r in drawing_records]
            )
            context["status_chart"] = self._generate_status_chart(batch_result)

        # Render template
        html_content = self._render_template("batch_report.html", context)

        # Write output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "html":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
        elif format == "pdf":
            self._convert_html_to_pdf(html_content, str(output_path))

        logger.info(f"Successfully generated batch report: {output_path}")

    def generate_cost_report(
        self, cost_report, output_path: str, format: str = "html"
    ) -> None:
        """
        Generate cost analysis report with charts.

        Args:
            cost_report: CostReport object
            output_path: Output file path
            format: Report format ('html' or 'pdf')

        Raises:
            ValueError: If invalid format
            IOError: If file write fails
        """
        logger.info(f"Generating cost report: {output_path}")

        if not JINJA2_AVAILABLE:
            self._generate_simple_cost_report(cost_report, output_path)
            return

        # Prepare context
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

        # Generate charts
        if MATPLOTLIB_AVAILABLE:
            context["use_case_chart"] = self._generate_cost_chart(
                cost_report.cost_by_use_case, title="Cost by Use Case"
            )
            context["provider_chart"] = self._generate_cost_chart(
                cost_report.cost_by_provider, title="Cost by Provider"
            )
            context["daily_trend_chart"] = self._generate_daily_trend_chart(
                cost_report.daily_costs
            )

        # Render template
        html_content = self._render_template("cost_report.html", context)

        # Write output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "html":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
        elif format == "pdf":
            self._convert_html_to_pdf(html_content, str(output_path))

        logger.info(f"Successfully generated cost report: {output_path}")

    def generate_drawing_report(
        self, drawing_record, output_path: str, format: str = "html"
    ) -> None:
        """
        Generate single drawing detailed report.

        Args:
            drawing_record: DrawingRecord object
            output_path: Output file path
            format: Report format ('html' or 'pdf')

        Raises:
            ValueError: If invalid format
            IOError: If file write fails
        """
        logger.info(f"Generating drawing report: {output_path}")

        if not JINJA2_AVAILABLE:
            self._generate_simple_drawing_report(drawing_record, output_path)
            return

        # Prepare context
        context = {
            "drawing_id": drawing_record.drawing_id,
            "source_file": drawing_record.source_file,
            "processing_timestamp": drawing_record.processing_timestamp.isoformat(),
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
            from ..models.data_structures import EntityType

            entity_counts = {}
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
            detection_counts = {}
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

        # Render template
        html_content = self._render_template("drawing_report.html", context)

        # Write output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "html":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
        elif format == "pdf":
            self._convert_html_to_pdf(html_content, str(output_path))

        logger.info(f"Successfully generated drawing report: {output_path}")

    def _render_template(self, template_name: str, context: Dict) -> str:
        """
        Render Jinja2 template with context.

        Args:
            template_name: Template filename
            context: Template context

        Returns:
            Rendered HTML string
        """
        try:
            template = self.env.get_template(template_name)
            return template.render(**context)
        except Exception as e:
            logger.error(f"Failed to render template {template_name}: {e}")
            # Return simple fallback HTML
            return self._generate_fallback_html(context)

    def _generate_confidence_chart(self, confidences: List[float]) -> str:
        """
        Generate confidence distribution chart.

        Args:
            confidences: List of confidence scores

        Returns:
            Base64-encoded image string
        """
        if not MATPLOTLIB_AVAILABLE or not confidences:
            return ""

        try:
            fig, ax = plt.subplots(figsize=(8, 4))

            # Histogram
            ax.hist(confidences, bins=20, edgecolor="black", alpha=0.7)
            ax.set_xlabel("Confidence Score")
            ax.set_ylabel("Count")
            ax.set_title("Confidence Score Distribution")
            ax.grid(True, alpha=0.3)

            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)

            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            logger.error(f"Failed to generate confidence chart: {e}")
            return ""

    def _generate_status_chart(self, batch_result) -> str:
        """Generate batch status pie chart."""
        if not MATPLOTLIB_AVAILABLE:
            return ""

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
                sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
            )
            ax.set_title("Batch Processing Status")

            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)

            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            logger.error(f"Failed to generate status chart: {e}")
            return ""

    def _generate_cost_chart(
        self, cost_data: Dict[str, float], title: str = "Cost Breakdown"
    ) -> str:
        """Generate cost breakdown chart."""
        if not MATPLOTLIB_AVAILABLE or not cost_data:
            return ""

        try:
            fig, ax = plt.subplots(figsize=(8, 5))

            items = list(cost_data.keys())
            costs = list(cost_data.values())

            ax.barh(items, costs, color="steelblue")
            ax.set_xlabel("Cost (USD)")
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis="x")

            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)

            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            logger.error(f"Failed to generate cost chart: {e}")
            return ""

    def _generate_daily_trend_chart(self, daily_costs: List) -> str:
        """Generate daily cost trend chart."""
        if not MATPLOTLIB_AVAILABLE or not daily_costs:
            return ""

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

            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)

            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            logger.error(f"Failed to generate daily trend chart: {e}")
            return ""

    def _convert_html_to_pdf(self, html_content: str, pdf_path: str) -> None:
        """
        Convert HTML report to PDF (if weasyprint available).

        Args:
            html_content: HTML string
            pdf_path: Output PDF path
        """
        try:
            from weasyprint import HTML

            HTML(string=html_content).write_pdf(pdf_path)
            logger.info(f"Successfully converted HTML to PDF: {pdf_path}")
        except ImportError:
            logger.warning("weasyprint not available, saving as HTML instead")
            # Save as HTML
            html_path = pdf_path.replace(".pdf", ".html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
        except Exception as e:
            logger.error(f"Failed to convert to PDF: {e}")
            raise

    def _generate_simple_batch_report(self, batch_result, output_path: str):
        """Generate simple text-based batch report (fallback)."""
        with open(output_path, "w") as f:
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
                f"Avg Processing Time: {batch_result.average_processing_time:.2f}s\n"
            )

    def _generate_simple_cost_report(self, cost_report, output_path: str):
        """Generate simple text-based cost report (fallback)."""
        with open(output_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("COST REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Period: {cost_report.period}\n")
            f.write(f"Total Cost: ${cost_report.total_cost:.2f}\n")
            f.write(f"Total Calls: {cost_report.total_calls}\n")
            f.write(f"Avg per Drawing: ${cost_report.average_cost_per_drawing:.4f}\n")

    def _generate_simple_drawing_report(self, drawing_record, output_path: str):
        """Generate simple text-based drawing report (fallback)."""
        with open(output_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("DRAWING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Drawing ID: {drawing_record.drawing_id}\n")
            f.write(f"Source File: {drawing_record.source_file}\n")
            f.write(f"Confidence: {drawing_record.overall_confidence:.2f}\n")
            f.write(f"Needs Review: {drawing_record.needs_review}\n")
            f.write(f"Status: {drawing_record.status}\n")

    def _generate_fallback_html(self, context: Dict) -> str:
        """Generate simple fallback HTML when template fails."""
        html = "<html><head><title>Report</title></head><body>"
        html += "<h1>Report</h1>"
        html += "<pre>" + str(context) + "</pre>"
        html += "</body></html>"
        return html
