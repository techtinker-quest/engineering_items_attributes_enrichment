"""
Cost tracking module for LLM usage.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

from ..models.data_structures import CostReport, DailyCost, TokenUsageSummary
from ..models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class CostTracker:
    """
    Track and report LLM API costs.

    Monitors costs by:
    - Time period (daily, weekly, monthly)
    - Provider
    - Model
    - Use case
    - Drawing
    """

    def __init__(self, db_manager=None):
        """
        Initialize cost tracker.

        Args:
            db_manager: Optional DatabaseManager for persistent tracking
        """
        self.db = db_manager
        logger.info("CostTracker initialized")

    def track_call(
        self,
        provider: str,
        model: str,
        tokens: "TokenUsage",
        cost: float,
        drawing_id: str,
        use_case: str,
    ) -> None:
        """
        Track an LLM API call.

        Args:
            provider: Provider name (openai, anthropic, google)
            model: Model identifier
            tokens: TokenUsage object
            cost: Cost in USD
            drawing_id: Drawing identifier
            use_case: Use case type
        """
        if self.db:
            try:
                self.db.store_llm_usage(
                    drawing_id=drawing_id,
                    provider=provider,
                    model=model,
                    tokens=tokens,
                    cost=cost,
                    use_case=use_case,
                )
            except Exception as e:
                logger.error(f"Failed to track LLM call: {e}")

        logger.debug(
            f"Tracked LLM call: {provider}/{model} - "
            f"${cost:.4f} ({tokens.input_tokens}→{tokens.output_tokens} tokens)"
        )

    def get_daily_cost(self, date: Optional[datetime] = None) -> float:
        """
        Get total cost for a specific day.

        Args:
            date: Date to query (default: today)

        Returns:
            Total cost in USD
        """
        if not self.db:
            return 0.0

        if date is None:
            date = datetime.now()

        # Query usage for the day
        start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1)

        usage_records = self.db.get_llm_usage(start_date, end_date)

        total_cost = sum(record["cost_usd"] for record in usage_records)
        return total_cost

    def get_cost_by_use_case(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, float]:
        """
        Get costs broken down by use case.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary mapping use case to cost
        """
        if not self.db:
            return {}

        usage_records = self.db.get_llm_usage(start_date, end_date)

        costs_by_use_case = {}
        for record in usage_records:
            use_case = record["use_case"]
            cost = record["cost_usd"]
            costs_by_use_case[use_case] = costs_by_use_case.get(use_case, 0.0) + cost

        return costs_by_use_case

    def get_cost_by_provider(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, float]:
        """
        Get costs broken down by provider.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary mapping provider to cost
        """
        if not self.db:
            return {}

        usage_records = self.db.get_llm_usage(start_date, end_date)

        costs_by_provider = {}
        for record in usage_records:
            provider = record["provider"]
            cost = record["cost_usd"]
            costs_by_provider[provider] = costs_by_provider.get(provider, 0.0) + cost

        return costs_by_provider

    def get_cost_by_model(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, float]:
        """
        Get costs broken down by model.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary mapping model to cost
        """
        if not self.db:
            return {}

        usage_records = self.db.get_llm_usage(start_date, end_date)

        costs_by_model = {}
        for record in usage_records:
            model = record["model"]
            cost = record["cost_usd"]
            costs_by_model[model] = costs_by_model.get(model, 0.0) + cost

        return costs_by_model

    def get_cost_by_drawing(self, drawing_id: str) -> float:
        """
        Get total LLM cost for a specific drawing.

        Args:
            drawing_id: Drawing identifier

        Returns:
            Total cost in USD
        """
        if not self.db:
            return 0.0

        # Query all usage for this drawing
        # Use very wide date range
        start_date = datetime(2000, 1, 1)
        end_date = datetime.now() + timedelta(days=1)

        usage_records = self.db.get_llm_usage(start_date, end_date)

        # Filter by drawing_id
        drawing_records = [r for r in usage_records if r["drawing_id"] == drawing_id]

        total_cost = sum(r["cost_usd"] for r in drawing_records)
        return total_cost

    def generate_cost_report(
        self, period: str, start_date: datetime, end_date: datetime
    ) -> CostReport:
        """
        Generate comprehensive cost report.

        Args:
            period: Period name (e.g., 'daily', 'weekly', 'monthly')
            start_date: Start date
            end_date: End date

        Returns:
            CostReport object
        """
        if not self.db:
            return self._empty_cost_report(period, start_date, end_date)

        usage_records = self.db.get_llm_usage(start_date, end_date)

        if not usage_records:
            return self._empty_cost_report(period, start_date, end_date)

        # Calculate totals
        total_cost = sum(r["cost_usd"] for r in usage_records)
        total_calls = len(usage_records)

        # Breakdowns
        cost_by_use_case = self.get_cost_by_use_case(start_date, end_date)
        cost_by_provider = self.get_cost_by_provider(start_date, end_date)
        cost_by_model = self.get_cost_by_model(start_date, end_date)

        # Daily costs
        daily_costs = self._calculate_daily_costs(usage_records, start_date, end_date)

        # Top drawings by cost
        drawing_costs = {}
        for record in usage_records:
            drawing_id = record["drawing_id"]
            cost = record["cost_usd"]
            drawing_costs[drawing_id] = drawing_costs.get(drawing_id, 0.0) + cost

        top_drawings = sorted(drawing_costs.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]

        # Average cost per drawing
        num_drawings = len(set(r["drawing_id"] for r in usage_records))
        avg_cost_per_drawing = total_cost / num_drawings if num_drawings > 0 else 0.0

        # Token usage summary
        token_summary = self._calculate_token_summary(usage_records)

        report = CostReport(
            period=period,
            start_date=start_date,
            end_date=end_date,
            total_cost=total_cost,
            total_calls=total_calls,
            cost_by_use_case=cost_by_use_case,
            cost_by_provider=cost_by_provider,
            cost_by_model=cost_by_model,
            daily_costs=daily_costs,
            top_drawings_by_cost=top_drawings,
            average_cost_per_drawing=avg_cost_per_drawing,
            token_usage_summary=token_summary,
        )

        logger.info(
            f"Cost report generated: {period} - "
            f"${total_cost:.2f} ({total_calls} calls)"
        )

        return report

    def _calculate_daily_costs(
        self, usage_records: list, start_date: datetime, end_date: datetime
    ) -> list:
        """Calculate daily cost breakdown."""
        daily_costs_dict = {}

        for record in usage_records:
            timestamp = datetime.fromisoformat(record["timestamp"])
            date_key = timestamp.date()

            if date_key not in daily_costs_dict:
                daily_costs_dict[date_key] = {"cost": 0.0, "calls": 0}

            daily_costs_dict[date_key]["cost"] += record["cost_usd"]
            daily_costs_dict[date_key]["calls"] += 1

        # Convert to list of DailyCost objects
        daily_costs = [
            DailyCost(
                date=datetime.combine(date_key, datetime.min.time()),
                total_cost=data["cost"],
                call_count=data["calls"],
            )
            for date_key, data in sorted(daily_costs_dict.items())
        ]

        return daily_costs

    def _calculate_token_summary(self, usage_records: list) -> TokenUsageSummary:
        """Calculate token usage statistics."""
        total_input = sum(r["tokens_input"] for r in usage_records)
        total_output = sum(r["tokens_output"] for r in usage_records)
        total_images = len([r for r in usage_records if r.get("image_count", 0) > 0])

        num_calls = len(usage_records)
        avg_input = total_input / num_calls if num_calls > 0 else 0
        avg_output = total_output / num_calls if num_calls > 0 else 0

        return TokenUsageSummary(
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_images=total_images,
            average_input_per_call=avg_input,
            average_output_per_call=avg_output,
        )

    def _empty_cost_report(
        self, period: str, start_date: datetime, end_date: datetime
    ) -> CostReport:
        """Create empty cost report."""
        return CostReport(
            period=period,
            start_date=start_date,
            end_date=end_date,
            total_cost=0.0,
            total_calls=0,
            cost_by_use_case={},
            cost_by_provider={},
            cost_by_model={},
            daily_costs=[],
            top_drawings_by_cost=[],
            average_cost_per_drawing=0.0,
            token_usage_summary=TokenUsageSummary(0, 0, 0, 0.0, 0.0),
        )

    def export_cost_data(
        self,
        start_date: datetime,
        end_date: datetime,
        output_path: str,
        format: str = "csv",
    ) -> None:
        """
        Export cost data to file.

        Args:
            start_date: Start date
            end_date: End date
            output_path: Output file path
            format: Export format ('csv' or 'json')
        """
        if not self.db:
            logger.warning("No database available for export")
            return

        usage_records = self.db.get_llm_usage(start_date, end_date)

        if format == "csv":
            self._export_csv(usage_records, output_path)
        elif format == "json":
            self._export_json(usage_records, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported cost data to {output_path}")

    def _export_csv(self, usage_records: list, output_path: str):
        """Export to CSV format."""
        import csv

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "drawing_id",
                    "provider",
                    "model",
                    "use_case",
                    "tokens_input",
                    "tokens_output",
                    "cost_usd",
                ],
            )
            writer.writeheader()
            writer.writerows(usage_records)

    def _export_json(self, usage_records: list, output_path: str):
        """Export to JSON format."""
        import json

        with open(output_path, "w") as f:
            json.dump(usage_records, f, indent=2, default=str)
