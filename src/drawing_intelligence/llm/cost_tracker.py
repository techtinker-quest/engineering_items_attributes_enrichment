"""
Cost tracking module for LLM usage.

This module provides comprehensive tracking and reporting of LLM API costs
across multiple dimensions including time periods, providers, models, use cases,
and individual drawings. All timestamps are handled in UTC.
"""

import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..database.database_manager import DatabaseManager
from ..models.data_structures import (
    CostReport,
    DailyCost,
    TokenUsage,
    TokenUsageSummary,
)
from ..models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class CostRecord:
    """Structured representation of an LLM usage record.

    Attributes:
        timestamp: UTC timestamp of the API call.
        drawing_id: Unique identifier for the drawing being processed.
        provider: Provider name (e.g., 'openai', 'anthropic', 'google').
        model: Model identifier (e.g., 'gpt-4o-2024-08-06').
        use_case: Type of operation (e.g., 'entity_extraction').
        tokens_input: Number of input tokens consumed.
        tokens_output: Number of output tokens generated.
        image_count: Number of images processed (0 if none).
        cost_usd: Total cost in USD for this API call.
    """

    timestamp: datetime
    drawing_id: str
    provider: str
    model: str
    use_case: str
    tokens_input: int
    tokens_output: int
    image_count: int
    cost_usd: float

    @classmethod
    def from_dict(cls, record: Dict[str, Any]) -> "CostRecord":
        """Create CostRecord from database dictionary.

        Args:
            record: Dictionary from database query result.

        Returns:
            CostRecord instance with parsed values.
        """
        return cls(
            timestamp=datetime.fromisoformat(record["timestamp"]),
            drawing_id=record["drawing_id"],
            provider=record["provider"],
            model=record["model"],
            use_case=record["use_case"],
            tokens_input=record["tokens_input"],
            tokens_output=record["tokens_output"],
            image_count=record.get("image_count", 0),
            cost_usd=record["cost_usd"],
        )


class CostTracker:
    """Track and report LLM API costs across multiple dimensions.

    This class provides comprehensive cost tracking for LLM API usage,
    supporting aggregation by time period, provider, model, use case,
    and individual drawings. All costs are tracked in USD and timestamps
    are handled in UTC.

    Attributes:
        db: Optional DatabaseManager instance for persistent cost tracking.
        top_n_default: Default number of top items to include in reports.

    Example:
        >>> tracker = CostTracker(db_manager)
        >>> tracker.track_call(
        ...     provider="openai",
        ...     model="gpt-4o",
        ...     tokens=token_usage,
        ...     cost=0.05,
        ...     drawing_id="DWG-001",
        ...     use_case="entity_extraction"
        ... )
        >>> report = tracker.generate_cost_report("weekly", start, end)
    """

    # Valid period types for validation
    VALID_PERIODS = {"daily", "weekly", "monthly", "custom"}

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        top_n_default: int = 10,
    ) -> None:
        """Initialize cost tracker.

        Args:
            db_manager: Optional DatabaseManager for persistent tracking.
                If None, cost tracking will log but not persist data.
            top_n_default: Default number of top items to include in
                reports (default: 10).

        Raises:
            ValueError: If top_n_default is less than 1.
        """
        if top_n_default < 1:
            raise ValueError(f"top_n_default must be >= 1, got {top_n_default}")

        self.db = db_manager
        self.top_n_default = top_n_default
        logger.info(
            f"CostTracker initialized (db={'available' if db_manager else 'none'}, "
            f"top_n_default={top_n_default})"
        )

    # ============================================================================
    # Public API - Cost Tracking
    # ============================================================================

    def track_call(
        self,
        provider: str,
        model: str,
        tokens: TokenUsage,
        cost: float,
        drawing_id: str,
        use_case: str,
    ) -> None:
        """Track a single LLM API call and persist usage data.

        Records token usage and cost information for a specific LLM API call.
        If database manager is unavailable, logs the call but does not persist.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic', 'google').
            model: Model identifier (e.g., 'gpt-4o-2024-08-06').
            tokens: TokenUsage object containing input/output token counts
                and optional image count.
            cost: Total cost of the API call in USD.
            drawing_id: Unique identifier for the drawing being processed.
            use_case: Type of operation (e.g., 'entity_extraction',
                'ocr_verification', 'drawing_assessment').

        Raises:
            ValueError: If cost is negative or tokens are invalid.

        Note:
            This method is designed to never interrupt processing pipeline
            even if cost tracking fails. Database errors are logged but
            not raised.
        """
        # Validate inputs
        self._validate_cost_inputs(provider, model, cost, drawing_id, use_case)

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
                logger.error(
                    f"Failed to track LLM call for drawing {drawing_id}: "
                    f"{provider}/{model} - {e}",
                    exc_info=True,
                )
                return

        logger.debug(
            f"Tracked LLM call: {provider}/{model} - "
            f"${cost:.4f} ({tokens.input_tokens}→{tokens.output_tokens} tokens) "
            f"[{use_case}]"
        )

    # ============================================================================
    # Public API - Cost Queries
    # ============================================================================

    def get_daily_cost(self, date: Optional[datetime] = None) -> float:
        """Get total LLM API cost for a specific day.

        Args:
            date: Date to query (timezone-aware or naive, treated as UTC).
                If None, defaults to current UTC date.

        Returns:
            Total cost in USD for all LLM API calls on the specified date.
            Returns 0.0 if no database is available or no records exist.

        Example:
            >>> cost = tracker.get_daily_cost(datetime(2025, 11, 8, tzinfo=timezone.utc))
            >>> print(f"Cost on 2025-11-08: ${cost:.2f}")
        """
        if not self.db:
            return 0.0

        if date is None:
            date = datetime.now(timezone.utc)

        # Ensure timezone-aware
        date = self._ensure_utc(date)

        # Query usage for the day
        start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1)

        try:
            usage_records = self.db.get_llm_usage(start_date, end_date)
            total_cost = sum(record["cost_usd"] for record in usage_records)
            return total_cost
        except Exception as e:
            logger.error(f"Failed to get daily cost for {date.date()}: {e}")
            return 0.0

    def get_cost_by_use_case(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, float]:
        """Get costs aggregated by use case type.

        Args:
            start_date: Start of date range (inclusive, UTC).
            end_date: End of date range (exclusive, UTC).

        Returns:
            Dictionary mapping use case names to total costs in USD.
            Empty dict if no database is available or no records exist.

        Example:
            >>> costs = tracker.get_cost_by_use_case(start, end)
            >>> print(f"Entity extraction: ${costs.get('entity_extraction', 0):.2f}")
        """
        return self._get_aggregated_costs(start_date, end_date, "use_case")

    def get_cost_by_provider(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, float]:
        """Get costs aggregated by LLM provider.

        Args:
            start_date: Start of date range (inclusive, UTC).
            end_date: End of date range (exclusive, UTC).

        Returns:
            Dictionary mapping provider names to total costs in USD.
            Empty dict if no database is available or no records exist.
        """
        return self._get_aggregated_costs(start_date, end_date, "provider")

    def get_cost_by_model(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, float]:
        """Get costs aggregated by model identifier.

        Args:
            start_date: Start of date range (inclusive, UTC).
            end_date: End of date range (exclusive, UTC).

        Returns:
            Dictionary mapping model identifiers to total costs in USD.
            Empty dict if no database is available or no records exist.
        """
        return self._get_aggregated_costs(start_date, end_date, "model")

    def get_cost_by_drawing(
        self,
        drawing_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> float:
        """Get total LLM cost for a specific drawing.

        Calculates cumulative cost across all LLM API calls made during
        processing of the specified drawing.

        Args:
            drawing_id: Unique identifier for the drawing.
            start_date: Optional start date filter (inclusive, UTC).
            end_date: Optional end date filter (exclusive, UTC).

        Returns:
            Total cost in USD for all LLM operations on this drawing.
            Returns 0.0 if no database is available or no records exist.

        Note:
            If date range is not specified, queries all historical records.
        """
        if not self.db:
            return 0.0

        if not drawing_id or not drawing_id.strip():
            logger.warning("Empty drawing_id provided to get_cost_by_drawing")
            return 0.0

        # Use all history if dates not specified
        if start_date is None:
            start_date = datetime.min.replace(tzinfo=timezone.utc)
        if end_date is None:
            end_date = datetime.now(timezone.utc)

        # Ensure timezone-aware
        start_date = self._ensure_utc(start_date)
        end_date = self._ensure_utc(end_date)

        try:
            usage_records = self.db.get_llm_usage(
                start_date, end_date, drawing_id=drawing_id
            )
            total_cost = sum(r["cost_usd"] for r in usage_records)
            return total_cost
        except Exception as e:
            logger.error(f"Failed to get cost for drawing {drawing_id}: {e}")
            return 0.0

    # ============================================================================
    # Public API - Cost Reporting
    # ============================================================================

    def generate_cost_report(
        self,
        period: str,
        start_date: datetime,
        end_date: datetime,
        top_n: Optional[int] = None,
    ) -> CostReport:
        """Generate comprehensive cost report with multiple breakdowns.

        Creates a detailed report including total costs, call counts,
        breakdowns by provider/model/use case, daily trends, top drawings,
        and token usage statistics. Fetches data from database once for
        efficiency.

        Args:
            period: Descriptive name for the reporting period
                (e.g., 'daily', 'weekly', 'monthly', 'custom').
            start_date: Start of reporting period (inclusive, UTC).
            end_date: End of reporting period (exclusive, UTC).
            top_n: Number of top drawings to include. If None, uses
                the instance's top_n_default value.

        Returns:
            CostReport object containing all cost metrics and breakdowns.
            Returns empty report if no database or no records exist.

        Raises:
            ValueError: If period is not a valid period type or date range
                is invalid.

        Example:
            >>> report = tracker.generate_cost_report(
            ...     "weekly",
            ...     datetime(2025, 11, 1, tzinfo=timezone.utc),
            ...     datetime(2025, 11, 8, tzinfo=timezone.utc)
            ... )
            >>> print(f"Total: ${report.total_cost:.2f}")
            >>> print(f"Calls: {report.total_calls}")
        """
        # Validate inputs
        if period not in self.VALID_PERIODS:
            raise ValueError(
                f"Invalid period '{period}'. Must be one of {self.VALID_PERIODS}"
            )

        start_date = self._ensure_utc(start_date)
        end_date = self._ensure_utc(end_date)
        self._validate_date_range(start_date, end_date)

        if top_n is None:
            top_n = self.top_n_default
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")

        if not self.db:
            return self._empty_cost_report(period, start_date, end_date)

        try:
            # SINGLE DATABASE FETCH - all subsequent operations use this data
            raw_records = self.db.get_llm_usage(start_date, end_date)

            if not raw_records:
                return self._empty_cost_report(period, start_date, end_date)

            # Convert to typed CostRecord objects
            cost_records = [CostRecord.from_dict(r) for r in raw_records]

            # Calculate all metrics from in-memory data
            total_cost = sum(r.cost_usd for r in cost_records)
            total_calls = len(cost_records)

            cost_by_use_case = self._aggregate_costs_by_field(cost_records, "use_case")
            cost_by_provider = self._aggregate_costs_by_field(cost_records, "provider")
            cost_by_model = self._aggregate_costs_by_field(cost_records, "model")

            daily_costs = self._calculate_daily_costs(cost_records)
            top_drawings = self._calculate_top_drawings(cost_records, top_n)
            token_summary = self._calculate_token_summary(cost_records)

            # Calculate average cost per drawing
            unique_drawings = len(set(r.drawing_id for r in cost_records))
            avg_cost_per_drawing = (
                total_cost / unique_drawings if unique_drawings > 0 else 0.0
            )

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
                f"Cost report generated: {period} ({start_date.date()} to {end_date.date()}) - "
                f"${total_cost:.2f} ({total_calls} calls, {unique_drawings} drawings)"
            )

            return report

        except Exception as e:
            logger.error(
                f"Failed to generate cost report for {period} "
                f"({start_date.date()} to {end_date.date()}): {e}",
                exc_info=True,
            )
            return self._empty_cost_report(period, start_date, end_date)

    # ============================================================================
    # Public API - Data Export
    # ============================================================================

    def export_cost_data(
        self,
        start_date: datetime,
        end_date: datetime,
        output_path: str,
        export_format: str = "csv",
    ) -> None:
        """Export cost data to file in CSV or JSON format.

        Args:
            start_date: Start of date range (inclusive, UTC).
            end_date: End of date range (exclusive, UTC).
            output_path: Full path for output file.
            export_format: Output format, either 'csv' or 'json'.

        Raises:
            ValueError: If export_format is not 'csv' or 'json', or if
                date range is invalid, or if output_path is invalid.
            IOError: If file writing fails.

        Note:
            Logs warning and returns early if no database is available.
        """
        if not self.db:
            logger.warning("No database available for export")
            return

        # Validate inputs
        start_date = self._ensure_utc(start_date)
        end_date = self._ensure_utc(end_date)
        self._validate_date_range(start_date, end_date)
        self._validate_output_path(output_path)

        if export_format not in {"csv", "json"}:
            raise ValueError(
                f"Unsupported format: {export_format}. Must be 'csv' or 'json'."
            )

        try:
            usage_records = self.db.get_llm_usage(start_date, end_date)

            if not usage_records:
                logger.warning(
                    f"No cost data to export for period {start_date.date()} to {end_date.date()}"
                )

            if export_format == "csv":
                self._export_csv(usage_records, output_path)
            else:  # json
                self._export_json(usage_records, output_path)

            logger.info(f"Exported {len(usage_records)} cost records to {output_path}")

        except (IOError, OSError) as e:
            logger.error(f"Failed to write export file {output_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to export cost data: {e}", exc_info=True)
            raise

    # ============================================================================
    # Private Helpers - Aggregation
    # ============================================================================

    def _get_aggregated_costs(
        self, start_date: datetime, end_date: datetime, field: str
    ) -> Dict[str, float]:
        """Get costs aggregated by a specific field.

        Args:
            start_date: Start of date range (inclusive, UTC).
            end_date: End of date range (exclusive, UTC).
            field: Field name to aggregate by ('use_case', 'provider', 'model').

        Returns:
            Dictionary mapping field values to total costs in USD.
            Empty dict if no database is available or no records exist.
        """
        if not self.db:
            return {}

        start_date = self._ensure_utc(start_date)
        end_date = self._ensure_utc(end_date)
        self._validate_date_range(start_date, end_date)

        try:
            raw_records = self.db.get_llm_usage(start_date, end_date)
            cost_records = [CostRecord.from_dict(r) for r in raw_records]
            return self._aggregate_costs_by_field(cost_records, field)
        except Exception as e:
            logger.error(f"Failed to get cost by {field}: {e}")
            return {}

    def _aggregate_costs_by_field(
        self, cost_records: List[CostRecord], field: str
    ) -> Dict[str, float]:
        """Aggregate costs by a specific field from CostRecord list.

        Args:
            cost_records: List of CostRecord objects.
            field: Field name to aggregate by (must be a CostRecord attribute).

        Returns:
            Dictionary mapping field values to total costs.
        """
        costs_by_field: Dict[str, float] = defaultdict(float)
        for record in cost_records:
            field_value = getattr(record, field)
            costs_by_field[field_value] += record.cost_usd
        return dict(costs_by_field)

    def _calculate_daily_costs(self, cost_records: List[CostRecord]) -> List[DailyCost]:
        """Calculate daily cost breakdown from CostRecord list.

        Args:
            cost_records: List of CostRecord objects.

        Returns:
            List of DailyCost objects sorted by date.
        """
        daily_data: Dict[Any, Dict[str, Any]] = defaultdict(
            lambda: {"cost": 0.0, "calls": 0}
        )

        for record in cost_records:
            date_key = record.timestamp.date()
            daily_data[date_key]["cost"] += record.cost_usd
            daily_data[date_key]["calls"] += 1

        # Convert to list of DailyCost objects, sorted by date
        daily_costs = [
            DailyCost(
                date=datetime.combine(date_key, datetime.min.time()).replace(
                    tzinfo=timezone.utc
                ),
                total_cost=data["cost"],
                call_count=data["calls"],
            )
            for date_key, data in sorted(daily_data.items())
        ]

        return daily_costs

    def _calculate_top_drawings(
        self, cost_records: List[CostRecord], top_n: int
    ) -> List[Tuple[str, float]]:
        """Calculate top N drawings by total cost.

        Args:
            cost_records: List of CostRecord objects.
            top_n: Number of top drawings to return.

        Returns:
            List of (drawing_id, total_cost) tuples sorted by cost descending.
        """
        drawing_costs: Dict[str, float] = defaultdict(float)
        for record in cost_records:
            drawing_costs[record.drawing_id] += record.cost_usd

        top_drawings = sorted(drawing_costs.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]

        return top_drawings

    def _calculate_token_summary(
        self, cost_records: List[CostRecord]
    ) -> TokenUsageSummary:
        """Calculate aggregate token usage statistics.

        Args:
            cost_records: List of CostRecord objects.

        Returns:
            TokenUsageSummary with totals and averages for input/output
            tokens and image counts.
        """
        if not cost_records:
            return TokenUsageSummary(0, 0, 0, 0.0, 0.0)

        total_input = sum(r.tokens_input for r in cost_records)
        total_output = sum(r.tokens_output for r in cost_records)
        total_images = sum(r.image_count for r in cost_records)

        num_calls = len(cost_records)
        avg_input = total_input / num_calls
        avg_output = total_output / num_calls

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
        """Create an empty cost report for periods with no data.

        Args:
            period: Descriptive name for the reporting period.
            start_date: Start of reporting period (UTC).
            end_date: End of reporting period (UTC).

        Returns:
            CostReport with all numeric fields set to zero.
        """
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

    # ============================================================================
    # Private Helpers - Export
    # ============================================================================

    def _export_csv(
        self, usage_records: List[Dict[str, Any]], output_path: str
    ) -> None:
        """Export usage records to CSV format.

        Args:
            usage_records: List of LLM usage record dictionaries.
            output_path: Full path for output CSV file.

        Raises:
            IOError: If file writing fails.
        """
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            if not usage_records:
                # Write header only
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
                        "image_count",
                        "cost_usd",
                    ],
                )
                writer.writeheader()
            else:
                writer = csv.DictWriter(f, fieldnames=usage_records[0].keys())
                writer.writeheader()
                writer.writerows(usage_records)

    def _export_json(
        self, usage_records: List[Dict[str, Any]], output_path: str
    ) -> None:
        """Export usage records to JSON format.

        Args:
            usage_records: List of LLM usage record dictionaries.
            output_path: Full path for output JSON file.

        Raises:
            IOError: If file writing fails.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(usage_records, f, indent=2, default=str)

    # ============================================================================
    # Private Helpers - Validation
    # ============================================================================

    def _validate_date_range(self, start_date: datetime, end_date: datetime) -> None:
        """Validate that date range is valid.

        Args:
            start_date: Start date (UTC).
            end_date: End date (UTC).

        Raises:
            ValueError: If start_date >= end_date.
        """
        if start_date >= end_date:
            raise ValueError(
                f"Invalid date range: start_date ({start_date}) must be before "
                f"end_date ({end_date})"
            )

    def _validate_cost_inputs(
        self, provider: str, model: str, cost: float, drawing_id: str, use_case: str
    ) -> None:
        """Validate inputs for cost tracking.

        Args:
            provider: Provider name.
            model: Model identifier.
            cost: Cost in USD.
            drawing_id: Drawing identifier.
            use_case: Use case type.

        Raises:
            ValueError: If any input is invalid.
        """
        if not provider or not provider.strip():
            raise ValueError("Provider must be a non-empty string")
        if not model or not model.strip():
            raise ValueError("Model must be a non-empty string")
        if cost < 0:
            raise ValueError(f"Cost must be non-negative, got {cost}")
        if not drawing_id or not drawing_id.strip():
            raise ValueError("Drawing ID must be a non-empty string")
        if not use_case or not use_case.strip():
            raise ValueError("Use case must be a non-empty string")

    def _validate_output_path(self, output_path: str) -> None:
        """Validate output path for export.

        Args:
            output_path: Path to validate.

        Raises:
            ValueError: If path is invalid or potentially dangerous.
        """
        if not output_path or not output_path.strip():
            raise ValueError("Output path must be a non-empty string")

        # Convert to Path object for validation
        path = Path(output_path)

        # Check for path traversal attempts
        try:
            path.resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid output path: {e}")

        # Ensure parent directory exists or can be created
        parent_dir = path.parent
        if not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise ValueError(f"Cannot create parent directory {parent_dir}: {e}")

    def _ensure_utc(self, dt: datetime) -> datetime:
        """Ensure datetime is timezone-aware in UTC.

        Args:
            dt: Datetime to convert (naive or aware).

        Returns:
            Timezone-aware datetime in UTC.
        """
        if dt.tzinfo is None:
            # Naive datetime - assume UTC
            return dt.replace(tzinfo=timezone.utc)
        else:
            # Convert to UTC if not already
            return dt.astimezone(timezone.utc)
