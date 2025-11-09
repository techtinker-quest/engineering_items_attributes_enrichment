"""
Query filters for database queries.

This module provides filtering capabilities for querying engineering drawing
records from the database with support for status, confidence scores, dates,
sorting, and pagination.
"""

from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, Set


class DrawingStatus(str, Enum):
    """Valid processing status values for engineering drawings."""

    COMPLETE = "complete"
    PROCESSING = "processing"
    FAILED = "failed"


class SortOrder(str, Enum):
    """Valid sort order values for query results."""

    ASC = "asc"
    DESC = "desc"


@dataclass
class QueryFilters:
    """Filters for querying engineering drawing records.

    This class encapsulates all filter criteria that can be applied when
    querying the drawing database. It includes validation logic to ensure
    filter values are within acceptable ranges and provides utility methods
    for pagination and filter manipulation.

    Attributes:
        status: Filter by processing status. None returns drawings with any status.
        needs_review: Filter by review flag. True returns only drawings
            requiring human review, False returns only drawings not requiring
            review, None returns all.
        min_confidence: Minimum overall confidence score (0.0-1.0). Only
            drawings with confidence >= this value are returned.
        max_confidence: Maximum overall confidence score (0.0-1.0). Only
            drawings with confidence <= this value are returned.
        has_part_number: Filter for drawings with part numbers. True returns
            only drawings with part numbers, False returns only drawings
            without, None returns all.
        date_from: Start date (inclusive) for filtering by processing timestamp.
            Only drawings processed on or after this date are returned.
        date_to: End date (inclusive) for filtering by processing timestamp.
            Only drawings processed on or before this date are returned.
        sort_by: Column name to sort by (e.g., 'processing_timestamp', 'confidence').
        sort_order: Sort direction (ascending or descending).
        limit: Maximum number of results to return. Defaults to 100. Maximum is 1000.
        offset: Number of results to skip for pagination. Defaults to 0.

    Raises:
        ValueError: If validation fails for any filter value.

    Examples:
        >>> # Filter for high-confidence drawings needing review
        >>> filters = QueryFilters(
        ...     needs_review=True,
        ...     min_confidence=0.8,
        ...     limit=50
        ... )

        >>> # Filter by date range with sorting
        >>> filters = QueryFilters(
        ...     date_from=datetime(2025, 1, 1),
        ...     date_to=datetime(2025, 1, 31),
        ...     status=DrawingStatus.COMPLETE,
        ...     sort_by='processing_timestamp',
        ...     sort_order=SortOrder.DESC
        ... )

        >>> # Pagination example
        >>> first_page = QueryFilters(limit=50)
        >>> second_page = first_page.next_page()
        >>> print(second_page.page_number)  # 2
    """

    status: Optional[DrawingStatus] = None
    needs_review: Optional[bool] = None
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    has_part_number: Optional[bool] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    sort_by: Optional[str] = None
    sort_order: Optional[SortOrder] = None
    limit: int = 100
    offset: int = 0

    # Constants
    MAX_LIMIT: int = 1000
    VALID_SORT_COLUMNS: Set[str] = {
        "processing_timestamp",
        "overall_confidence",
        "drawing_id",
        "source_file",
    }

    def __post_init__(self) -> None:
        """Validate filter values after initialization.

        Performs comprehensive validation of all filter parameters to ensure
        they are within acceptable ranges and logically consistent.

        Raises:
            ValueError: If any validation check fails with a descriptive
                message indicating which parameter is invalid and why.
        """
        self._validate_status()
        self._validate_confidence_scores()
        self._validate_date_range()
        self._validate_pagination()
        self._validate_sorting()

    def _validate_status(self) -> None:
        """Validate status field."""
        if self.status is not None:
            # Handle case-insensitive string input
            if isinstance(self.status, str):
                try:
                    self.status = DrawingStatus(self.status.lower())
                except ValueError:
                    valid_values = [s.value for s in DrawingStatus]
                    raise ValueError(
                        f"status must be one of {valid_values}, got '{self.status}'"
                    )

    def _validate_confidence_scores(self) -> None:
        """Validate confidence score bounds and consistency."""
        if self.min_confidence is not None:
            if not 0.0 <= self.min_confidence <= 1.0:
                raise ValueError(
                    f"min_confidence must be between 0.0 and 1.0, "
                    f"got {self.min_confidence}"
                )

        if self.max_confidence is not None:
            if not 0.0 <= self.max_confidence <= 1.0:
                raise ValueError(
                    f"max_confidence must be between 0.0 and 1.0, "
                    f"got {self.max_confidence}"
                )

        if (
            self.min_confidence is not None
            and self.max_confidence is not None
            and self.min_confidence > self.max_confidence
        ):
            raise ValueError(
                f"min_confidence ({self.min_confidence}) cannot be greater than "
                f"max_confidence ({self.max_confidence})"
            )

    def _validate_date_range(self) -> None:
        """Validate date range consistency and timezone handling."""
        if self.date_from is not None and self.date_to is not None:
            # Check timezone consistency
            if (self.date_from.tzinfo is None) != (self.date_to.tzinfo is None):
                raise ValueError(
                    "date_from and date_to must both be timezone-aware or both "
                    "be timezone-naive"
                )

            # Check logical ordering
            if self.date_from > self.date_to:
                raise ValueError(
                    f"date_from ({self.date_from}) cannot be after "
                    f"date_to ({self.date_to})"
                )

    def _validate_pagination(self) -> None:
        """Validate pagination parameters."""
        if self.limit < 0:
            raise ValueError(f"limit must be non-negative, got {self.limit}")

        if self.limit > self.MAX_LIMIT:
            raise ValueError(f"limit cannot exceed {self.MAX_LIMIT}, got {self.limit}")

        if self.offset < 0:
            raise ValueError(f"offset must be non-negative, got {self.offset}")

        if self.limit == 0 and self.offset > 0:
            raise ValueError("offset cannot be greater than 0 when limit is 0")

    def _validate_sorting(self) -> None:
        """Validate sorting parameters."""
        if self.sort_by is not None and self.sort_by not in self.VALID_SORT_COLUMNS:
            raise ValueError(
                f"sort_by must be one of {self.VALID_SORT_COLUMNS}, "
                f"got '{self.sort_by}'"
            )

        if self.sort_order is not None:
            # Handle case-insensitive string input
            if isinstance(self.sort_order, str):
                try:
                    self.sort_order = SortOrder(self.sort_order.lower())
                except ValueError:
                    valid_values = [s.value for s in SortOrder]
                    raise ValueError(
                        f"sort_order must be one of {valid_values}, "
                        f"got '{self.sort_order}'"
                    )

            # sort_order requires sort_by
            if self.sort_by is None:
                raise ValueError("sort_order cannot be specified without sort_by")

    def has_confidence_filter(self) -> bool:
        """Check if any confidence filtering is applied.

        Returns:
            True if min_confidence or max_confidence is set, False otherwise.
        """
        return self.min_confidence is not None or self.max_confidence is not None

    def has_date_filter(self) -> bool:
        """Check if any date filtering is applied.

        Returns:
            True if date_from or date_to is set, False otherwise.
        """
        return self.date_from is not None or self.date_to is not None

    def is_empty(self) -> bool:
        """Check if any non-pagination filters are active.

        Returns:
            True if no filters (excluding limit/offset) are set, False otherwise.
        """
        return not any(
            [
                self.status,
                self.needs_review is not None,
                self.has_confidence_filter(),
                self.has_part_number is not None,
                self.has_date_filter(),
                self.sort_by,
            ]
        )

    def to_query_params(self) -> Dict[str, Any]:
        """Export active filters as a dictionary.

        Returns only non-None filter values, useful for logging, serialization,
        or passing to query builders.

        Returns:
            Dictionary containing only the active filter parameters.

        Example:
            >>> filters = QueryFilters(min_confidence=0.8, limit=50)
            >>> filters.to_query_params()
            {'min_confidence': 0.8, 'limit': 50, 'offset': 0}
        """
        params = {}

        if self.status is not None:
            params["status"] = self.status.value
        if self.needs_review is not None:
            params["needs_review"] = self.needs_review
        if self.min_confidence is not None:
            params["min_confidence"] = self.min_confidence
        if self.max_confidence is not None:
            params["max_confidence"] = self.max_confidence
        if self.has_part_number is not None:
            params["has_part_number"] = self.has_part_number
        if self.date_from is not None:
            params["date_from"] = self.date_from.isoformat()
        if self.date_to is not None:
            params["date_to"] = self.date_to.isoformat()
        if self.sort_by is not None:
            params["sort_by"] = self.sort_by
        if self.sort_order is not None:
            params["sort_order"] = self.sort_order.value

        # Always include pagination
        params["limit"] = self.limit
        params["offset"] = self.offset

        return params

    @property
    def page_number(self) -> int:
        """Calculate the current 1-based page number.

        Returns:
            Current page number (1-indexed). Returns 1 if limit is 0.
        """
        if self.limit == 0:
            return 1
        return (self.offset // self.limit) + 1

    def next_page(self) -> "QueryFilters":
        """Create a new filter instance representing the next page.

        Returns:
            New QueryFilters instance with offset incremented by limit.
        """
        return self.copy_with(offset=self.offset + self.limit)

    def previous_page(self) -> "QueryFilters":
        """Create a new filter instance representing the previous page.

        Returns:
            New QueryFilters instance with offset decremented by limit.
            Offset will not go below 0.
        """
        new_offset = max(0, self.offset - self.limit)
        return self.copy_with(offset=new_offset)

    def reset_pagination(self) -> "QueryFilters":
        """Create a new filter instance with pagination reset to defaults.

        Returns:
            New QueryFilters instance with limit=100 and offset=0.
        """
        return self.copy_with(limit=100, offset=0)

    def copy_with(self, **kwargs: Any) -> "QueryFilters":
        """Create a modified copy of this filter instance.

        Uses dataclasses.replace to create a new instance with specified
        attributes changed while preserving all other values.

        Args:
            **kwargs: Attribute names and new values to apply.

        Returns:
            New QueryFilters instance with modifications applied.

        Example:
            >>> original = QueryFilters(limit=50)
            >>> modified = original.copy_with(limit=100, offset=50)
        """
        return replace(self, **kwargs)

    def merge(self, other: "QueryFilters") -> "QueryFilters":
        """Merge with another filter instance, giving precedence to other.

        Creates a new filter where non-None values from 'other' override
        values from self. Useful for applying default filters with overrides.

        Args:
            other: QueryFilters instance whose values take precedence.

        Returns:
            New QueryFilters instance with merged values.

        Example:
            >>> defaults = QueryFilters(limit=100, sort_by='processing_timestamp')
            >>> overrides = QueryFilters(limit=50)
            >>> merged = defaults.merge(overrides)
            >>> merged.limit  # 50
            >>> merged.sort_by  # 'processing_timestamp'
        """
        merged_params = {}

        for field in self.__dataclass_fields__:
            other_value = getattr(other, field)
            self_value = getattr(self, field)

            # Use other's value if it's not None, otherwise use self's value
            if other_value is not None:
                merged_params[field] = other_value
            else:
                merged_params[field] = self_value

        return QueryFilters(**merged_params)

    def __str__(self) -> str:
        """Return human-readable string representation of active filters.

        Returns:
            Concise string showing only non-default filter values.
        """
        parts = []

        if self.status:
            parts.append(f"status={self.status.value}")
        if self.needs_review is not None:
            parts.append(f"needs_review={self.needs_review}")
        if self.min_confidence is not None:
            parts.append(f"min_conf={self.min_confidence:.2f}")
        if self.max_confidence is not None:
            parts.append(f"max_conf={self.max_confidence:.2f}")
        if self.has_part_number is not None:
            parts.append(f"has_part_num={self.has_part_number}")
        if self.date_from:
            parts.append(f"from={self.date_from.date()}")
        if self.date_to:
            parts.append(f"to={self.date_to.date()}")
        if self.sort_by:
            order = self.sort_order.value if self.sort_order else "asc"
            parts.append(f"sort={self.sort_by}:{order}")

        parts.append(f"page={self.page_number}")
        parts.append(f"limit={self.limit}")

        return f"QueryFilters({', '.join(parts)})"

    def __repr__(self) -> str:
        """Return detailed string representation for debugging."""
        return (
            f"QueryFilters("
            f"status={self.status!r}, "
            f"needs_review={self.needs_review!r}, "
            f"min_confidence={self.min_confidence!r}, "
            f"max_confidence={self.max_confidence!r}, "
            f"has_part_number={self.has_part_number!r}, "
            f"date_from={self.date_from!r}, "
            f"date_to={self.date_to!r}, "
            f"sort_by={self.sort_by!r}, "
            f"sort_order={self.sort_order!r}, "
            f"limit={self.limit!r}, "
            f"offset={self.offset!r})"
        )
