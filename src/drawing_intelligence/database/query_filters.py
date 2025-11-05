"""
Query filters for database queries.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class QueryFilters:
    """
    Filters for drawing queries.

    Attributes:
        status: Filter by status ('complete', 'processing', 'failed')
        needs_review: Filter by review flag
        min_confidence: Minimum confidence score (0.0-1.0)
        max_confidence: Maximum confidence score (0.0-1.0)
        has_part_number: Filter for drawings with part numbers
        date_from: Start date for processing timestamp
        date_to: End date for processing timestamp
        limit: Maximum results to return
        offset: Offset for pagination
    """

    status: Optional[str] = None
    needs_review: Optional[bool] = None
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    has_part_number: Optional[bool] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = 100
    offset: int = 0

    def __post_init__(self):
        """Validate filter values."""
        if self.min_confidence is not None:
            if not 0.0 <= self.min_confidence <= 1.0:
                raise ValueError("min_confidence must be between 0.0 and 1.0")

        if self.max_confidence is not None:
            if not 0.0 <= self.max_confidence <= 1.0:
                raise ValueError("max_confidence must be between 0.0 and 1.0")

        if self.limit < 0:
            raise ValueError("limit must be non-negative")

        if self.offset < 0:
            raise ValueError("offset must be non-negative")
