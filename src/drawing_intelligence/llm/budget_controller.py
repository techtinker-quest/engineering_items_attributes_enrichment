# src/drawing_intelligence/llm/budget_controller.py
"""Budget controller for LLM API cost management.

This module provides unified budget control with automatic model tier step-down
functionality. It integrates with the ModelRegistry for consistent model
management and tracks both daily and per-drawing spending limits.

Example:
    Basic usage with budget limits::

        controller = BudgetController(
            daily_budget_usd=50.00,
            per_drawing_limit_usd=0.30
        )

        allowed, reason, model = controller.pre_call_check(
            estimated_input_tokens=2500,
            estimated_output_tokens=800,
            image_count=1,
            use_case=UseCaseType.ENTITY_EXTRACTION,
            drawing_id="DWG-001"
        )
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Dict, Optional, Tuple, Union

from ..models.model_registry import ModelRegistry, ModelSpec, ModelTier

if TYPE_CHECKING:
    from ..database.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class UseCaseType(Enum):
    """Enumeration of LLM use case types.

    Attributes:
        DRAWING_ASSESSMENT: Overall drawing quality and complexity analysis.
        OCR_VERIFICATION: Text recognition verification and correction.
        ENTITY_EXTRACTION: Structured data extraction from drawings.
        SHAPE_VALIDATION: Geometric shape detection validation.
        COMPLEX_REASONING: Advanced multi-step reasoning tasks.
    """

    DRAWING_ASSESSMENT = "drawing_assessment"
    OCR_VERIFICATION = "ocr_verification"
    ENTITY_EXTRACTION = "entity_extraction"
    SHAPE_VALIDATION = "shape_validation"
    COMPLEX_REASONING = "complex_reasoning"


@dataclass
class UseCaseModelConfig:
    """Model configuration for a specific use case with tier fallbacks.

    Attributes:
        use_case: The use case type this configuration applies to.
        preferred_model: Canonical model name from ModelRegistry (highest
            tier).
        tier_1_fallback: Mid-tier fallback model canonical name.
        tier_0_fallback: Cheapest fallback model canonical name.
        max_tokens: Maximum tokens for model output.
        temperature: Sampling temperature (0.0 = deterministic).
        requires_vision: Whether the use case requires vision capabilities.
    """

    use_case: UseCaseType
    preferred_model: str
    tier_1_fallback: str
    tier_0_fallback: str
    max_tokens: int
    temperature: float = 0.0
    requires_vision: bool = False


class BudgetController:
    """Manages LLM budget with automatic model step-down.

    This class tracks daily and per-drawing LLM spending, automatically
    steps down to cheaper models when budget thresholds are reached, and
    prevents calls that would exceed configured limits.

    Attributes:
        daily_budget: Maximum daily spending limit in USD.
        per_drawing_limit: Maximum spending per drawing in USD.
        alert_threshold: Spending level that triggers tier step-down.
        db: Optional database manager for persistent tracking.
        current_tier: Current active model tier (may be stepped down).
        tier_stepped_down: Whether automatic step-down has occurred.
        current_drawing_spend: Accumulated spend for current drawing.

    Example:
        Initialize and check budget before LLM call::

            controller = BudgetController(
                daily_budget_usd=50.00,
                per_drawing_limit_usd=0.30,
                alert_threshold_pct=0.80
            )

            allowed, reason, model = controller.pre_call_check(
                estimated_input_tokens=2500,
                estimated_output_tokens=800,
                image_count=1,
                use_case=UseCaseType.ENTITY_EXTRACTION,
                drawing_id="DWG-001"
            )

            if allowed:
                # Make LLM call with model
                pass
    """

    USE_CASE_CONFIGS: ClassVar[Dict[UseCaseType, UseCaseModelConfig]] = {
        UseCaseType.DRAWING_ASSESSMENT: UseCaseModelConfig(
            use_case=UseCaseType.DRAWING_ASSESSMENT,
            preferred_model="claude-3-haiku",
            tier_1_fallback="claude-3-haiku",
            tier_0_fallback="claude-3-haiku",
            max_tokens=1000,
            requires_vision=True,
        ),
        UseCaseType.OCR_VERIFICATION: UseCaseModelConfig(
            use_case=UseCaseType.OCR_VERIFICATION,
            preferred_model="claude-3-sonnet",
            tier_1_fallback="claude-3-sonnet",
            tier_0_fallback="claude-3-haiku",
            max_tokens=2000,
            requires_vision=True,
        ),
        UseCaseType.ENTITY_EXTRACTION: UseCaseModelConfig(
            use_case=UseCaseType.ENTITY_EXTRACTION,
            preferred_model="gpt-4-turbo",
            tier_1_fallback="gpt-4-turbo",
            tier_0_fallback="gpt-4o-mini",
            max_tokens=4000,
            requires_vision=False,
        ),
        UseCaseType.SHAPE_VALIDATION: UseCaseModelConfig(
            use_case=UseCaseType.SHAPE_VALIDATION,
            preferred_model="gpt-4o",
            tier_1_fallback="gpt-4o",
            tier_0_fallback="gpt-4o-mini",
            max_tokens=1500,
            requires_vision=True,
        ),
        UseCaseType.COMPLEX_REASONING: UseCaseModelConfig(
            use_case=UseCaseType.COMPLEX_REASONING,
            preferred_model="claude-3-opus",
            tier_1_fallback="claude-3.5-sonnet",
            tier_0_fallback="claude-3-haiku",
            max_tokens=8000,
            requires_vision=True,
        ),
    }

    def __init__(
        self,
        daily_budget_usd: float,
        per_drawing_limit_usd: float,
        alert_threshold_pct: float = 0.80,
        db_manager: Optional["DatabaseManager"] = None,
    ) -> None:
        """Initialize budget controller with spending limits.

        Args:
            daily_budget_usd: Maximum daily spending in USD.
            per_drawing_limit_usd: Maximum spending per drawing in USD.
            alert_threshold_pct: Percentage of daily budget that triggers
                model tier step-down (default: 0.80).
            db_manager: Optional database manager for persistent tracking.
                If None, tracking is in-memory only.

        Raises:
            ValueError: If budget values are negative or alert_threshold_pct
                is not between 0 and 1.
        """
        if daily_budget_usd < 0:
            raise ValueError("daily_budget_usd must be non-negative")
        if per_drawing_limit_usd < 0:
            raise ValueError("per_drawing_limit_usd must be non-negative")
        if not 0 <= alert_threshold_pct <= 1:
            raise ValueError("alert_threshold_pct must be between 0 and 1")

        self.daily_budget = daily_budget_usd
        self.per_drawing_limit = per_drawing_limit_usd
        self.alert_threshold = daily_budget_usd * alert_threshold_pct
        self.db = db_manager

        # Track current tier level (starts at preferred)
        self.current_tier = ModelTier.TIER_2_PREMIUM
        self.tier_stepped_down = False

        # Track per-drawing spend
        self.current_drawing_spend = 0.0

    def get_model_for_use_case(
        self, use_case: UseCaseType, override_model: Optional[str] = None
    ) -> Tuple[ModelSpec, str]:
        """Get the appropriate model for a use case considering budget.

        Selects a model based on the current tier level (which may have been
        stepped down due to budget constraints) and the use case requirements.

        Args:
            use_case: The type of LLM task being performed.
            override_model: Optional model canonical name to force use of
                a specific model (bypasses tier selection).

        Returns:
            Tuple containing:
                - ModelSpec: The selected model specification.
                - str: Reason for model selection.

        Raises:
            ValueError: If the selected model doesn't support required
                capabilities (e.g., vision when needed).
            KeyError: If override_model is not found in ModelRegistry.
        """
        # Allow override for testing/specific requests
        if override_model:
            model = ModelRegistry.get_model(override_model)
            return model, f"Override requested: {override_model}"

        config = self.USE_CASE_CONFIGS[use_case]

        # Select model based on current tier
        if self.current_tier == ModelTier.TIER_2_PREMIUM:
            model_name = config.preferred_model
            reason = "Using preferred model"
        elif self.current_tier == ModelTier.TIER_1_BALANCED:
            model_name = config.tier_1_fallback
            reason = (
                f"Stepped down to Tier 1 (budget at "
                f"{self._get_budget_used_pct():.0f}%)"
            )
        else:  # TIER_0_CHEAP
            model_name = config.tier_0_fallback
            reason = (
                f"Stepped down to Tier 0 (budget at "
                f"{self._get_budget_used_pct():.0f}%)"
            )

        model = ModelRegistry.get_model(model_name)

        # Validate vision requirement
        if config.requires_vision and not model.supports_vision:
            logger.error(f"Model {model_name} doesn't support vision for {use_case}")
            raise ValueError(f"Model must support vision for {use_case.value}")

        return model, reason

    def pre_call_check(
        self,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
        image_count: int,
        use_case: UseCaseType,
        drawing_id: str,
    ) -> Tuple[bool, str, Optional[ModelSpec]]:
        """Pre-flight check before making LLM call.

        Validates that the proposed LLM call would not exceed budget limits,
        automatically steps down to cheaper models if near threshold, and
        returns the appropriate model to use.

        Args:
            estimated_input_tokens: Expected number of input tokens.
            estimated_output_tokens: Expected number of output tokens.
            image_count: Number of images to be processed.
            use_case: Type of LLM task being performed.
            drawing_id: Unique identifier for the drawing being processed.

        Returns:
            Tuple containing:
                - bool: Whether the call should proceed.
                - str: Explanation of the decision.
                - Optional[ModelSpec]: The model to use (None if call denied).

        Note:
            This method may modify `self.current_tier` if budget threshold
            is reached, affecting subsequent calls.
        """
        # Get current spend (cached for this check)
        try:
            current_spend = self._get_daily_spend()
            drawing_spend = self._get_drawing_spend(drawing_id)
        except Exception as e:
            logger.error(f"Failed to retrieve spending data: {e}")
            # Fail safe - deny call if we can't check budget
            return (
                False,
                f"Unable to verify budget: {str(e)}",
                None,
            )

        # Try current tier model
        model, tier_reason = self.get_model_for_use_case(use_case)
        estimated_cost = model.calculate_cost(
            estimated_input_tokens, estimated_output_tokens, image_count
        )

        # Check per-drawing limit
        if drawing_spend + estimated_cost > self.per_drawing_limit:
            return (
                False,
                (
                    f"Per-drawing limit (${self.per_drawing_limit}) would be "
                    f"exceeded"
                ),
                None,
            )

        # Check if we should step down
        if (
            current_spend >= self.alert_threshold
            and self.current_tier > ModelTier.TIER_0_CHEAP
        ):
            # Try stepping down
            self._step_down_tier()
            model, tier_reason = self.get_model_for_use_case(use_case)
            estimated_cost = model.calculate_cost(
                estimated_input_tokens, estimated_output_tokens, image_count
            )

        # Final budget check with (possibly stepped-down) model
        if current_spend + estimated_cost > self.daily_budget:
            return (
                False,
                (
                    f"Daily budget (${self.daily_budget}) would be exceeded "
                    f"even with cheapest model"
                ),
                None,
            )

        # All checks passed
        return (
            True,
            f"{tier_reason} | Estimated cost: ${estimated_cost:.4f}",
            model,
        )

    def track_call(
        self,
        provider: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        image_count: int,
        drawing_id: str,
        use_case: UseCaseType,
        timestamp: Optional[datetime] = None,
    ) -> float:
        """Track an LLM call and return actual cost.

        Records the LLM call in the database (if available) and updates
        per-drawing spending tracker. Should be called after successful
        LLM API call with actual token counts.

        Args:
            provider: Name of the LLM provider (e.g., "anthropic", "openai").
            model_id: Specific model version identifier.
            input_tokens: Actual number of input tokens used.
            output_tokens: Actual number of output tokens generated.
            image_count: Number of images processed.
            drawing_id: Unique identifier for the drawing.
            use_case: Type of LLM task that was performed.
            timestamp: Optional timestamp for the call. If None, uses current
                time.

        Returns:
            Actual cost in USD for the call.

        Raises:
            KeyError: If model_id is not found in ModelRegistry.
        """
        # Get model spec to calculate accurate cost
        model = ModelRegistry.get_model(model_id)
        actual_cost = model.calculate_cost(input_tokens, output_tokens, image_count)

        # Use provided timestamp or current time
        if timestamp is None:
            timestamp = datetime.now()

        # Store in database
        if self.db:
            try:
                self.db.store_llm_usage(
                    drawing_id=drawing_id,
                    use_case=use_case.value,
                    provider=provider,
                    model=model_id,
                    tokens_input=input_tokens,
                    tokens_output=output_tokens,
                    image_count=image_count,
                    cost_usd=actual_cost,
                    timestamp=timestamp,
                )
            except Exception as e:
                logger.error(f"Failed to store LLM usage in database: {e}")
                # Continue - don't fail the tracking due to DB issues

        # Update drawing spend tracker
        self.current_drawing_spend += actual_cost

        logger.info(
            f"LLM call tracked: {use_case.value} | {model.canonical_name} | "
            f"${actual_cost:.4f} | Tokens: {input_tokens}/{output_tokens}"
        )

        return actual_cost

    def _step_down_tier(self) -> None:
        """Step down to next cheaper tier.

        Reduces the current tier level and logs the change. Called
        automatically by pre_call_check when budget threshold is reached.

        Note:
            Has no effect if already at TIER_0_CHEAP (lowest tier).
        """
        if self.current_tier == ModelTier.TIER_2_PREMIUM:
            self.current_tier = ModelTier.TIER_1_BALANCED
            logger.warning("Budget threshold reached: Stepping down to Tier 1 models")
            self.tier_stepped_down = True
        elif self.current_tier == ModelTier.TIER_1_BALANCED:
            self.current_tier = ModelTier.TIER_0_CHEAP
            logger.warning(
                "Budget threshold reached: Stepping down to Tier 0 " "(cheapest) models"
            )
            self.tier_stepped_down = True

    def _get_daily_spend(self) -> float:
        """Get today's total LLM spend.

        Returns:
            Total spending in USD for the current date. Returns 0.0 if
            database manager is not configured.
        """
        if not self.db:
            return 0.0

        today = datetime.now().date()
        try:
            return self.db.get_llm_spend_for_date(today)
        except AttributeError:
            logger.warning(
                "DatabaseManager does not implement get_llm_spend_for_date()"
            )
            return 0.0

    def _get_drawing_spend(self, drawing_id: str) -> float:
        """Get total spend for a specific drawing.

        Args:
            drawing_id: Unique identifier for the drawing.

        Returns:
            Total spending in USD for the specified drawing. Returns
            in-memory tracker value if database is not configured.
        """
        if not self.db:
            return self.current_drawing_spend

        try:
            return self.db.get_llm_spend_for_drawing(drawing_id)
        except AttributeError:
            logger.warning(
                "DatabaseManager does not implement " "get_llm_spend_for_drawing()"
            )
            return self.current_drawing_spend

    def _get_budget_used_pct(self) -> float:
        """Get percentage of daily budget used.

        Returns:
            Percentage (0-100) of daily budget that has been spent.
        """
        if self.daily_budget == 0:
            return 0.0
        return (self._get_daily_spend() / self.daily_budget) * 100

    def reset_drawing_tracker(self) -> None:
        """Reset per-drawing spend tracker.

        Should be called when starting processing of a new drawing to
        ensure per-drawing limits are correctly enforced.

        Note:
            This only resets in-memory tracking. Historical database
            records are not affected.
        """
        self.current_drawing_spend = 0.0

    def get_usage_summary(self) -> Dict[str, Union[float, str, bool]]:
        """Get current usage summary.

        Returns:
            Dictionary containing:
                - daily_budget (float): Configured daily limit.
                - daily_spend (float): Current daily spending.
                - remaining_budget (float): Budget remaining today.
                - budget_used_pct (float): Percentage of budget used.
                - current_tier (str): Active model tier name.
                - tier_stepped_down (bool): Whether step-down occurred.
                - alert_threshold (float): Threshold that triggers step-down.
                - alert_triggered (bool): Whether threshold was reached.
        """
        daily_spend = self._get_daily_spend()
        return {
            "daily_budget": self.daily_budget,
            "daily_spend": daily_spend,
            "remaining_budget": self.daily_budget - daily_spend,
            "budget_used_pct": self._get_budget_used_pct(),
            "current_tier": self.current_tier.name,
            "tier_stepped_down": self.tier_stepped_down,
            "alert_threshold": self.alert_threshold,
            "alert_triggered": daily_spend >= self.alert_threshold,
        }


# Example usage:
if __name__ == "__main__":
    # Initialize budget controller
    controller = BudgetController(
        daily_budget_usd=50.00,
        per_drawing_limit_usd=0.30,
        alert_threshold_pct=0.80,
    )

    # Check if we can make a call
    allowed, reason, model = controller.pre_call_check(
        estimated_input_tokens=2500,
        estimated_output_tokens=800,
        image_count=1,
        use_case=UseCaseType.ENTITY_EXTRACTION,
        drawing_id="DWG-001",
    )

    print(f"Call allowed: {allowed}")
    print(f"Reason: {reason}")
    if model:
        print(f"Model to use: {model.canonical_name} ({model.model_id})")

    # Get usage summary
    summary = controller.get_usage_summary()
    print(f"\nBudget summary: {summary}")
