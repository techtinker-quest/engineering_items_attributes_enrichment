# src/drawing_intelligence/llm/budget_controller.py
"""
Unified budget control with automatic model step-down
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from enum import Enum
import logging
from datetime import datetime, timedelta

from models.model_registry import ModelRegistry, ModelSpec, ModelTier

logger = logging.getLogger(__name__)


class UseCaseType(Enum):
    DRAWING_ASSESSMENT = "drawing_assessment"
    OCR_VERIFICATION = "ocr_verification"
    ENTITY_EXTRACTION = "entity_extraction"
    SHAPE_VALIDATION = "shape_validation"
    COMPLEX_REASONING = "complex_reasoning"


@dataclass
class UseCaseModelConfig:
    """Model configuration for a specific use case with tier fallbacks"""

    use_case: UseCaseType
    preferred_model: str  # Canonical name from ModelRegistry
    tier_1_fallback: str  # Mid-tier fallback
    tier_0_fallback: str  # Cheapest fallback
    max_tokens: int
    temperature: float = 0.0
    requires_vision: bool = False


class BudgetController:
    """
    Manages LLM budget with automatic model step-down.
    Integrates with ModelRegistry for unified model management.
    """

    # Define use-case configurations with fallback chains
    USE_CASE_CONFIGS: Dict[UseCaseType, UseCaseModelConfig] = {
        UseCaseType.DRAWING_ASSESSMENT: UseCaseModelConfig(
            use_case=UseCaseType.DRAWING_ASSESSMENT,
            preferred_model="claude-3-haiku",  # Already cheapest
            tier_1_fallback="claude-3-haiku",  # Same
            tier_0_fallback="claude-3-haiku",
            max_tokens=1000,
            requires_vision=True,
        ),
        UseCaseType.OCR_VERIFICATION: UseCaseModelConfig(
            use_case=UseCaseType.OCR_VERIFICATION,
            preferred_model="claude-3-sonnet",  # Tier 1
            tier_1_fallback="claude-3-sonnet",  # Same
            tier_0_fallback="claude-3-haiku",  # Step down
            max_tokens=2000,
            requires_vision=True,
        ),
        UseCaseType.ENTITY_EXTRACTION: UseCaseModelConfig(
            use_case=UseCaseType.ENTITY_EXTRACTION,
            preferred_model="gpt-4-turbo",  # Tier 1
            tier_1_fallback="gpt-4-turbo",  # Same
            tier_0_fallback="gpt-4o-mini",  # Step down
            max_tokens=4000,
            requires_vision=False,
        ),
        UseCaseType.SHAPE_VALIDATION: UseCaseModelConfig(
            use_case=UseCaseType.SHAPE_VALIDATION,
            preferred_model="gpt-4o",  # Tier 1
            tier_1_fallback="gpt-4o",  # Same
            tier_0_fallback="gpt-4o-mini",  # Step down
            max_tokens=1500,
            requires_vision=True,
        ),
        UseCaseType.COMPLEX_REASONING: UseCaseModelConfig(
            use_case=UseCaseType.COMPLEX_REASONING,
            preferred_model="claude-3-opus",  # Tier 2
            tier_1_fallback="claude-3.5-sonnet",  # Step down
            tier_0_fallback="claude-3-haiku",  # Step down further
            max_tokens=8000,
            requires_vision=True,
        ),
    }

    def __init__(
        self,
        daily_budget_usd: float,
        per_drawing_limit_usd: float,
        alert_threshold_pct: float = 0.80,
        db_manager=None,
    ):
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
        """
        Get the appropriate model for a use case, considering budget constraints.

        Returns:
            Tuple of (ModelSpec, reason)
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
                f"Stepped down to Tier 1 (budget at {self._get_budget_used_pct():.0f}%)"
            )
        else:  # TIER_0_CHEAP
            model_name = config.tier_0_fallback
            reason = (
                f"Stepped down to Tier 0 (budget at {self._get_budget_used_pct():.0f}%)"
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
        """
        Pre-flight check before making LLM call.

        Returns:
            Tuple of (allowed, reason, model_to_use)
            - allowed: Whether the call should proceed
            - reason: Explanation of the decision
            - model_to_use: The model to use (None if call not allowed)
        """
        # Get current spend
        current_spend = self._get_daily_spend()
        drawing_spend = self._get_drawing_spend(drawing_id)

        # Try current tier model
        model, tier_reason = self.get_model_for_use_case(use_case)
        estimated_cost = model.calculate_cost(
            estimated_input_tokens, estimated_output_tokens, image_count
        )

        # Check per-drawing limit
        if drawing_spend + estimated_cost > self.per_drawing_limit:
            return (
                False,
                f"Per-drawing limit (${self.per_drawing_limit}) would be exceeded",
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
                f"Daily budget (${self.daily_budget}) would be exceeded even with cheapest model",
                None,
            )

        # All checks passed
        return True, f"{tier_reason} | Estimated cost: ${estimated_cost:.4f}", model

    def track_call(
        self,
        provider: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        image_count: int,
        drawing_id: str,
        use_case: UseCaseType,
    ) -> float:
        """
        Track an LLM call and return actual cost.
        """
        # Get model spec to calculate accurate cost
        model = ModelRegistry.get_model(model_id)
        actual_cost = model.calculate_cost(input_tokens, output_tokens, image_count)

        # Store in database
        if self.db:
            self.db.store_llm_usage(
                drawing_id=drawing_id,
                use_case=use_case.value,
                provider=provider,
                model=model_id,
                tokens_input=input_tokens,
                tokens_output=output_tokens,
                image_count=image_count,
                cost_usd=actual_cost,
                timestamp=datetime.now(),
            )

        # Update drawing spend tracker
        self.current_drawing_spend += actual_cost

        logger.info(
            f"LLM call tracked: {use_case.value} | {model.canonical_name} | "
            f"${actual_cost:.4f} | Tokens: {input_tokens}/{output_tokens}"
        )

        return actual_cost

    def _step_down_tier(self):
        """Step down to next cheaper tier"""
        if self.current_tier == ModelTier.TIER_2_PREMIUM:
            self.current_tier = ModelTier.TIER_1_BALANCED
            logger.warning("Budget threshold reached: Stepping down to Tier 1 models")
        elif self.current_tier == ModelTier.TIER_1_BALANCED:
            self.current_tier = ModelTier.TIER_0_CHEAP
            logger.warning(
                "Budget threshold reached: Stepping down to Tier 0 (cheapest) models"
            )

        self.tier_stepped_down = True

    def _get_daily_spend(self) -> float:
        """Get today's total LLM spend"""
        if not self.db:
            return 0.0

        today = datetime.now().date()
        return self.db.get_llm_spend_for_date(today)

    def _get_drawing_spend(self, drawing_id: str) -> float:
        """Get total spend for a specific drawing"""
        if not self.db:
            return self.current_drawing_spend

        return self.db.get_llm_spend_for_drawing(drawing_id)

    def _get_budget_used_pct(self) -> float:
        """Get percentage of daily budget used"""
        return (self._get_daily_spend() / self.daily_budget) * 100

    def reset_drawing_tracker(self):
        """Reset per-drawing spend tracker (call when starting new drawing)"""
        self.current_drawing_spend = 0.0

    def get_usage_summary(self) -> Dict:
        """Get current usage summary"""
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
    from models.model_registry import ModelRegistry

    # Initialize budget controller
    controller = BudgetController(
        daily_budget_usd=50.00, per_drawing_limit_usd=0.30, alert_threshold_pct=0.80
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
