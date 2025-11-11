"""Test model registry"""

import pytest
from drawing_intelligence.models.model_registry import ModelRegistry, ModelTier


def test_get_model():
    model = ModelRegistry.get_model("claude-3-sonnet")
    assert model.canonical_name == "claude-3-sonnet"
    assert model.tier == ModelTier.TIER_1_BALANCED


def test_cost_calculation():
    model = ModelRegistry.get_model("gpt-4o-mini")
    cost = model.calculate_cost(1000, 500, image_count=1)
    assert cost > 0
    print(f"Cost for 1000 input, 500 output, 1 image: ${cost:.6f}")
