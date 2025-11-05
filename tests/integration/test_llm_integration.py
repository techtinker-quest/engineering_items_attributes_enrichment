"""
Integration tests for LLM integration.
"""

import pytest
import os

from src.drawing_intelligence.llm import LLMGateway, LLMConfig
from src.drawing_intelligence.llm.budget_controller import BudgetController
from tests.utils.test_helpers import skip_if_no_llm, setup_test_database


@pytest.mark.integration
@pytest.mark.llm
class TestLLMIntegration:
    """Test LLM integration."""

    def setup_method(self):
        """Setup test environment."""
        skip_if_no_llm()
        self.db = setup_test_database()

    def teardown_method(self):
        """Cleanup."""
        if hasattr(self, "db"):
            self.db.close()

    def test_openai_provider(self):
        """Test OpenAI provider integration."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not configured")

        from src.drawing_intelligence.llm.providers import OpenAIProvider

        provider = OpenAIProvider()

        # Test simple call
        response = provider.call(
            prompt="Say hello", model="gpt-3.5-turbo", max_tokens=10
        )

        assert response.success is True
        assert len(response.content) > 0

    def test_anthropic_provider(self):
        """Test Anthropic provider integration."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("Anthropic API key not configured")

        from src.drawing_intelligence.llm.providers import AnthropicProvider

        provider = AnthropicProvider()

        # Test simple call
        response = provider.call(
            prompt="Say hello", model="claude-3-haiku-20240307", max_tokens=10
        )

        assert response.success is True
        assert len(response.content) > 0

    def test_budget_controller(self):
        """Test budget controller."""
        budget_controller = BudgetController(
            daily_budget_usd=1.0, per_drawing_limit_usd=0.10, db_manager=self.db
        )

        # Should be able to afford small request
        can_afford = budget_controller.can_afford(0.01, "test-drawing")

        assert can_afford is True
