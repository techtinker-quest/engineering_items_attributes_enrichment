"""
Anthropic Provider

Implementation of Anthropic (Claude) API integration.
"""

import os
import logging
from typing import Optional, Dict, Any, List
import anthropic

from .base_provider import LLMProvider, LLMResponse
from ...models.data_structures import TokenUsage


logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Provider for Anthropic (Claude) API."""

    SUPPORTED_MODELS = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307",
    ]

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (or from ANTHROPIC_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required")

        # Lazy initialization
        self._client = None
        logger.info("AnthropicProvider initialized")

    @property
    def client(self):
        """Lazy load Anthropic client."""
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def call(
        self,
        prompt: str,
        image: Optional[bytes] = None,
        model: str = "claude-3-sonnet-20240229",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        response_format: Optional[Dict] = None,
    ) -> LLMResponse:
        """
        Call Anthropic API.

        Args:
            prompt: Text prompt
            image: Optional image bytes
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            response_format: Response format (not used by Anthropic)

        Returns:
            LLMResponse object
        """
        try:
            # Build messages
            messages = []

            if image:
                # Image + text message
                import base64

                image_base64 = base64.b64encode(image).decode("utf-8")

                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                )
            else:
                # Text-only message
                messages.append({"role": "user", "content": prompt})

            # Call API
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )

            # Extract content
            content = response.content[0].text

            # Track token usage
            tokens_used = TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                image_count=1 if image else 0,
            )

            return LLMResponse(
                content=content,
                tokens_used=tokens_used,
                model_used=model,
                provider="anthropic",
                success=True,
                error_message=None,
            )

        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            return LLMResponse(
                content="",
                tokens_used=TokenUsage(input_tokens=0, output_tokens=0, image_count=0),
                model_used=model,
                provider="anthropic",
                success=False,
                error_message=str(e),
            )

    def get_available_models(self) -> List[str]:
        """Return list of supported models."""
        return self.SUPPORTED_MODELS.copy()

    def validate_credentials(self) -> bool:
        """Validate API key by making a test call."""
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except Exception as e:
            logger.error(f"Credential validation failed: {e}")
            return False
