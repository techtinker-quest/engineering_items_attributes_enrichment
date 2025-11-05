"""
OpenAI provider implementation.
"""

import logging
import base64
from typing import Optional, Dict, Any

from .base_provider import LLMProvider, LLMResponse, TokenUsage

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider implementation.

    Supports GPT-4, GPT-4 Turbo, GPT-4 Vision models.
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            base_url: Optional custom base URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

        logger.info("OpenAI provider initialized")

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                if self.base_url:
                    self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                else:
                    self._client = OpenAI(api_key=self.api_key)

            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai"
                )

        return self._client

    def call(
        self,
        prompt: str,
        image: Optional[bytes],
        model: str,
        max_tokens: int,
        temperature: float,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        Call OpenAI API.

        Args:
            prompt: Text prompt
            image: Optional image bytes
            model: Model identifier (e.g., 'gpt-4-turbo')
            max_tokens: Maximum response tokens
            temperature: Temperature (0.0-1.0)
            response_format: Optional format (e.g., {"type": "json_object"})

        Returns:
            LLMResponse
        """
        client = self._get_client()

        try:
            # Build messages
            if image:
                # Vision model required
                image_b64 = self._encode_image(image)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ]
            else:
                messages = [{"role": "user", "content": prompt}]

            # Make API call
            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            if response_format:
                kwargs["response_format"] = response_format

            response = client.chat.completions.create(**kwargs)

            # Extract content and usage
            content = response.choices[0].message.content

            tokens_used = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                image_count=1 if image else 0,
            )

            return LLMResponse(
                content=content,
                tokens_used=tokens_used,
                model_used=model,
                provider="openai",
                success=True,
            )

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return LLMResponse(
                content="",
                tokens_used=TokenUsage(0, 0, 0),
                model_used=model,
                provider="openai",
                success=False,
                error_message=str(e),
            )

    def _encode_image(self, image_bytes: bytes) -> str:
        """
        Base64 encode image for API.

        Args:
            image_bytes: Image bytes

        Returns:
            Base64-encoded string
        """
        return base64.b64encode(image_bytes).decode("utf-8")

    def get_available_models(self) -> list:
        """Get list of available OpenAI models."""
        return [
            "gpt-4-turbo-2024-04-09",
            "gpt-4-vision-preview",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4",
            "gpt-3.5-turbo",
        ]

    def validate_credentials(self) -> bool:
        """
        Validate OpenAI API credentials.

        Returns:
            True if credentials are valid
        """
        try:
            client = self._get_client()
            # Test with minimal request
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )
            return True
        except Exception as e:
            logger.error(f"OpenAI credential validation failed: {e}")
            return False
