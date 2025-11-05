"""
Google AI provider implementation.
"""

import logging
from typing import Optional, Dict, Any
from PIL import Image
import io

from .base_provider import LLMProvider, LLMResponse, TokenUsage

logger = logging.getLogger(__name__)


class GoogleProvider(LLMProvider):
    """
    Google AI (Gemini) API provider implementation.

    Supports Gemini Pro and Gemini Pro Vision models.
    """

    def __init__(self, api_key: str):
        """
        Initialize Google provider.

        Args:
            api_key: Google AI API key
        """
        self.api_key = api_key
        self._genai = None

        logger.info("Google provider initialized")

    def _get_genai(self):
        """Lazy initialization of Google GenAI."""
        if self._genai is None:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self._genai = genai
            except ImportError:
                raise ImportError(
                    "Google GenerativeAI package not installed. "
                    "Install with: pip install google-generativeai"
                )

        return self._genai

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
        Call Google AI API.

        Args:
            prompt: Text prompt
            image: Optional image bytes
            model: Model identifier (e.g., 'gemini-1.5-pro')
            max_tokens: Maximum response tokens
            temperature: Temperature (0.0-1.0)
            response_format: Not used for Google

        Returns:
            LLMResponse
        """
        genai = self._get_genai()

        try:
            # Initialize model
            model_instance = genai.GenerativeModel(model)

            # Build content
            if image:
                # Convert bytes to PIL Image
                pil_image = Image.open(io.BytesIO(image))
                content = [prompt, pil_image]
            else:
                content = prompt

            # Configure generation
            generation_config = {
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }

            # Make API call
            response = model_instance.generate_content(
                content, generation_config=generation_config
            )

            # Extract content
            content_text = response.text

            # Google doesn't provide detailed token usage in response
            # Estimate based on prompt and response length
            input_tokens = len(prompt.split()) * 4 // 3  # Rough estimate
            output_tokens = len(content_text.split()) * 4 // 3

            tokens_used = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                image_count=1 if image else 0,
            )

            return LLMResponse(
                content=content_text,
                tokens_used=tokens_used,
                model_used=model,
                provider="google",
                success=True,
            )

        except Exception as e:
            logger.error(f"Google AI API call failed: {e}")
            return LLMResponse(
                content="",
                tokens_used=TokenUsage(0, 0, 0),
                model_used=model,
                provider="google",
                success=False,
                error_message=str(e),
            )

    def get_available_models(self) -> list:
        """Get list of available Google models."""
        return ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro", "gemini-pro-vision"]

    def validate_credentials(self) -> bool:
        """
        Validate Google AI API credentials.

        Returns:
            True if credentials are valid
        """
        try:
            genai = self._get_genai()
            # Test with minimal request
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content("test")
            return True
        except Exception as e:
            logger.error(f"Google credential validation failed: {e}")
            return False
