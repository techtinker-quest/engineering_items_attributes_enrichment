"""Anthropic (Claude) API Provider Implementation.

This module provides integration with Anthropic's Claude API, supporting both
text-only and vision-enabled prompts. Implements the LLMProvider interface
with automatic token usage tracking and credential validation.

The provider handles:
- Lazy client initialization with thread safety
- Image encoding and format validation for vision requests
- Token usage extraction and cost calculation
- Comprehensive error handling with retry logic
- Rate limiting with exponential backoff

Note:
    Anthropic's API does not support structured output formats natively.
    Use prompt engineering for structured outputs.
"""

import base64
import logging
import os
import time
from functools import cached_property
from typing import Any, Dict, List, Optional

import anthropic

from ...models.data_structures import TokenUsage
from ...models.model_registry import ModelRegistry
from .base_provider import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMImageError,
    LLMModelError,
    LLMProvider,
    LLMRateLimitError,
    LLMResponse,
)

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Concrete implementation of LLMProvider for Anthropic's Claude models.

    Supports Claude 3 model family including Opus, Sonnet, and Haiku variants.
    Implements lazy client initialization with thread safety to avoid connection
    overhead until first API call.

    Attributes:
        api_key: The API key for Anthropic authentication.
        model_registry: Reference to ModelRegistry for pricing/capabilities.
        max_retries: Maximum number of retry attempts for rate limits.
        PROVIDER_NAME: Constant identifier for this provider.

    Example:
        >>> provider = AnthropicProvider(
        ...     api_key="sk-ant-...",
        ...     model_registry=registry
        ... )
        >>> response = provider.call(
        ...     prompt="Extract part numbers from this drawing",
        ...     image=drawing_bytes,
        ...     model="claude-3-sonnet-20240229"
        ... )
        >>> print(response.content)
        >>> cost = provider.calculate_cost(response.tokens_used, model)
    """

    PROVIDER_NAME = "anthropic"

    # Image constraints per Anthropic API documentation
    MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
    MAX_IMAGE_DIMENSION = 4096  # pixels

    # Image format detection via magic numbers
    IMAGE_SIGNATURES = {
        b"\x89PNG": "image/png",
        b"\xff\xd8\xff": "image/jpeg",
        b"RIFF": "image/webp",
        b"GIF87a": "image/gif",
        b"GIF89a": "image/gif",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_registry: Optional[ModelRegistry] = None,
        max_retries: int = 3,
        suppress_format_warning: bool = False,
    ) -> None:
        """Initialize the Anthropic provider with API credentials.

        Args:
            api_key: Anthropic API key. If not provided, attempts to load
                from ANTHROPIC_API_KEY environment variable.
            model_registry: ModelRegistry instance for pricing/capabilities.
                If None, a default instance is created.
            max_retries: Maximum retry attempts for rate limit errors.
            suppress_format_warning: If True, suppress warnings about
                response_format being ignored.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Provide via constructor or "
                "ANTHROPIC_API_KEY environment variable."
            )

        self.model_registry = model_registry or ModelRegistry()
        self.max_retries = max_retries
        self.suppress_format_warning = suppress_format_warning

        logger.info("AnthropicProvider initialized with API key from environment")

    @cached_property
    def client(self) -> anthropic.Anthropic:
        """Thread-safe lazy-loaded Anthropic API client.

        Uses functools.cached_property for thread-safe initialization.
        Creates the client on first access to avoid initialization overhead.

        Returns:
            Initialized Anthropic client instance.
        """
        return anthropic.Anthropic(api_key=self.api_key)

    def _detect_image_format(self, image_bytes: bytes) -> str:
        """Detect image MIME type from byte signature.

        Args:
            image_bytes: Raw image data.

        Returns:
            MIME type string (e.g., 'image/png').

        Raises:
            LLMImageError: If image format cannot be determined or is invalid.
        """
        if not image_bytes:
            raise LLMImageError("Empty image data provided")

        for signature, mime_type in self.IMAGE_SIGNATURES.items():
            if image_bytes.startswith(signature):
                # Special handling for WebP (needs WEBP at offset 8)
                if mime_type == "image/webp":
                    if len(image_bytes) > 12 and image_bytes[8:12] == b"WEBP":
                        return mime_type
                    continue

                # Warning for GIF (only first frame processed)
                if mime_type == "image/gif":
                    logger.warning(
                        "GIF image detected. Anthropic API only processes "
                        "the first frame."
                    )

                return mime_type

        raise LLMImageError(
            "Unsupported or unrecognizable image format. "
            "Supported: PNG, JPEG, WebP, GIF"
        )

    def _validate_image_constraints(self, image_bytes: bytes) -> None:
        """Validate image size constraints before API call.

        Args:
            image_bytes: Raw image data.

        Raises:
            LLMImageError: If image exceeds size limits.
        """
        image_size = len(image_bytes)
        if image_size > self.MAX_IMAGE_SIZE_BYTES:
            raise LLMImageError(
                f"Image size ({image_size / 1024 / 1024:.2f} MB) exceeds "
                f"maximum allowed size of "
                f"{self.MAX_IMAGE_SIZE_BYTES / 1024 / 1024} MB"
            )

    def _validate_model(self, model: str, requires_vision: bool = False) -> None:
        """Validate model against model registry.

        Args:
            model: Model identifier string.
            requires_vision: If True, ensure model supports vision.

        Raises:
            LLMModelError: If model is not supported or doesn't meet
                requirements.
        """
        try:
            model_spec = self.model_registry.get_model(model)
        except KeyError:
            raise LLMModelError(
                f"Model '{model}' not found in registry. "
                f"Check model_registry for available models."
            )

        if requires_vision and not model_spec.supports_vision:
            raise LLMModelError(
                f"Model '{model}' does not support vision. "
                f"Choose a vision-enabled model for image inputs."
            )

    def _validate_parameters(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> None:
        """Validate call parameters.

        Args:
            prompt: User prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Raises:
            ValueError: If parameters are invalid.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty or whitespace-only")

        if not (1 <= max_tokens <= 4096):
            raise ValueError(f"max_tokens must be between 1 and 4096, got {max_tokens}")

        if not (0.0 <= temperature <= 1.0):
            raise ValueError(
                f"temperature must be between 0.0 and 1.0, got {temperature}"
            )

    def _create_error_response(
        self,
        model: str,
        error_message: str,
    ) -> LLMResponse:
        """Create standardized error response.

        Args:
            model: Model identifier that failed.
            error_message: Error description.

        Returns:
            LLMResponse with success=False and zero tokens.
        """
        return LLMResponse(
            content="",
            tokens_used=TokenUsage(input_tokens=0, output_tokens=0, image_count=0),
            model_used=model,
            provider=self.PROVIDER_NAME,
            success=False,
            error_message=error_message,
        )

    def call(
        self,
        prompt: str,
        image: Optional[bytes] = None,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Execute a synchronous call to the Anthropic API with retry logic.

        Supports both text-only and vision-enabled requests. Images are
        automatically validated, format-detected, and base64-encoded.
        Implements exponential backoff for rate limit errors.

        Args:
            prompt: The user prompt/instruction text (non-empty).
            image: Optional image data as raw bytes. Supported formats:
                PNG, JPEG, WebP, GIF. Must be ≤5MB and ≤4096px per dimension.
            model: Claude model identifier from ModelRegistry.
            max_tokens: Maximum tokens in completion (1-4096).
            temperature: Sampling temperature (0.0-1.0). Lower = deterministic.
            response_format: Ignored with warning unless suppressed.

        Returns:
            LLMResponse containing:
                - content: Generated text response
                - tokens_used: Input/output tokens and image count
                - model_used: Actual model that processed the request
                - provider: Always "anthropic"
                - success: True if call succeeded
                - error_message: None on success, details on failure

        Raises:
            LLMAuthenticationError: Invalid API credentials.
            LLMRateLimitError: Rate limit exceeded after retries.
            LLMModelError: Invalid or unsupported model.
            LLMImageError: Invalid image format or size.
            LLMAPIError: Other API errors.
            ValueError: Invalid parameters.

        Note:
            For vision requests, Anthropic charges ~1,150 tokens per image.
            This is included in the API's input_tokens count.
        """
        try:
            # Validate parameters
            self._validate_parameters(prompt, max_tokens, temperature)
            self._validate_model(model, requires_vision=bool(image))

            # Warn if response_format provided (not supported)
            if response_format and not self.suppress_format_warning:
                logger.warning(
                    "response_format is not supported by Anthropic API and "
                    "will be ignored. Use prompt engineering for structured "
                    "outputs (e.g., 'Return JSON with keys: ...')."
                )

            # Build messages
            messages: List[Dict[str, Any]] = []

            if image:
                # Validate and detect image format
                self._validate_image_constraints(image)
                media_type = self._detect_image_format(image)

                # Encode image
                image_base64 = base64.b64encode(image).decode("utf-8")

                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_base64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                )
            else:
                messages.append({"role": "user", "content": prompt})

            # Call API with retry logic
            last_exception = None
            for attempt in range(self.max_retries):
                try:
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
                        provider=self.PROVIDER_NAME,
                        success=True,
                        error_message=None,
                    )

                except anthropic.RateLimitError as e:
                    last_exception = e
                    if attempt < self.max_retries - 1:
                        wait_time = 2**attempt  # Exponential backoff
                        logger.warning(
                            f"Rate limit hit (attempt {attempt + 1}/"
                            f"{self.max_retries}). Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"Rate limit exceeded after {self.max_retries} " f"attempts"
                        )
                        raise LLMRateLimitError(
                            f"Rate limit exceeded after {self.max_retries} "
                            f"retries: {str(e)}"
                        ) from e

        except anthropic.AuthenticationError as e:
            logger.error(f"Anthropic authentication failed: {e}")
            raise LLMAuthenticationError(f"Authentication failed: {str(e)}") from e

        except anthropic.APIError as e:
            logger.exception(f"Anthropic API error: {e}")
            raise LLMAPIError(f"API error: {str(e)}") from e

        except (LLMImageError, LLMModelError, ValueError) as e:
            # Re-raise validation errors
            logger.error(f"Validation error: {e}")
            raise

        except Exception as e:
            logger.exception(f"Unexpected error in Anthropic API call: {e}")
            raise LLMAPIError(f"Unexpected error: {str(e)}") from e

    def calculate_cost(
        self,
        tokens_used: TokenUsage,
        model: str,
    ) -> float:
        """Calculate cost for a completed API call.

        Args:
            tokens_used: Token usage from LLMResponse.
            model: Model identifier used for the call.

        Returns:
            Cost in USD.

        Raises:
            LLMModelError: If model not found in registry.

        Example:
            >>> tokens = TokenUsage(
            ...     input_tokens=1000,
            ...     output_tokens=500,
            ...     image_count=1
            ... )
            >>> cost = provider.calculate_cost(
            ...     tokens, "claude-3-sonnet-20240229"
            ... )
            >>> print(f"${cost:.4f}")
        """
        try:
            model_spec = self.model_registry.get_model(model)
        except KeyError:
            raise LLMModelError(f"Model '{model}' not found in registry")

        return model_spec.calculate_cost(
            input_tokens=tokens_used.input_tokens,
            output_tokens=tokens_used.output_tokens,
            images=tokens_used.image_count,
        )

    def get_available_models(self) -> List[str]:
        """Retrieve list of Anthropic models from registry.

        Returns:
            List of model identifiers (canonical names and model IDs).
        """
        return [
            model_spec.model_id
            for model_spec in self.model_registry.MODELS.values()
            if model_spec.provider.value == self.PROVIDER_NAME
        ]

    def validate_credentials(self) -> bool:
        """Test API key validity with a minimal API call.

        Sends a minimal request to claude-3-haiku-20240307 to verify
        authentication. Does not check quota limits.

        Returns:
            True if credentials are valid and API is accessible.
            False if authentication fails or API is unreachable.

        Note:
            This method consumes ~10 tokens and counts toward rate limits.

        Example:
            >>> provider = AnthropicProvider(api_key="sk-ant-...")
            >>> if provider.validate_credentials():
            ...     print("Credentials valid")
            ... else:
            ...     print("Authentication failed")
        """
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            logger.info("Anthropic credential validation successful")
            return True
        except Exception as e:
            logger.exception(f"Credential validation failed: {e}")
            return False

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return (
            f"AnthropicProvider("
            f"api_key={'***' if self.api_key else None}, "
            f"max_retries={self.max_retries})"
        )
