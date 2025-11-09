"""
OpenAI LLM provider implementation for text and vision models.

This module implements the BaseLLMProvider interface for OpenAI's GPT models,
including GPT-4, GPT-4 Turbo, and GPT-4 Vision variants. Supports both text-only
and image-based completions with configurable parameters.

Features:
    - Automatic retry with exponential backoff for transient errors
    - Vision model validation and multi-image support
    - Cost estimation and budget warnings
    - Secure API key handling with format validation
    - Streaming response support
    - Comprehensive error handling with custom exceptions

Example:
    >>> provider = OpenAIProvider(
    ...     api_key="sk-proj-...",
    ...     timeout=60.0,
    ...     max_retries=3
    ... )
    >>> response = provider.call(
    ...     prompt="Describe this drawing",
    ...     image=image_bytes,
    ...     model="gpt-4o",
    ...     max_tokens=500,
    ...     temperature=0.0
    ... )
    >>> print(f"Cost: ${response.estimated_cost_usd:.4f}")

Model Pricing (per 1M tokens):
    - gpt-4o: $2.50 input / $10.00 output
    - gpt-4-turbo: $10.00 input / $30.00 output
    - gpt-3.5-turbo: $0.50 input / $1.50 output

Note:
    Requires 'openai' package: pip install openai>=1.0.0
"""

import base64
import io
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Tuple

from .base_provider import BaseLLMProvider, LLMResponse, TokenUsage

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================


class OpenAIProviderError(Exception):
    """Base exception for OpenAI provider errors."""

    pass


class InvalidModelError(OpenAIProviderError):
    """Raised when an unsupported model is requested."""

    pass


class InvalidParameterError(OpenAIProviderError):
    """Raised when invalid parameters are provided."""

    pass


class RateLimitError(OpenAIProviderError):
    """Raised when API rate limit is exceeded."""

    pass


class AuthenticationError(OpenAIProviderError):
    """Raised when API authentication fails."""

    pass


class ImageValidationError(OpenAIProviderError):
    """Raised when image validation fails."""

    pass


# ============================================================================
# Provider Implementation
# ============================================================================


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI API provider for LLM text and vision completions.

    Implements the BaseLLMProvider interface with support for GPT-4 family models.
    Handles lazy client initialization, credential validation, and both text-only
    and vision-enabled requests with automatic retry logic.

    Attributes:
        api_key: OpenAI API authentication key.
        base_url: Optional custom API endpoint URL.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts for transient errors.
        organization: Optional organization ID for billing/tracking.
        http_client: Optional custom HTTP client configuration.

    Supported Models:
        - gpt-4o: Latest multimodal model (128K context)
        - gpt-4-turbo-2024-04-09: Fast GPT-4 variant (128K context)
        - gpt-4-vision-preview: Vision-enabled GPT-4 (128K context)
        - gpt-4o-mini: Cost-effective multimodal (128K context)
        - gpt-3.5-turbo: Baseline text model (16K context)

    Model Pricing (per 1M tokens):
        - gpt-4o: $2.50 input / $10.00 output
        - gpt-4-turbo: $10.00 input / $30.00 output
        - gpt-4o-mini: $0.15 input / $0.60 output
        - gpt-3.5-turbo: $0.50 input / $1.50 output
    """

    # Vision-capable models for validation
    VISION_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4-vision-preview", "gpt-4-turbo"}

    # Model pricing (per 1M tokens)
    MODEL_PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4-turbo-2024-04-09": {"input": 10.00, "output": 30.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4-vision-preview": {"input": 10.00, "output": 30.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }

    # Model context limits (tokens)
    MODEL_CONTEXT_LIMITS = {
        "gpt-4o": 128000,
        "gpt-4-turbo-2024-04-09": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4-vision-preview": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16385,
    }

    # Image constraints
    MAX_IMAGE_SIZE_MB = 20
    SUPPORTED_IMAGE_FORMATS = {"image/jpeg", "image/png", "image/gif", "image/webp"}

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        organization: Optional[str] = None,
        http_client: Optional[Any] = None,
    ) -> None:
        """
        Initialize OpenAI provider with credentials and configuration.

        Args:
            api_key: OpenAI API key (format: 'sk-...' or 'sk-proj-...').
            base_url: Optional custom base URL for Azure OpenAI or proxies.
            timeout: Request timeout in seconds (default: 60.0).
            max_retries: Maximum retry attempts for transient errors (default: 3).
            organization: Optional organization ID for billing tracking.
            http_client: Optional custom httpx.Client for advanced proxy/TLS config.

        Raises:
            ValueError: If api_key is empty or has invalid format.

        Example:
            >>> # Basic initialization
            >>> provider = OpenAIProvider(api_key="sk-proj-...")
            >>>
            >>> # With organization and custom timeout
            >>> provider = OpenAIProvider(
            ...     api_key="sk-proj-...",
            ...     organization="org-123",
            ...     timeout=120.0,
            ...     max_retries=5
            ... )
        """
        if not api_key or not api_key.strip():
            raise ValueError("api_key cannot be empty")

        # Validate API key format
        if not (api_key.startswith("sk-") or api_key.startswith("sk-proj-")):
            logger.warning(
                "API key does not match expected format (sk-... or sk-proj-...)"
            )

        self.api_key: str = api_key
        self.base_url: Optional[str] = base_url
        self.timeout: float = timeout
        self.max_retries: int = max_retries
        self.organization: Optional[str] = organization
        self.http_client: Optional[Any] = http_client
        self._client: Optional["OpenAI"] = None

        # Mask API key for logging
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        logger.info(f"OpenAI provider initialized (key: {masked_key})")

    def _get_client(self) -> "OpenAI":
        """
        Lazy initialization of OpenAI client.

        Creates the OpenAI client on first call to avoid initialization overhead
        when provider is instantiated but not used.

        Returns:
            OpenAI: Configured OpenAI client instance.

        Raises:
            ImportError: If openai package is not installed.

        Note:
            Client is cached after first initialization. Subsequent calls return
            the same instance.
        """
        if self._client is None:
            try:
                from openai import OpenAI

                client_kwargs: Dict[str, Any] = {
                    "api_key": self.api_key,
                    "timeout": self.timeout,
                    "max_retries": self.max_retries,
                }

                if self.base_url:
                    client_kwargs["base_url"] = self.base_url

                if self.organization:
                    client_kwargs["organization"] = self.organization

                if self.http_client:
                    client_kwargs["http_client"] = self.http_client

                self._client = OpenAI(**client_kwargs)

                logger.debug("OpenAI client initialized successfully")

            except ImportError as e:
                logger.error("OpenAI package not installed")
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai>=1.0.0"
                ) from e

        return self._client

    def call(
        self,
        prompt: str,
        image: Optional[bytes] = None,
        model: str = "gpt-4o",
        max_tokens: int = 1000,
        temperature: float = 0.0,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        stream: bool = False,
        images: Optional[List[bytes]] = None,
    ) -> LLMResponse:
        """
        Execute OpenAI API completion request with retry logic.

        Supports both text-only and vision-enabled completions. Automatically
        constructs appropriate message format based on presence of image data.
        Implements exponential backoff for transient errors.

        Args:
            prompt: User prompt text. For vision requests, describes what
                to extract or analyze from images.
            image: Optional single image bytes (JPEG/PNG/WebP/GIF).
            model: OpenAI model identifier (default: 'gpt-4o').
            max_tokens: Maximum tokens in response (1-4096 typical range).
            temperature: Sampling temperature (0.0=deterministic, 1.0=creative).
            response_format: Optional format specification. Use {"type": "json_object"}
                to force JSON output.
            seed: Optional integer for deterministic sampling (beta feature).
            stream: If True, returns streaming response (not yet implemented).
            images: Optional list of multiple images (alternative to single image).

        Returns:
            LLMResponse: Structured response containing:
                - content: Model's text response
                - tokens_used: TokenUsage breakdown (input/output/images)
                - model_used: Actual model used
                - provider: Always "openai"
                - success: True if call succeeded
                - error_message: Error details if success=False
                - request_id: OpenAI request ID for tracing
                - finish_reason: Completion reason (stop, length, etc.)
                - timestamp: Response timestamp
                - estimated_cost_usd: Estimated cost in USD

        Raises:
            InvalidModelError: If model is not supported.
            InvalidParameterError: If parameters are out of valid range.
            ImageValidationError: If image validation fails.

        Example:
            >>> # Text-only request
            >>> response = provider.call(
            ...     prompt="Explain quantum computing",
            ...     model="gpt-4o",
            ...     max_tokens=500,
            ...     temperature=0.0
            ... )
            >>>
            >>> # Vision request with deterministic output
            >>> response = provider.call(
            ...     prompt="Extract part number from this drawing",
            ...     image=pdf_page_bytes,
            ...     model="gpt-4o",
            ...     max_tokens=500,
            ...     temperature=0.0,
            ...     seed=42
            ... )
            >>> print(f"Cost: ${response.estimated_cost_usd:.4f}")

        Note:
            - Vision models required when image/images provided
            - Images are base64-encoded before sending to API
            - Automatically retries on rate limits (429) and network errors
            - Cost estimation uses hardcoded pricing (may drift from actual)
        """
        # Validate model
        available_models = self.get_available_models()
        if model not in available_models:
            raise InvalidModelError(
                f"Model '{model}' not in available models: {available_models}"
            )

        # Validate parameters
        self._validate_parameters(model, max_tokens, temperature)

        # Handle images (single or multiple)
        image_list: List[bytes] = []
        if image:
            image_list.append(image)
        if images:
            image_list.extend(images)

        # Validate vision model requirement
        if image_list and not self._is_vision_model(model):
            raise InvalidModelError(
                f"Model '{model}' does not support vision. "
                f"Use one of: {self.VISION_MODELS}"
            )

        # Validate images
        if image_list:
            for idx, img_bytes in enumerate(image_list):
                self._validate_image(img_bytes, idx)

        # Streaming not yet implemented
        if stream:
            raise NotImplementedError("Streaming support not yet implemented")

        client = self._get_client()
        start_time = time.time()

        # Retry loop with exponential backoff
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                # Build messages
                messages = self._build_messages(prompt, image_list)

                # Build API call kwargs
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }

                if response_format:
                    kwargs["response_format"] = response_format

                if seed is not None:
                    kwargs["seed"] = seed

                logger.debug(
                    f"OpenAI API call (attempt {attempt + 1}/{self.max_retries}): "
                    f"model={model}, max_tokens={max_tokens}, images={len(image_list)}"
                )

                response = client.chat.completions.create(**kwargs)

                # Validate response structure
                if not response.choices:
                    raise ValueError("API returned empty choices list")

                # Extract content and metadata
                choice = response.choices[0]
                content = choice.message.content or ""
                finish_reason = choice.finish_reason or "unknown"

                tokens_used = TokenUsage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    image_count=len(image_list),
                )

                # Calculate cost
                estimated_cost = self._estimate_cost(model, tokens_used)

                # Log cost warning if high
                if estimated_cost > 0.10:
                    logger.warning(
                        f"High API call cost: ${estimated_cost:.4f} "
                        f"({tokens_used.input_tokens} input + "
                        f"{tokens_used.output_tokens} output tokens)"
                    )

                latency_ms = (time.time() - start_time) * 1000

                logger.info(
                    f"OpenAI API success: {tokens_used.input_tokens} input, "
                    f"{tokens_used.output_tokens} output tokens, "
                    f"${estimated_cost:.4f}, {latency_ms:.0f}ms"
                )

                return LLMResponse(
                    content=content,
                    tokens_used=tokens_used,
                    model_used=model,
                    provider="openai",
                    success=True,
                    request_id=getattr(response, "id", None),
                    finish_reason=finish_reason,
                    timestamp=datetime.now(),
                    estimated_cost_usd=estimated_cost,
                )

            except Exception as e:
                last_error = e
                error_type = type(e).__name__

                # Check if error is retryable
                if not self._is_retryable_error(e):
                    logger.error(f"Non-retryable error: {error_type}: {e}")
                    break

                # Calculate backoff delay
                if attempt < self.max_retries - 1:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(
                        f"Retryable error ({error_type}), "
                        f"retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Max retries ({self.max_retries}) exceeded: {error_type}: {e}"
                    )

        # All retries exhausted
        error_msg = f"API call failed after {self.max_retries} attempts: {last_error}"
        logger.error(error_msg)

        return LLMResponse(
            content="",
            tokens_used=TokenUsage(0, 0, 0),
            model_used=model,
            provider="openai",
            success=False,
            error_message=error_msg,
            timestamp=datetime.now(),
        )

    def _build_messages(self, prompt: str, images: List[bytes]) -> List[Dict[str, Any]]:
        """
        Build OpenAI API message format.

        Args:
            prompt: User prompt text.
            images: List of image bytes (empty for text-only).

        Returns:
            List of message dictionaries in OpenAI format.
        """
        if images:
            # Vision model format
            content_parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

            for image_bytes in images:
                image_b64 = self._encode_image(image_bytes)
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    }
                )

            return [{"role": "user", "content": content_parts}]
        else:
            # Text-only format
            return [{"role": "user", "content": prompt}]

    def _validate_parameters(
        self, model: str, max_tokens: int, temperature: float
    ) -> None:
        """
        Validate API call parameters.

        Args:
            model: Model identifier.
            max_tokens: Maximum response tokens.
            temperature: Sampling temperature.

        Raises:
            InvalidParameterError: If parameters are out of valid range.
        """
        # Validate max_tokens
        context_limit = self.MODEL_CONTEXT_LIMITS.get(model, 128000)
        if max_tokens < 1 or max_tokens > context_limit:
            raise InvalidParameterError(
                f"max_tokens must be between 1 and {context_limit} for {model}"
            )

        # Validate temperature
        if not (0.0 <= temperature <= 2.0):
            raise InvalidParameterError("temperature must be between 0.0 and 2.0")

    def _validate_image(self, image_bytes: bytes, index: int = 0) -> None:
        """
        Validate image data before sending to API.

        Args:
            image_bytes: Raw image data.
            index: Image index for error messages.

        Raises:
            ImageValidationError: If image validation fails.
        """
        if not image_bytes:
            raise ImageValidationError(f"Image {index} is empty")

        if not isinstance(image_bytes, bytes):
            raise ImageValidationError(
                f"Image {index} must be bytes, got {type(image_bytes)}"
            )

        # Check size
        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > self.MAX_IMAGE_SIZE_MB:
            raise ImageValidationError(
                f"Image {index} size ({size_mb:.1f}MB) exceeds "
                f"maximum {self.MAX_IMAGE_SIZE_MB}MB"
            )

        # Check format (basic magic number check)
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(image_bytes))
            format_mime = f"image/{img.format.lower()}"

            if format_mime not in self.SUPPORTED_IMAGE_FORMATS:
                raise ImageValidationError(
                    f"Image {index} format '{format_mime}' not supported. "
                    f"Use: {self.SUPPORTED_IMAGE_FORMATS}"
                )

        except ImportError:
            # PIL not available, skip format check
            logger.debug("PIL not available, skipping image format validation")
        except Exception as e:
            raise ImageValidationError(f"Image {index} validation failed: {e}") from e

    def _encode_image(self, image_bytes: bytes) -> str:
        """
        Encode image bytes to base64 string for API transmission.

        Args:
            image_bytes: Raw image data (JPEG, PNG, etc.).

        Returns:
            str: Base64-encoded image string suitable for OpenAI API.

        Raises:
            ImageValidationError: If encoding fails.
        """
        try:
            return base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            raise ImageValidationError(f"Failed to encode image: {e}") from e

    def _is_vision_model(self, model: str) -> bool:
        """
        Check if a model supports vision capabilities.

        Args:
            model: Model identifier to check.

        Returns:
            bool: True if model supports image inputs.
        """
        # Exact match or substring match for versioned models
        return model in self.VISION_MODELS or any(
            vm in model for vm in self.VISION_MODELS
        )

    def _estimate_cost(self, model: str, tokens: TokenUsage) -> float:
        """
        Estimate API call cost in USD.

        Args:
            model: Model used for the call.
            tokens: Token usage breakdown.

        Returns:
            float: Estimated cost in USD.

        Note:
            Uses hardcoded pricing table. Actual costs may vary slightly
            due to image token calculations and pricing updates.
        """
        pricing = self.MODEL_PRICING.get(model)
        if not pricing:
            logger.warning(f"No pricing data for model '{model}', cost estimate = 0")
            return 0.0

        input_cost = (tokens.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (tokens.output_tokens / 1_000_000) * pricing["output"]

        # Image costs are included in input tokens by OpenAI
        return input_cost + output_cost

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable.

        Args:
            error: Exception raised during API call.

        Returns:
            bool: True if error is transient and should be retried.
        """
        error_str = str(error).lower()

        # Retryable conditions
        retryable_patterns = [
            "rate limit",
            "429",
            "timeout",
            "connection",
            "network",
            "temporary",
            "503",
            "502",
            "500",
        ]

        return any(pattern in error_str for pattern in retryable_patterns)

    def _calculate_backoff_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay.

        Args:
            attempt: Current attempt number (0-indexed).

        Returns:
            float: Delay in seconds.
        """
        # Exponential backoff: 2^attempt seconds (1s, 2s, 4s, 8s, ...)
        base_delay = 2**attempt
        # Cap at 30 seconds
        return min(base_delay, 30.0)

    def get_available_models(self) -> list[str]:
        """
        Get list of supported OpenAI model identifiers.

        Returns:
            list[str]: Model IDs that can be passed to call() method.
                Ordered from most to least capable.

        Note:
            This is a static list of commonly-used models. OpenAI's model
            availability may change. Use validate_credentials() to test
            specific model access.

        Example:
            >>> models = provider.get_available_models()
            >>> print(models[0])  # Most capable model
            'gpt-4o'
        """
        return [
            "gpt-4o",
            "gpt-4-turbo-2024-04-09",
            "gpt-4-turbo",
            "gpt-4-vision-preview",
            "gpt-4o-mini",
            "gpt-4",
            "gpt-3.5-turbo",
        ]

    def validate_credentials(self) -> bool:
        """
        Test OpenAI API credentials with minimal request.

        Makes a lightweight API call to verify that the api_key is valid and
        has access to at least one model. Uses gpt-3.5-turbo with 5-token limit
        to minimize cost (<$0.001 per validation).

        Returns:
            bool: True if credentials are valid and API is accessible,
                False otherwise.

        Note:
            - This method makes a real API call and incurs minimal cost
            - Network errors or service outages will return False
            - Successful validation does not guarantee access to all models

        Example:
            >>> provider = OpenAIProvider(api_key="sk-...")
            >>> if provider.validate_credentials():
            ...     print("Ready to process drawings")
            ... else:
            ...     print("Invalid API key or OpenAI service unavailable")
        """
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )
            logger.info("OpenAI credentials validated successfully")
            return True
        except Exception as e:
            logger.error(f"OpenAI credential validation failed: {e}")
            return False

    def close(self) -> None:
        """
        Close the OpenAI client and cleanup resources.

        Call this method when the provider is no longer needed to release
        network connections and other resources.

        Example:
            >>> provider = OpenAIProvider(api_key="sk-...")
            >>> try:
            ...     response = provider.call(...)
            ... finally:
            ...     provider.close()
        """
        if self._client is not None:
            try:
                self._client.close()
                logger.debug("OpenAI client closed successfully")
            except Exception as e:
                logger.warning(f"Error closing OpenAI client: {e}")
            finally:
                self._client = None

    def __enter__(self) -> "OpenAIProvider":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
