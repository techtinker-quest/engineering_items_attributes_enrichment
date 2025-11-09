"""
Google AI (Gemini) LLM provider implementation for text and vision models.

This module implements the BaseLLMProvider interface for Google's Gemini models,
including Gemini 1.5 Pro, Flash, and legacy Pro/Pro Vision variants. Supports both
text-only and image-based completions with configurable parameters.

Features:
    - Automatic retry with exponential backoff for transient errors
    - Multi-image support for batch vision analysis
    - Accurate cost estimation with native token counting
    - Secure API key handling with Vertex AI support
    - Comprehensive error handling with custom exceptions
    - PIL-based image validation with dimension checking
    - Configurable retry and budget controls
    - Structured logging with request tracing

Example:
    >>> provider = GoogleProvider(
    ...     api_key="AIza...",
    ...     timeout=60.0,
    ...     max_retries=3,
    ...     budget_limit_usd=10.0
    ... )
    >>> response = provider.call(
    ...     prompt="Extract dimensions from this drawing",
    ...     images=[image_bytes],
    ...     model="gemini-1.5-pro",
    ...     max_tokens=1000,
    ...     temperature=0.0
    ... )
    >>> print(f"Cost: ${response.estimated_cost_usd:.4f}")

Model Pricing (per 1M tokens):
    - gemini-1.5-pro: $1.25 input / $5.00 output (2M context)
    - gemini-1.5-flash: $0.075 input / $0.30 output (1M context)
    - gemini-pro: $0.50 input / $1.50 output (32K context)

Note:
    Requires 'google-generativeai' package: pip install google-generativeai

Exceptions:
    GoogleProviderError: Base exception for all provider errors.
    InvalidModelError: Unsupported model requested.
    InvalidParameterError: Invalid API parameters.
    RateLimitError: API rate limit exceeded.
    QuotaExceededError: API quota exhausted.
    AuthenticationError: Authentication failure.
    ImageValidationError: Image validation failure.
    BudgetExceededError: Cost exceeds budget limit.
    UnsupportedFeatureError: Unsupported feature requested.
"""

import io
import logging
import time
import types
import warnings
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from PIL import Image

from .base_provider import BaseLLMProvider, LLMResponse, TokenUsage

if TYPE_CHECKING:
    import google.generativeai as genai

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================


class GoogleProviderError(Exception):
    """Base exception for Google provider errors."""

    pass


class InvalidModelError(GoogleProviderError):
    """Raised when an unsupported model is requested."""

    pass


class InvalidParameterError(GoogleProviderError):
    """Raised when invalid parameters are provided."""

    pass


class RateLimitError(GoogleProviderError):
    """Raised when API rate limit is exceeded."""

    pass


class QuotaExceededError(GoogleProviderError):
    """Raised when API quota is exceeded."""

    pass


class AuthenticationError(GoogleProviderError):
    """Raised when API authentication fails."""

    pass


class ImageValidationError(GoogleProviderError):
    """Raised when image validation fails."""

    pass


class BudgetExceededError(GoogleProviderError):
    """Raised when estimated cost exceeds budget limit."""

    pass


class UnsupportedFeatureError(GoogleProviderError):
    """Raised when an unsupported feature is requested."""

    pass


# ============================================================================
# Model Metadata
# ============================================================================


class GeminiModel(str, Enum):
    """Supported Gemini model identifiers."""

    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_PRO_LATEST = "gemini-1.5-pro-latest"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_1_5_FLASH_LATEST = "gemini-1.5-flash-latest"
    GEMINI_PRO = "gemini-pro"
    GEMINI_PRO_VISION = "gemini-pro-vision"


@dataclass(frozen=True)
class ModelInfo:
    """Model metadata including pricing and capabilities."""

    name: str
    context_limit: int
    input_price_per_1m: float
    output_price_per_1m: float
    supports_vision: bool
    supports_json_mode: bool


# Model metadata registry
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    GeminiModel.GEMINI_1_5_PRO.value: ModelInfo(
        name="gemini-1.5-pro",
        context_limit=2097152,
        input_price_per_1m=1.25,
        output_price_per_1m=5.00,
        supports_vision=True,
        supports_json_mode=True,
    ),
    GeminiModel.GEMINI_1_5_PRO_LATEST.value: ModelInfo(
        name="gemini-1.5-pro-latest",
        context_limit=2097152,
        input_price_per_1m=1.25,
        output_price_per_1m=5.00,
        supports_vision=True,
        supports_json_mode=True,
    ),
    GeminiModel.GEMINI_1_5_FLASH.value: ModelInfo(
        name="gemini-1.5-flash",
        context_limit=1048576,
        input_price_per_1m=0.075,
        output_price_per_1m=0.30,
        supports_vision=True,
        supports_json_mode=True,
    ),
    GeminiModel.GEMINI_1_5_FLASH_LATEST.value: ModelInfo(
        name="gemini-1.5-flash-latest",
        context_limit=1048576,
        input_price_per_1m=0.075,
        output_price_per_1m=0.30,
        supports_vision=True,
        supports_json_mode=True,
    ),
    GeminiModel.GEMINI_PRO.value: ModelInfo(
        name="gemini-pro",
        context_limit=32768,
        input_price_per_1m=0.50,
        output_price_per_1m=1.50,
        supports_vision=False,
        supports_json_mode=False,
    ),
    GeminiModel.GEMINI_PRO_VISION.value: ModelInfo(
        name="gemini-pro-vision",
        context_limit=16384,
        input_price_per_1m=0.50,
        output_price_per_1m=1.50,
        supports_vision=True,
        supports_json_mode=False,
    ),
}


# ============================================================================
# Provider Implementation
# ============================================================================


class GoogleProvider(BaseLLMProvider):
    """
    Google AI (Gemini) API provider for LLM text and vision completions.

    Implements the BaseLLMProvider interface with support for Gemini model family.
    Handles lazy client initialization, credential validation, and both text-only
    and vision-enabled requests with automatic retry logic.

    Attributes:
        api_key: Google AI API authentication key.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts for transient errors.
        budget_limit_usd: Maximum allowed cost per call (None = unlimited).
        base_delay: Base delay for exponential backoff (seconds).
        max_delay: Maximum delay for exponential backoff (seconds).
        use_vertex_ai: Whether to use Vertex AI endpoint.
        project_id: GCP project ID (required for Vertex AI).
        location: GCP location (required for Vertex AI).

    Supported Models:
        - gemini-1.5-pro: Latest multimodal model (2M context)
        - gemini-1.5-flash: Fast multimodal model (1M context)
        - gemini-pro: Baseline text model (32K context)
        - gemini-pro-vision: Legacy vision model (16K context)
    """

    # Image constraints
    MAX_IMAGE_SIZE_MB: int = 20
    MIN_IMAGE_DIMENSION: int = 336  # Minimum width/height for vision models
    SUPPORTED_IMAGE_FORMATS: set = {"JPEG", "PNG", "WEBP", "GIF"}

    def __init__(
        self,
        api_key: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        budget_limit_usd: Optional[float] = None,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        use_vertex_ai: bool = False,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
    ) -> None:
        """
        Initialize Google AI provider with credentials and configuration.

        Args:
            api_key: Google AI API key (format: 'AIza...') or service account JSON.
            timeout: Request timeout in seconds (default: 60.0).
            max_retries: Maximum retry attempts for transient errors (default: 3).
            budget_limit_usd: Maximum cost per call in USD (None = unlimited).
            base_delay: Base delay for exponential backoff (default: 1.0s).
            max_delay: Maximum backoff delay (default: 30.0s).
            use_vertex_ai: Use Vertex AI endpoint instead of AI Studio (default: False).
            project_id: GCP project ID (required if use_vertex_ai=True).
            location: GCP location (required if use_vertex_ai=True, e.g. 'us-central1').

        Raises:
            ValueError: If api_key is empty or Vertex AI config is incomplete.

        Example:
            >>> # Standard AI Studio usage
            >>> provider = GoogleProvider(api_key="AIza...")
            >>>
            >>> # With budget control
            >>> provider = GoogleProvider(
            ...     api_key="AIza...",
            ...     budget_limit_usd=5.0,
            ...     max_retries=5
            ... )
            >>>
            >>> # Vertex AI usage
            >>> provider = GoogleProvider(
            ...     api_key="path/to/service-account.json",
            ...     use_vertex_ai=True,
            ...     project_id="my-project",
            ...     location="us-central1"
            ... )
        """
        if not api_key or not api_key.strip():
            raise ValueError("api_key cannot be empty")

        if use_vertex_ai and (not project_id or not location):
            raise ValueError("project_id and location required for Vertex AI")

        # Validate API key format for AI Studio
        if not use_vertex_ai and not api_key.startswith("AIza"):
            logger.warning("API key does not match expected format (AIza...)")

        self.api_key: str = api_key
        self.timeout: float = timeout
        self.max_retries: int = max_retries
        self.budget_limit_usd: Optional[float] = budget_limit_usd
        self.base_delay: float = base_delay
        self.max_delay: float = max_delay
        self.use_vertex_ai: bool = use_vertex_ai
        self.project_id: Optional[str] = project_id
        self.location: Optional[str] = location
        self._genai: Optional[types.ModuleType] = None

        # Mask API key for logging
        masked_key: str = (
            f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        )
        logger.info(
            f"Google provider initialized (key: {masked_key}, "
            f"vertex_ai: {use_vertex_ai}, budget: ${budget_limit_usd or 'unlimited'})"
        )

    def _get_genai(self) -> types.ModuleType:
        """
        Lazy initialization of Google GenerativeAI module.

        Creates the genai module on first call to avoid initialization overhead
        when provider is instantiated but not used.

        Returns:
            google.generativeai module instance.

        Raises:
            ImportError: If google-generativeai package is not installed.

        Note:
            Module is cached after first initialization. Subsequent calls return
            the same instance.
        """
        if self._genai is None:
            try:
                import google.generativeai as genai

                if self.use_vertex_ai:
                    # Vertex AI configuration
                    try:
                        import vertexai  # type: ignore[import-untyped]

                        vertexai.init(
                            project=self.project_id,
                            location=self.location,
                            credentials=(
                                self.api_key if self.api_key.endswith(".json") else None
                            ),
                        )
                        logger.debug("Vertex AI initialized successfully")
                    except ImportError as vertex_err:
                        raise ImportError(
                            "Vertex AI package not installed. "
                            "Install with: pip install google-cloud-aiplatform"
                        ) from vertex_err
                else:
                    # AI Studio configuration
                    genai.configure(api_key=self.api_key)
                    logger.debug("Google AI Studio configured successfully")

                self._genai = genai

            except ImportError as e:
                logger.error("Google GenerativeAI package not installed")
                raise ImportError(
                    "Google GenerativeAI package not installed. "
                    "Install with: pip install google-generativeai"
                ) from e

        return self._genai

    def get_model_info(self, model: str) -> ModelInfo:
        """
        Get metadata for a specific model.

        Args:
            model: Model identifier.

        Returns:
            ModelInfo with pricing, context limits, and capabilities.

        Raises:
            InvalidModelError: If model is not in registry.

        Example:
            >>> info = provider.get_model_info("gemini-1.5-pro")
            >>> print(f"Context: {info.context_limit}, Vision: {info.supports_vision}")
        """
        if model not in MODEL_REGISTRY:
            raise InvalidModelError(
                f"Model '{model}' not in registry: {list(MODEL_REGISTRY.keys())}"
            )
        return MODEL_REGISTRY[model]

    def call(
        self,
        prompt: str,
        image: Optional[bytes] = None,
        model: str = "gemini-1.5-pro",
        max_tokens: int = 1000,
        temperature: float = 0.0,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        stream: bool = False,
        images: Optional[List[bytes]] = None,
        safety_settings: Optional[Dict[str, Any]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> LLMResponse:
        """
        Execute Google AI API completion request with retry logic.

        Supports both text-only and vision-enabled completions. Automatically
        constructs appropriate message format based on presence of image data.
        Implements exponential backoff for transient errors.

        Args:
            prompt: User prompt text. For vision requests, describes what
                to extract or analyze from images.
            image: Optional single image bytes (deprecated, use images instead).
            model: Gemini model identifier (default: 'gemini-1.5-pro').
            max_tokens: Maximum tokens in response (1-8192 typical range).
            temperature: Sampling temperature (0.0=deterministic, 2.0=creative).
            response_format: Optional format specification. Use {"type": "json_object"}
                to force JSON output (only supported on newer models).
            seed: Optional integer for deterministic sampling (not supported).
            stream: If True, returns streaming response (not yet implemented).
            images: Optional list of image bytes (JPEG/PNG/WebP/GIF).
            safety_settings: Optional content filtering settings.
            top_p: Optional nucleus sampling parameter (0.0-1.0).
            top_k: Optional top-k sampling parameter (1-40).

        Returns:
            LLMResponse: Structured response containing content, tokens, cost, etc.

        Raises:
            InvalidModelError: If model is not supported.
            InvalidParameterError: If parameters are out of valid range.
            ImageValidationError: If image validation fails.
            BudgetExceededError: If estimated cost exceeds budget limit.
            UnsupportedFeatureError: If streaming is requested.
            RateLimitError: If API rate limit is exceeded.
            QuotaExceededError: If API quota is exceeded.
            AuthenticationError: If API authentication fails.

        Example:
            >>> response = provider.call(
            ...     prompt="Extract part number from this drawing",
            ...     images=[pdf_page_bytes],
            ...     model="gemini-1.5-pro",
            ...     max_tokens=500,
            ...     temperature=0.0,
            ...     safety_settings={"HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"}
            ... )
            >>> print(f"Cost: ${response.estimated_cost_usd:.4f}")
        """
        # Get model metadata
        try:
            model_info = self.get_model_info(model)
        except InvalidModelError:
            raise InvalidModelError(
                f"Model '{model}' not supported. Use: {self.get_available_models()}"
            )

        # Validate parameters
        self._validate_parameters(model_info, max_tokens, temperature, top_p, top_k)

        # Handle images (single or multiple)
        image_list: List[bytes] = []
        if image:
            warnings.warn(
                "'image' parameter is deprecated, use 'images' instead",
                DeprecationWarning,
                stacklevel=2,
            )
            image_list.append(image)
        if images:
            image_list.extend(images)

        # Validate vision model requirement
        if image_list and not model_info.supports_vision:
            raise InvalidModelError(
                f"Model '{model}' does not support vision. Use a vision-capable model."
            )

        # Validate images
        if image_list:
            for idx, img_bytes in enumerate(image_list):
                self._validate_image(img_bytes, idx)

        # Validate JSON mode support
        if response_format and response_format.get("type") == "json_object":
            if not model_info.supports_json_mode:
                raise InvalidParameterError(
                    f"Model '{model}' does not support JSON mode"
                )

        # Streaming not yet implemented
        if stream:
            raise UnsupportedFeatureError("Streaming support not yet implemented")

        # Log unsupported parameters
        if seed is not None:
            logger.warning("seed parameter is not supported by Gemini (ignored)")

        genai = self._get_genai()

        # Pre-call cost estimation and budget check
        estimated_input_tokens = self._estimate_input_tokens(genai, model, prompt)
        estimated_output_tokens = max_tokens
        estimated_cost_pre = self._calculate_cost(
            model_info,
            TokenUsage(
                estimated_input_tokens, estimated_output_tokens, len(image_list)
            ),
        )

        if self.budget_limit_usd and estimated_cost_pre > self.budget_limit_usd:
            raise BudgetExceededError(
                f"Estimated cost ${estimated_cost_pre:.4f} exceeds budget "
                f"limit ${self.budget_limit_usd:.2f}"
            )

        start_time = time.time()

        # Retry loop with exponential backoff
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                # Initialize model
                model_instance = genai.GenerativeModel(model)

                # Build content
                content = self._build_content(prompt, image_list)

                # Configure generation
                generation_config: Dict[str, Any] = {
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                }

                if top_p is not None:
                    generation_config["top_p"] = top_p

                if top_k is not None:
                    generation_config["top_k"] = top_k

                # Add JSON mode if requested
                if response_format and response_format.get("type") == "json_object":
                    generation_config["response_mime_type"] = "application/json"

                logger.debug(
                    f"Google AI API call (attempt {attempt + 1}/{self.max_retries})",
                    extra={
                        "model": model,
                        "max_tokens": max_tokens,
                        "images": len(image_list),
                        "estimated_cost": f"${estimated_cost_pre:.4f}",
                    },
                )

                # Make API call
                response = model_instance.generate_content(
                    content,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )

                # Extract content
                content_text = response.text

                # Get accurate token counts
                input_tokens = 0
                output_tokens = 0
                if hasattr(response, "usage_metadata"):
                    usage = response.usage_metadata
                    input_tokens = getattr(usage, "prompt_token_count", 0)
                    output_tokens = getattr(usage, "candidates_token_count", 0)
                else:
                    # Fallback to estimation
                    input_tokens = estimated_input_tokens
                    output_tokens = self._estimate_tokens(content_text)

                tokens_used = TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    image_count=len(image_list),
                )

                # Calculate actual cost
                actual_cost = self._calculate_cost(model_info, tokens_used)

                # Post-call budget validation
                if self.budget_limit_usd and actual_cost > self.budget_limit_usd:
                    logger.error(
                        f"Actual cost ${actual_cost:.4f} exceeded budget "
                        f"${self.budget_limit_usd:.2f} (estimated: ${estimated_cost_pre:.4f})"
                    )

                # Log cost warning if high
                if actual_cost > 0.10:
                    logger.warning(
                        f"High API call cost: ${actual_cost:.4f}",
                        extra={
                            "input_tokens": tokens_used.input_tokens,
                            "output_tokens": tokens_used.output_tokens,
                        },
                    )

                latency_ms = (time.time() - start_time) * 1000

                # Extract finish reason
                finish_reason = "stop"
                if hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, "finish_reason"):
                        finish_reason = str(candidate.finish_reason).lower()

                logger.info(
                    "Google AI API success",
                    extra={
                        "input_tokens": tokens_used.input_tokens,
                        "output_tokens": tokens_used.output_tokens,
                        "cost_usd": f"${actual_cost:.4f}",
                        "latency_ms": f"{latency_ms:.0f}",
                        "finish_reason": finish_reason,
                    },
                )

                return LLMResponse(
                    content=content_text,
                    tokens_used=tokens_used,
                    model_used=model,
                    provider="google",
                    success=True,
                    request_id=getattr(response, "request_id", None),
                    finish_reason=finish_reason,
                    timestamp=datetime.now(),
                    estimated_cost_usd=actual_cost,
                )

            except (
                RateLimitError,
                QuotaExceededError,
                AuthenticationError,
            ) as custom_err:
                # Re-raise custom exceptions immediately
                last_error = custom_err
                error_type = type(custom_err).__name__
                logger.error(f"{error_type}: {custom_err}")
                raise

            except Exception as e:
                last_error = e
                error_type = type(e).__name__

                # Map to custom exceptions
                if self._is_rate_limit_error(e):
                    last_error = RateLimitError(f"Rate limit exceeded: {e}")
                elif self._is_quota_error(e):
                    last_error = QuotaExceededError(f"Quota exceeded: {e}")
                elif self._is_auth_error(e):
                    last_error = AuthenticationError(f"Authentication failed: {e}")

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
            provider="google",
            success=False,
            error_message=error_msg,
            timestamp=datetime.now(),
        )

    def _build_content(
        self, prompt: str, images: List[bytes]
    ) -> Union[str, List[Union[str, Image.Image]]]:
        """
        Build Gemini API content format.

        Args:
            prompt: User prompt text.
            images: List of image bytes (empty for text-only).

        Returns:
            String for text-only requests, or list of content parts for vision requests.

        Raises:
            ImageValidationError: If PIL image conversion fails.
        """
        if images:
            # Vision model format
            content_parts: List[Union[str, Image.Image]] = [prompt]

            for idx, image_bytes in enumerate(images):
                try:
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    content_parts.append(pil_image)
                except Exception as e:
                    raise ImageValidationError(
                        f"Failed to convert image {idx} to PIL format: {e}"
                    ) from e

            return content_parts
        else:
            # Text-only format
            return prompt

    def _validate_parameters(
        self,
        model_info: ModelInfo,
        max_tokens: int,
        temperature: float,
        top_p: Optional[float],
        top_k: Optional[int],
    ) -> None:
        """
        Validate API call parameters.

        Args:
            model_info: Model metadata.
            max_tokens: Maximum response tokens.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.

        Raises:
            InvalidParameterError: If parameters are out of valid range.
        """
        # Validate max_tokens
        if max_tokens < 1 or max_tokens > model_info.context_limit:
            raise InvalidParameterError(
                f"max_tokens must be between 1 and {model_info.context_limit} "
                f"for {model_info.name}"
            )

        # Validate temperature
        if not (0.0 <= temperature <= 2.0):
            raise InvalidParameterError("temperature must be between 0.0 and 2.0")

        # Validate top_p
        if top_p is not None and not (0.0 <= top_p <= 1.0):
            raise InvalidParameterError("top_p must be between 0.0 and 1.0")

        # Validate top_k
        if top_k is not None and not (1 <= top_k <= 40):
            raise InvalidParameterError("top_k must be between 1 and 40")

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

        # Check format and dimensions with PIL
        try:
            img = Image.open(io.BytesIO(image_bytes))

            if img.format not in self.SUPPORTED_IMAGE_FORMATS:
                raise ImageValidationError(
                    f"Image {index} format '{img.format}' not supported. "
                    f"Use: {self.SUPPORTED_IMAGE_FORMATS}"
                )

            # Check minimum dimensions
            width, height = img.size
            if width < self.MIN_IMAGE_DIMENSION or height < self.MIN_IMAGE_DIMENSION:
                raise ImageValidationError(
                    f"Image {index} dimensions ({width}x{height}) below minimum "
                    f"{self.MIN_IMAGE_DIMENSION}x{self.MIN_IMAGE_DIMENSION}"
                )

        except Exception as e:
            if isinstance(e, ImageValidationError):
                raise
            raise ImageValidationError(f"Image {index} validation failed: {e}") from e

    def _estimate_input_tokens(
        self, genai: types.ModuleType, model: str, prompt: str
    ) -> int:
        """
        Estimate input token count using Google's native counter.

        Args:
            genai: Google GenerativeAI module.
            model: Model identifier.
            prompt: Input prompt text.

        Returns:
            Estimated input token count.
        """
        try:
            model_instance = genai.GenerativeModel(model)
            result = model_instance.count_tokens(prompt)
            return result.total_tokens
        except Exception as e:
            logger.debug(f"Token counting failed, using fallback: {e}")
            return self._estimate_tokens(prompt)

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text (fallback method).

        Args:
            text: Text to estimate tokens for.

        Returns:
            Estimated token count.

        Note:
            Uses rough approximation: words * 4/3.
            For accurate counts, use _estimate_input_tokens.
        """
        words = len(text.split())
        return max(1, int(words * 4 / 3))

    def _calculate_cost(self, model_info: ModelInfo, tokens: TokenUsage) -> float:
        """
        Calculate API call cost in USD.

        Args:
            model_info: Model metadata with pricing.
            tokens: Token usage breakdown.

        Returns:
            Cost in USD.
        """
        input_cost = (tokens.input_tokens / 1_000_000) * model_info.input_price_per_1m
        output_cost = (
            tokens.output_tokens / 1_000_000
        ) * model_info.output_price_per_1m
        return input_cost + output_cost

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """
        Check if error is a rate limit error.

        Args:
            error: Exception to check.

        Returns:
            True if error indicates rate limiting.
        """
        error_str = str(error).lower()
        return any(
            pattern in error_str
            for pattern in ["rate limit", "429", "resource_exhausted"]
        )

    def _is_quota_error(self, error: Exception) -> bool:
        """
        Check if error is a quota exceeded error.

        Args:
            error: Exception to check.

        Returns:
            True if error indicates quota exhaustion.
        """
        error_str = str(error).lower()
        return "quota" in error_str

    def _is_auth_error(self, error: Exception) -> bool:
        """
        Check if error is an authentication error.

        Args:
            error: Exception to check.

        Returns:
            True if error indicates authentication failure.
        """
        error_str = str(error).lower()
        return any(
            pattern in error_str
            for pattern in ["authentication", "unauthorized", "401", "403", "api key"]
        )

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable.

        Args:
            error: Exception raised during API call.

        Returns:
            True if error is transient and should be retried.
        """
        # Check by exception type first (more reliable)
        error_type_name = type(error).__name__

        # Retryable exception types
        retryable_types = {
            "ResourceExhausted",
            "DeadlineExceeded",
            "Unavailable",
            "InternalServerError",
            "ServiceUnavailable",
            "TooManyRequests",
        }

        if error_type_name in retryable_types:
            return True

        # Fallback to string matching for unknown exception types
        error_str = str(error).lower()
        retryable_patterns = [
            "rate limit",
            "quota",
            "429",
            "timeout",
            "connection",
            "network",
            "temporary",
            "503",
            "502",
            "500",
            "unavailable",
        ]

        return any(pattern in error_str for pattern in retryable_patterns)

    def _calculate_backoff_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay.

        Args:
            attempt: Current attempt number (0-indexed).

        Returns:
            Delay in seconds (capped at max_delay).
        """
        # Exponential backoff: base_delay * 2^attempt
        delay = self.base_delay * (2**attempt)
        # Cap at max_delay
        return min(delay, self.max_delay)

    def get_available_models(self) -> List[str]:
        """
        Get list of supported Gemini model identifiers.

        Returns:
            Model IDs that can be passed to call() method.
            Ordered from most to least capable.

        Note:
            This is a static list. Google's model availability may change.
            Use validate_credentials() to test specific model access.

        Example:
            >>> models = provider.get_available_models()
            >>> print(models[0])  # Most capable model
            'gemini-1.5-pro'
        """
        return [model.value for model in GeminiModel]

    def validate_credentials(self) -> bool:
        """
        Test Google AI API credentials with minimal request.

        Makes a lightweight API call to verify that the api_key is valid and
        has access to at least one model. Uses gemini-1.5-flash with minimal
        token limit to minimize cost (<$0.001 per validation).

        Returns:
            True if credentials are valid and API is accessible,
            False otherwise.

        Note:
            - This method makes a real API call and incurs minimal cost
            - Network errors or service outages will return False
            - Successful validation does not guarantee access to all models

        Example:
            >>> provider = GoogleProvider(api_key="AIza...")
            >>> if provider.validate_credentials():
            ...     print("Ready to process drawings")
            ... else:
            ...     print("Invalid API key or Google AI service unavailable")
        """
        try:
            genai = self._get_genai()
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                "test", generation_config={"max_output_tokens": 5}
            )
            logger.info("Google AI credentials validated successfully")
            return True
        except Exception as e:
            logger.error(f"Google AI credential validation failed: {e}")
            return False

    def close(self) -> None:
        """
        Close the Google AI client and cleanup resources.

        Call this method when the provider is no longer needed to release
        resources.

        Example:
            >>> provider = GoogleProvider(api_key="AIza...")
            >>> try:
            ...     response = provider.call(...)
            ... finally:
            ...     provider.close()
        """
        if self._genai is not None:
            try:
                # Google SDK doesn't have explicit close method
                # Just clear the reference
                self._genai = None
                logger.debug("Google AI client resources released")
            except Exception as e:
                logger.warning(f"Error releasing Google AI resources: {e}")

    def __enter__(self) -> "GoogleProvider":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
