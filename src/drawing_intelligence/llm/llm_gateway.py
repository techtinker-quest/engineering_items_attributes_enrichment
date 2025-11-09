"""
LLM Gateway - unified interface to multiple LLM providers with budget control.

This module provides a centralized gateway for managing LLM API calls across
multiple providers (OpenAI, Anthropic, Google) with features including:
- Automatic provider selection and fallback
- Budget enforcement and cost tracking
- Use-case specific model selection
- Prompt template management
- Retry logic with exponential backoff and jitter

Typical usage example:

    config = LLMConfig(
        enabled=True,
        primary_provider=ProviderConfig(
            name="openai",
            api_key_env="OPENAI_API_KEY"
        )
    )
    budget_controller = BudgetController(daily_budget_usd=50.0)

    with LLMGateway(config, budget_controller) as gateway:
        response = gateway.call_llm(
            prompt="Extract part number",
            image=drawing_image,
            use_case=UseCaseType.ENTITY_EXTRACTION,
            drawing_id="DWG-001"
        )
"""

import json
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..models.data_structures import (
    DrawingAssessment,
    Entity,
    EntityType,
    OCRVerification,
)
from ..models.model_registry import ModelRegistry, ModelSpec
from ..utils.error_handlers import (
    BudgetExceededException,
    ConfigurationError,
    LLMAPIError,
)
from ..utils.file_utils import generate_unique_id
from ..utils.geometry_utils import BoundingBox
from ..utils.image_utils import encode_image_base64
from .budget_controller import BudgetController, UseCaseType
from .cost_tracker import CostTracker
from .prompt_library import PromptLibrary
from .providers.base_provider import BaseLLMProvider, LLMResponse, TokenUsage
from .providers.anthropic_provider import AnthropicProvider
from .providers.google_provider import GoogleProvider
from .providers.openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)


# Configuration defaults
class LLMDefaults:
    """Default configuration values for LLM operations."""

    MAX_TOKENS: int = 2000
    TEMPERATURE: float = 0.0
    TOKEN_ESTIMATION_MULTIPLIER: float = 1.33  # Words to tokens (4/3)
    CONSERVATIVE_OUTPUT_TOKENS: int = 1000
    ENTITY_CONFIDENCE: float = 0.9
    OCR_CORRECTED_CONFIDENCE: float = 0.95
    OCR_UNCHANGED_CONFIDENCE: float = 0.85
    QUALITY_SCORE_DIVISOR: float = 10.0  # LLM returns 0-10, normalize to 0-1
    IMAGE_JPEG_QUALITY: int = 95
    MAX_IMAGE_DIMENSION: int = 2048  # Resize limit for cost optimization


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider.

    Attributes:
        name: Provider name ('openai', 'anthropic', 'google').
        api_key_env: Environment variable name containing API key.
        base_url: Optional custom base URL for proxies/Azure (OpenAI only).
        timeout: Optional custom timeout in seconds.
        organization: Optional organization ID (OpenAI only).
    """

    name: str
    api_key_env: str
    base_url: Optional[str] = None
    timeout: Optional[int] = None
    organization: Optional[str] = None


@dataclass
class LLMConfig:
    """Configuration for LLM gateway initialization.

    Attributes:
        enabled: Whether LLM integration is enabled.
        primary_provider: Primary provider configuration.
        fallback_provider: Optional fallback provider config.
        timeout_seconds: Default API request timeout.
        max_retries: Maximum retry attempts for transient errors.
        defaults: Default values for LLM operations.
    """

    enabled: bool
    primary_provider: ProviderConfig
    fallback_provider: Optional[ProviderConfig] = None
    timeout_seconds: int = 30
    max_retries: int = 3
    defaults: LLMDefaults = field(default_factory=LLMDefaults)


class ProviderFactory:
    """Factory for creating LLM provider instances."""

    _providers: Dict[
        str, Callable[[str, ModelRegistry, Dict[str, Any]], BaseLLMProvider]
    ] = {
        "openai": lambda api_key, registry, config: OpenAIProvider(
            api_key=api_key,
            model_registry=registry,
            base_url=config.get("base_url"),
            timeout=config.get("timeout"),
        ),
        "anthropic": lambda api_key, registry, config: AnthropicProvider(
            api_key=api_key,
            model_registry=registry,
        ),
        "google": lambda api_key, registry, config: GoogleProvider(
            api_key=api_key,
            model_registry=registry,
        ),
    }

    @classmethod
    def create(
        cls,
        name: str,
        api_key: str,
        model_registry: ModelRegistry,
        config: Dict[str, Any],
    ) -> BaseLLMProvider:
        """Create a provider instance by name.

        Args:
            name: Provider name (case-insensitive).
            api_key: API authentication key.
            model_registry: Model registry instance.
            config: Provider-specific configuration.

        Returns:
            Instantiated provider object.

        Raises:
            ValueError: If provider name is unknown.
        """
        provider_key = name.lower()
        if provider_key not in cls._providers:
            raise ValueError(
                f"Unknown provider: {name}. "
                f"Available: {', '.join(cls._providers.keys())}"
            )

        return cls._providers[provider_key](api_key, model_registry, config)


class LLMGateway:
    """Unified interface to multiple LLM providers with budget control.

    This gateway manages LLM API calls across multiple providers, handling
    provider initialization, automatic fallback, budget enforcement, cost
    tracking, and retry logic. Supports use-case specific model selection
    via ModelRegistry and BudgetController.

    The gateway implements a hybrid approach where:
    - 90-95% of drawings use open-source processing only
    - 5-10% of drawings flagged as low-confidence use LLM enhancement
    - All LLM usage is tracked against daily budget limits

    Attributes:
        config: LLMConfig instance defining provider settings and timeouts.
        budget_controller: BudgetController for enforcing daily spend limits.
        cost_tracker: Optional CostTracker for detailed cost analytics.
        prompt_library: PromptLibrary for managing reusable prompt templates.
        model_registry: ModelRegistry instance for model specifications.
    """

    def __init__(
        self,
        config: LLMConfig,
        budget_controller: BudgetController,
        cost_tracker: Optional[CostTracker] = None,
        prompt_library: Optional[PromptLibrary] = None,
    ) -> None:
        """Initialize LLM gateway with configuration and dependencies.

        Args:
            config: LLM configuration including provider settings.
            budget_controller: Budget controller for cost enforcement.
            cost_tracker: Optional cost tracker for detailed analytics.
            prompt_library: Optional prompt library for template management.

        Raises:
            ConfigurationError: If primary provider configuration is invalid.
        """
        self.config = config
        self.budget_controller = budget_controller
        self.cost_tracker = cost_tracker
        self.prompt_library = prompt_library or PromptLibrary()
        self.model_registry = ModelRegistry()

        self._providers: Dict[str, BaseLLMProvider] = {}
        self._provider_configs: Dict[str, ProviderConfig] = {}
        self._initialize_providers()

        logger.info("LLMGateway initialized successfully")

    def _initialize_providers(self) -> None:
        """Initialize configured LLM providers from environment variables.

        Reads API keys from environment and instantiates provider objects.
        Stores both provider instances and their configurations for fallback.

        Raises:
            ConfigurationError: If primary provider is missing required config.
        """
        # Initialize primary provider
        self._init_provider(self.config.primary_provider, is_primary=True)

        # Initialize optional fallback provider
        if self.config.fallback_provider:
            self._init_provider(self.config.fallback_provider, is_primary=False)

    def _init_provider(self, provider_config: ProviderConfig, is_primary: bool) -> None:
        """Initialize a single provider from configuration.

        Args:
            provider_config: Provider configuration object.
            is_primary: Whether this is the primary provider.

        Raises:
            ConfigurationError: If primary provider initialization fails.
        """
        provider_name = provider_config.name.lower()
        api_key = os.getenv(provider_config.api_key_env, "").strip()

        if not provider_name:
            error_msg = "Provider name not specified"
            if is_primary:
                raise ConfigurationError(error_msg, config_key="primary_provider.name")
            logger.warning(f"Fallback provider: {error_msg}")
            return

        if not api_key:
            error_msg = f"API key not found: {provider_config.api_key_env}"
            if is_primary and self.config.enabled:
                raise ConfigurationError(
                    error_msg, config_key=provider_config.api_key_env
                )
            logger.warning(f"{provider_name}: {error_msg}")
            return

        try:
            provider = ProviderFactory.create(
                name=provider_name,
                api_key=api_key,
                model_registry=self.model_registry,
                config={
                    "base_url": provider_config.base_url,
                    "timeout": provider_config.timeout or self.config.timeout_seconds,
                    "organization": provider_config.organization,
                },
            )
            self._providers[provider_name] = provider
            self._provider_configs[provider_name] = provider_config
            logger.info(
                f"Initialized {'primary' if is_primary else 'fallback'} "
                f"provider: {provider_name}"
            )
        except Exception as e:
            error_msg = f"Failed to create provider {provider_name}: {e}"
            if is_primary:
                raise ConfigurationError(error_msg, config_key="primary_provider")
            logger.exception(error_msg)

    def call_llm(
        self,
        prompt: str,
        image: Optional[np.ndarray],
        use_case: UseCaseType,
        drawing_id: str,
        override_model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Execute LLM API call with automatic provider and model selection.

        This is the primary method for all LLM interactions. Handles:
        - Input validation
        - Pre-call budget validation
        - Image format conversion and optimization
        - Model selection via BudgetController or override
        - Retry logic with exponential backoff and jitter
        - Automatic fallback to secondary provider
        - Cost tracking and budget updates

        Args:
            prompt: Text prompt describing the task.
            image: Optional numpy array image in BGR format.
            use_case: Type of operation for budget allocation.
            drawing_id: Unique identifier for cost tracking.
            override_model: Optional model ID to force specific model.
            max_tokens: Optional max tokens override.
            temperature: Optional temperature override.

        Returns:
            LLMResponse object containing content, tokens, and cost.

        Raises:
            BudgetExceededException: If budget check fails.
            LLMAPIError: If call fails after all retries.
            ValueError: If inputs are invalid.
        """
        # Validate inputs
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if not isinstance(use_case, UseCaseType):
            raise ValueError(f"Invalid use_case type: {type(use_case)}")

        # Prepare image if provided
        image_bytes: Optional[bytes] = None
        image_count: int = 0
        if image is not None:
            image_bytes = self._prepare_image(image)
            image_count = 1

        # Get model specification
        model_spec = self._get_model_spec(
            use_case=use_case,
            drawing_id=drawing_id,
            override_model=override_model,
            estimated_input_tokens=self._estimate_tokens(prompt),
            estimated_output_tokens=self.config.defaults.CONSERVATIVE_OUTPUT_TOKENS,
            image_count=image_count,
        )

        # Try primary provider first, then fallback
        providers_to_try = [
            (model_spec.provider.value.lower(), True),  # (provider_name, is_primary)
        ]

        # Add fallback if available and different from primary
        if self.config.fallback_provider:
            fallback_name = self.config.fallback_provider.name.lower()
            if fallback_name != model_spec.provider.value.lower():
                providers_to_try.append((fallback_name, False))

        last_error: Optional[Exception] = None

        for provider_name, is_primary in providers_to_try:
            try:
                response = self._call_with_retry(
                    provider_name=provider_name,
                    prompt=prompt,
                    image_bytes=image_bytes,
                    model_spec=model_spec,
                    max_tokens=max_tokens or self.config.defaults.MAX_TOKENS,
                    temperature=(
                        temperature
                        if temperature is not None
                        else self.config.defaults.TEMPERATURE
                    ),
                )

                if response and response.success:
                    # Track cost
                    self._track_cost(
                        response=response,
                        model_spec=model_spec,
                        drawing_id=drawing_id,
                        use_case=use_case,
                    )
                    return response

            except Exception as e:
                last_error = e
                logger.warning(
                    f"{'Primary' if is_primary else 'Fallback'} provider "
                    f"{provider_name} failed: {e}"
                )
                # Continue to fallback if available

        # All providers failed
        error_msg = f"LLM call failed for all providers after retries"
        if last_error:
            error_msg += f": {last_error}"
        raise LLMAPIError(
            error_msg,
            drawing_id=drawing_id,
            provider=model_spec.provider.value,
        )

    def _call_with_retry(
        self,
        provider_name: str,
        prompt: str,
        image_bytes: Optional[bytes],
        model_spec: ModelSpec,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Execute LLM call with retry logic.

        Args:
            provider_name: Name of provider to use.
            prompt: Text prompt.
            image_bytes: Optional image bytes.
            model_spec: Model specification.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            LLMResponse if successful.

        Raises:
            LLMAPIError: If all retries exhausted.
        """
        provider = self._providers.get(provider_name)
        if not provider:
            raise LLMAPIError(
                f"Provider not available: {provider_name}",
                provider=provider_name,
            )

        last_error: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = provider.call(
                    prompt=prompt,
                    image=image_bytes,
                    model=model_spec.model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=None,
                )

                if response and response.success:
                    logger.info(
                        f"LLM call successful (attempt {attempt}): "
                        f"{model_spec.canonical_name}"
                    )
                    return response

            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM call attempt {attempt}/{self.config.max_retries} "
                    f"failed: {e}"
                )

                # Only retry on transient errors
                if attempt < self.config.max_retries and self._is_retryable_error(e):
                    backoff_seconds = self._calculate_backoff(attempt)
                    logger.info(f"Retrying in {backoff_seconds:.2f}s...")
                    time.sleep(backoff_seconds)
                else:
                    break

        raise LLMAPIError(
            f"Failed after {self.config.max_retries} attempts: {last_error}",
            provider=provider_name,
        )

    def _get_model_spec(
        self,
        use_case: UseCaseType,
        drawing_id: str,
        override_model: Optional[str],
        estimated_input_tokens: int,
        estimated_output_tokens: int,
        image_count: int,
    ) -> ModelSpec:
        """Get model specification with budget validation.

        Args:
            use_case: Use case for model selection.
            drawing_id: Drawing ID for tracking.
            override_model: Optional model override.
            estimated_input_tokens: Estimated input token count.
            estimated_output_tokens: Estimated output token count.
            image_count: Number of images.

        Returns:
            ModelSpec instance.

        Raises:
            BudgetExceededException: If budget check fails.
            ValueError: If override_model is invalid.
        """
        # Pre-flight budget check
        allowed, reason, model_spec = self.budget_controller.pre_call_check(
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            image_count=image_count,
            use_case=use_case,
            drawing_id=drawing_id,
        )

        if not allowed:
            current_cost = self.budget_controller.get_total_cost()
            raise BudgetExceededException(
                message=reason,
                current_cost=current_cost,
                budget_limit=self.budget_controller.daily_budget_usd,
                drawing_id=drawing_id,
            )

        # Use override or budget controller's selection
        if override_model:
            model_spec = self.model_registry.get_model(override_model)
            if model_spec is None:
                raise ValueError(
                    f"Override model not found in registry: {override_model}"
                )

        return model_spec

    def _track_cost(
        self,
        response: LLMResponse,
        model_spec: ModelSpec,
        drawing_id: str,
        use_case: UseCaseType,
    ) -> None:
        """Track LLM call cost in budget controller and cost tracker.

        Args:
            response: LLM response object.
            model_spec: Model specification used.
            drawing_id: Drawing ID for tracking.
            use_case: Use case type.
        """
        actual_cost = model_spec.calculate_cost(
            response.tokens_used.input_tokens,
            response.tokens_used.output_tokens,
            response.tokens_used.image_count,
        )

        # Track in budget controller
        self.budget_controller.track_call(
            provider=model_spec.provider.value,
            model_id=model_spec.model_id,
            input_tokens=response.tokens_used.input_tokens,
            output_tokens=response.tokens_used.output_tokens,
            image_count=response.tokens_used.image_count,
            drawing_id=drawing_id,
            use_case=use_case,
        )

        # Track in cost tracker if available
        if self.cost_tracker:
            self.cost_tracker.track_call(
                provider=model_spec.provider.value,
                model=model_spec.model_id,
                tokens=response.tokens_used,
                cost=actual_cost,
                drawing_id=drawing_id,
                use_case=use_case.value,
            )

        logger.info(
            f"Cost tracked: {model_spec.canonical_name} - ${actual_cost:.4f} "
            f"({response.tokens_used.input_tokens}→"
            f"{response.tokens_used.output_tokens} tokens)"
        )

    def assess_drawing_quality(
        self, image: np.ndarray, drawing_id: str, file_name: str
    ) -> DrawingAssessment:
        """Evaluate drawing quality for pipeline routing decision.

        Uses LLM vision capabilities to assess drawing quality and recommend
        appropriate processing pipeline (baseline vs enhanced).

        Args:
            image: Drawing image as numpy array (BGR format).
            drawing_id: Unique drawing identifier.
            file_name: Original filename for context.

        Returns:
            DrawingAssessment with quality scores and pipeline recommendation.

        Note:
            Uses structured JSON output for reliable parsing. Falls back to
            default scores (0.5) if parsing fails.
        """
        template = self.prompt_library.get_prompt("drawing_assessment")
        prompt = self.prompt_library.render_prompt(template, {"file_name": file_name})

        # Request JSON format for structured output
        prompt += "\n\nRespond ONLY with valid JSON in this format:\n"
        prompt += json.dumps(
            {
                "overall_quality": 7.5,
                "complexity": 6.0,
                "text_clarity": 8.0,
                "shape_clarity": 7.0,
                "recommended_pipeline": "baseline",
                "reasoning": "explanation here",
            },
            indent=2,
        )

        response = self.call_llm(
            prompt=prompt,
            image=image,
            use_case=UseCaseType.DRAWING_ASSESSMENT,
            drawing_id=drawing_id,
        )

        # Parse structured JSON response
        try:
            data = json.loads(response.content.strip())
            return DrawingAssessment(
                overall_quality=data.get("overall_quality", 5.0)
                / self.config.defaults.QUALITY_SCORE_DIVISOR,
                complexity_score=data.get("complexity", 5.0)
                / self.config.defaults.QUALITY_SCORE_DIVISOR,
                text_clarity=data.get("text_clarity", 5.0)
                / self.config.defaults.QUALITY_SCORE_DIVISOR,
                shape_clarity=data.get("shape_clarity", 5.0)
                / self.config.defaults.QUALITY_SCORE_DIVISOR,
                recommended_pipeline=data.get("recommended_pipeline", "baseline"),
                reasoning=data.get("reasoning", response.content),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(
                f"Failed to parse drawing assessment JSON for {drawing_id}: {e}. "
                f"Response: {response.content[:200]}"
            )
            # Return defaults
            return DrawingAssessment(
                overall_quality=0.5,
                complexity_score=0.5,
                text_clarity=0.5,
                shape_clarity=0.5,
                recommended_pipeline="baseline",
                reasoning="Parse error: using defaults",
            )

    def verify_ocr(
        self,
        image_crop: np.ndarray,
        ocr_text: str,
        drawing_id: str,
        region_type: str = "text",
    ) -> OCRVerification:
        """Verify and correct OCR text using LLM vision capabilities.

        Args:
            image_crop: Cropped image region (BGR format).
            ocr_text: Original OCR extracted text.
            drawing_id: Drawing identifier.
            region_type: Region type for context.

        Returns:
            OCRVerification with corrected text and confidence.
        """
        template = self.prompt_library.get_prompt("ocr_verification")
        prompt = self.prompt_library.render_prompt(
            template, {"ocr_text": ocr_text, "region_type": region_type}
        )

        # Request structured JSON
        prompt += "\n\nRespond ONLY with valid JSON:\n"
        prompt += json.dumps(
            {
                "corrected_text": "example",
                "corrections": ["change 1", "change 2"],
            },
            indent=2,
        )

        response = self.call_llm(
            prompt=prompt,
            image=image_crop,
            use_case=UseCaseType.OCR_VERIFICATION,
            drawing_id=drawing_id,
        )

        try:
            data = json.loads(response.content.strip())
            corrected_text = data.get("corrected_text", ocr_text)
            corrections = data.get("corrections", [])

            return OCRVerification(
                corrected_text=corrected_text,
                corrections_made=corrections if isinstance(corrections, list) else [],
                confidence=(
                    self.config.defaults.OCR_CORRECTED_CONFIDENCE
                    if corrected_text != ocr_text
                    else self.config.defaults.OCR_UNCHANGED_CONFIDENCE
                ),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(
                f"Failed to parse OCR verification JSON for {drawing_id}: {e}. "
                f"Response: {response.content[:200]}"
            )
            return OCRVerification(
                corrected_text=ocr_text,
                corrections_made=[],
                confidence=self.config.defaults.OCR_UNCHANGED_CONFIDENCE,
            )

    def extract_entities_llm(
        self,
        text: str,
        context: str,
        entity_types: List[str],
        drawing_id: str = "unknown",
    ) -> List[Entity]:
        """Extract structured entities from text using LLM reasoning.

        Args:
            text: Text to extract entities from.
            context: Additional context for disambiguation.
            entity_types: List of entity type names to extract.
            drawing_id: Drawing identifier for tracking.

        Returns:
            List of Entity objects.

        Raises:
            LLMAPIError: If LLM call fails.
        """
        template = self.prompt_library.get_prompt("entity_extraction")
        prompt = self.prompt_library.render_prompt(
            template,
            {"text": text, "context": context, "entity_types": ", ".join(entity_types)},
        )

        # Request structured JSON
        prompt += "\n\nRespond ONLY with valid JSON:\n"
        prompt += json.dumps(
            {
                "entities": [
                    {"type": "PART_NUMBER", "value": "ABC-123", "confidence": 0.95}
                ]
            },
            indent=2,
        )

        response = self.call_llm(
            prompt=prompt,
            image=None,
            use_case=UseCaseType.ENTITY_EXTRACTION,
            drawing_id=drawing_id,
        )

        entities: List[Entity] = []
        try:
            data = json.loads(response.content.strip())

            for entity_data in data.get("entities", []):
                entity = Entity(
                    entity_id=generate_unique_id("ENT"),
                    entity_type=EntityType[entity_data["type"]],
                    value=entity_data["value"],
                    original_text=entity_data["value"],
                    normalized_value={"raw": entity_data["value"]},
                    confidence=entity_data.get(
                        "confidence", self.config.defaults.ENTITY_CONFIDENCE
                    ),
                    extraction_method="llm",
                    source_text_id="",
                    bbox=None,
                )
                entities.append(entity)

        except (json.JSONDecodeError, KeyError) as e:
            logger.exception(
                f"Failed to parse entity extraction JSON for {drawing_id}: {e}. "
                f"Response: {response.content[:200]}"
            )

        return entities

    def _prepare_image(self, image: np.ndarray) -> bytes:
        """Convert and optimize numpy image for API transmission.

        Performs validation, optional resizing for cost optimization,
        and JPEG encoding.

        Args:
            image: Numpy array image in BGR format.

        Returns:
            JPEG-encoded image bytes.

        Raises:
            ValueError: If image is invalid or encoding fails.
        """
        if image is None or image.size == 0:
            raise ValueError("Image is empty or None")

        if len(image.shape) not in [2, 3]:
            raise ValueError(
                f"Invalid image shape {image.shape}. Expected (H, W) or (H, W, C)"
            )

        if image.dtype != np.uint8:
            logger.warning(
                f"Image dtype is {image.dtype}, expected uint8. Converting..."
            )
            image = image.astype(np.uint8)

        # Optimize: resize if too large
        height, width = image.shape[:2]
        max_dim = self.config.defaults.MAX_IMAGE_DIMENSION

        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
            logger.debug(
                f"Resized image from {width}x{height} to {new_width}x{new_height}"
            )

        # Encode as JPEG
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY,
            self.config.defaults.IMAGE_JPEG_QUALITY,
        ]
        success, encoded = cv2.imencode(".jpg", image, encode_params)

        if not success:
            raise ValueError(
                f"Failed to encode image with shape {image.shape} and dtype {image.dtype}"
            )

        return encoded.tobytes()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text.

        Args:
            text: Input text.

        Returns:
            Estimated token count.

        Note:
            Uses simple word-based estimation. For production, use
            provider-specific tokenizers (e.g., tiktoken for OpenAI).
        """
        return int(len(text.split()) * self.config.defaults.TOKEN_ESTIMATION_MULTIPLIER)

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate backoff time with exponential backoff and jitter.

        Args:
            attempt: Current attempt number (1-indexed).

        Returns:
            Backoff time in seconds.
        """
        base_delay = 2**attempt
        jitter = random.uniform(0, 0.3 * base_delay)  # 0-30% jitter
        return base_delay + jitter

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is transient and should be retried.

        Args:
            error: Exception raised during LLM API call.

        Returns:
            True if error is likely transient.
        """
        error_str = str(error).lower()

        # Permanent errors - do not retry
        permanent_patterns = [
            "api key",
            "authentication",
            "401",
            "403",
            "invalid",
            "malformed",
        ]

        for pattern in permanent_patterns:
            if pattern in error_str:
                return False

        # Transient errors - retry
        transient_patterns = [
            "rate limit",
            "timeout",
            "503",
            "502",
            "500",
            "connection",
            "temporary",
            "overloaded",
            "too many requests",
        ]

        for pattern in transient_patterns:
            if pattern in error_str:
                return True

        # Default: do not retry unknown errors
        return False

    def close(self) -> None:
        """Close all provider connections and cleanup resources.

        Call this method when the gateway is no longer needed to release
        network connections and other resources held by providers.
        """
        for provider_name, provider in list(self._providers.items()):
            try:
                if hasattr(provider, "close"):
                    provider.close()
                logger.debug(f"Closed provider: {provider_name}")
            except Exception as e:
                logger.exception(f"Error closing provider {provider_name}: {e}")

        self._providers.clear()
        self._provider_configs.clear()

    def __enter__(self) -> "LLMGateway":
        """Context manager entry.

        Returns:
            Self for use in 'with' statements.
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
        """
        self.close()
