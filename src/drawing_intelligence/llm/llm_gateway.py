"""
LLM Gateway - unified interface to multiple LLM providers.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum
import numpy as np

from .providers.base_provider import LLMProvider, LLMResponse, TokenUsage
from .providers.openai_provider import OpenAIProvider
from .providers.anthropic_provider import AnthropicProvider
from .providers.google_provider import GoogleProvider
from .budget_controller import BudgetController, UseCaseType
from .cost_tracker import CostTracker
from .prompt_library import PromptLibrary
from ..models.model_registry import ModelRegistry
from ..models.data_structures import DrawingAssessment, OCRVerification, Entity
from ..utils.image_utils import encode_image_base64
from ..utils.error_handlers import LLMAPIError, BudgetExceededException

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """
    Configuration for LLM gateway.

    Attributes:
        enabled: Enable LLM integration
        primary_provider: Primary provider config
        fallback_provider: Optional fallback provider
        timeout_seconds: API timeout
        max_retries: Maximum retry attempts
    """

    enabled: bool
    primary_provider: Dict[str, Any]
    fallback_provider: Optional[Dict[str, Any]] = None
    timeout_seconds: int = 30
    max_retries: int = 3


class LLMGateway:
    """
    Unified interface to multiple LLM providers.

    Features:
    - Multi-provider support (OpenAI, Anthropic, Google)
    - Automatic fallback
    - Cost tracking
    - Budget enforcement
    - Prompt management
    - Use-case specific model selection
    """

    def __init__(
        self,
        config: LLMConfig,
        budget_controller: BudgetController,
        cost_tracker: Optional[CostTracker] = None,
        prompt_library: Optional[PromptLibrary] = None,
    ):
        """
        Initialize LLM gateway.

        Args:
            config: LLM configuration
            budget_controller: Budget controller
            cost_tracker: Optional cost tracker
            prompt_library: Optional prompt library
        """
        self.config = config
        self.budget_controller = budget_controller
        self.cost_tracker = cost_tracker
        self.prompt_library = prompt_library or PromptLibrary()

        # Initialize providers
        self._providers: Dict[str, LLMProvider] = {}
        self._initialize_providers()

        logger.info("LLMGateway initialized")

    def _initialize_providers(self):
        """Initialize configured LLM providers."""
        # Primary provider
        primary_config = self.config.primary_provider
        provider_name = primary_config["name"].lower()
        api_key = os.getenv(primary_config["api_key_env"])

        if not api_key:
            logger.warning(
                f"API key not found for {provider_name}: "
                f"{primary_config['api_key_env']}"
            )
        else:
            provider = self._create_provider(provider_name, api_key, primary_config)
            if provider:
                self._providers[provider_name] = provider
                logger.info(f"Initialized primary provider: {provider_name}")

        # Fallback provider
        if self.config.fallback_provider:
            fallback_config = self.config.fallback_provider
            provider_name = fallback_config["name"].lower()
            api_key = os.getenv(fallback_config["api_key_env"])

            if api_key:
                provider = self._create_provider(
                    provider_name, api_key, fallback_config
                )
                if provider:
                    self._providers[provider_name] = provider
                    logger.info(f"Initialized fallback provider: {provider_name}")

    def _create_provider(
        self, name: str, api_key: str, config: Dict
    ) -> Optional[LLMProvider]:
        """Create provider instance."""
        try:
            if name == "openai":
                return OpenAIProvider(api_key=api_key, base_url=config.get("base_url"))
            elif name == "anthropic":
                return AnthropicProvider(api_key=api_key)
            elif name == "google":
                return GoogleProvider(api_key=api_key)
            else:
                logger.error(f"Unknown provider: {name}")
                return None
        except Exception as e:
            logger.error(f"Failed to create provider {name}: {e}")
            return None

    def call_llm(
        self,
        prompt: str,
        image: Optional[np.ndarray],
        use_case: UseCaseType,
        drawing_id: str,
        override_model: Optional[str] = None,
    ) -> LLMResponse:
        """
        Call LLM with automatic provider/model selection.

        Args:
            prompt: Text prompt
            image: Optional numpy image
            use_case: Use case type
            drawing_id: Drawing identifier for tracking
            override_model: Optional model override

        Returns:
            LLMResponse

        Raises:
            LLMAPIError: If call fails
            BudgetExceededException: If budget exceeded
        """
        # Convert image to bytes if provided
        image_bytes = None
        image_count = 0
        if image is not None:
            image_bytes = self._prepare_image(image)
            image_count = 1

        # Estimate tokens for budget check
        estimated_input = len(prompt.split()) * 4 // 3
        estimated_output = 1000  # Conservative estimate

        # Pre-flight budget check
        allowed, reason, model_spec = self.budget_controller.pre_call_check(
            estimated_input_tokens=estimated_input,
            estimated_output_tokens=estimated_output,
            image_count=image_count,
            use_case=use_case,
            drawing_id=drawing_id,
        )

        if not allowed:
            raise BudgetExceededException(
                message=reason,
                current_cost=self.budget_controller._get_daily_spend(),
                budget_limit=self.budget_controller.daily_budget_usd,
                drawing_id=drawing_id,
            )

        # Use provided model or from budget controller
        if override_model:
            model_spec = ModelRegistry.get_model(override_model)

        # Get provider
        provider = self._get_provider(model_spec.provider.value.lower())

        if not provider:
            raise LLMAPIError(
                f"Provider not available: {model_spec.provider.value}",
                drawing_id=drawing_id,
                provider=model_spec.provider.value,
            )

        # Make call with retry logic
        response = None
        last_error = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = provider.call(
                    prompt=prompt,
                    image=image_bytes,
                    model=model_spec.model_id,
                    max_tokens=2000,  # Default
                    temperature=0.0,  # Deterministic
                    response_format=None,
                )

                if response.success:
                    break

            except Exception as e:
                last_error = e
                logger.warning(f"LLM call attempt {attempt} failed: {e}")

                if attempt < self.config.max_retries:
                    import time

                    time.sleep(2**attempt)  # Exponential backoff

        if not response or not response.success:
            raise LLMAPIError(
                f"LLM call failed after {self.config.max_retries} attempts: {last_error}",
                drawing_id=drawing_id,
                provider=model_spec.provider.value,
            )

        # Calculate actual cost
        actual_cost = model_spec.calculate_cost(
            response.tokens_used.input_tokens,
            response.tokens_used.output_tokens,
            response.tokens_used.image_count,
        )

        # Track call
        self.budget_controller.track_call(
            provider=model_spec.provider.value,
            model_id=model_spec.model_id,
            input_tokens=response.tokens_used.input_tokens,
            output_tokens=response.tokens_used.output_tokens,
            image_count=response.tokens_used.image_count,
            drawing_id=drawing_id,
            use_case=use_case,
        )

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
            f"LLM call successful: {model_spec.canonical_name} - "
            f"${actual_cost:.4f} ({response.tokens_used.input_tokens}→"
            f"{response.tokens_used.output_tokens} tokens)"
        )

        return response

    def assess_drawing_quality(
        self, image: np.ndarray, drawing_id: str, file_name: str
    ) -> DrawingAssessment:
        """
        Use case: Evaluate drawing quality for routing.

        Args:
            image: Drawing image
            drawing_id: Drawing identifier
            file_name: Original filename

        Returns:
            DrawingAssessment
        """
        # Get prompt
        template = self.prompt_library.get_prompt("drawing_assessment")
        prompt = self.prompt_library.render_prompt(template, {"file_name": file_name})

        # Call LLM
        response = self.call_llm(
            prompt=prompt,
            image=image,
            use_case=UseCaseType.DRAWING_ASSESSMENT,
            drawing_id=drawing_id,
        )

        # Parse response (simplified)
        content = response.content

        # Extract scores
        import re

        overall_match = re.search(r"Overall Quality:\s*([\d.]+)", content)
        complexity_match = re.search(r"Complexity:\s*([\d.]+)", content)
        text_match = re.search(r"Text Clarity:\s*([\d.]+)", content)
        shape_match = re.search(r"Shape Clarity:\s*([\d.]+)", content)
        pipeline_match = re.search(r"Recommended Pipeline:\s*(\w+)", content)

        return DrawingAssessment(
            overall_quality=(
                float(overall_match.group(1)) / 10 if overall_match else 0.5
            ),
            complexity_score=(
                float(complexity_match.group(1)) / 10 if complexity_match else 0.5
            ),
            text_clarity=float(text_match.group(1)) / 10 if text_match else 0.5,
            shape_clarity=float(shape_match.group(1)) / 10 if shape_match else 0.5,
            recommended_pipeline=(
                pipeline_match.group(1) if pipeline_match else "baseline"
            ),
            reasoning=content,
        )

    def verify_ocr(
        self,
        image_crop: np.ndarray,
        ocr_text: str,
        drawing_id: str,
        region_type: str = "text",
    ) -> OCRVerification:
        """
        Use case: Verify/correct OCR text.

        Args:
            image_crop: Cropped image region
            ocr_text: Original OCR text
            drawing_id: Drawing identifier
            region_type: Type of region

        Returns:
            OCRVerification
        """
        # Get prompt
        template = self.prompt_library.get_prompt("ocr_verification")
        prompt = self.prompt_library.render_prompt(
            template, {"ocr_text": ocr_text, "region_type": region_type}
        )

        # Call LLM
        response = self.call_llm(
            prompt=prompt,
            image=image_crop,
            use_case=UseCaseType.OCR_VERIFICATION,
            drawing_id=drawing_id,
        )

        # Parse response
        content = response.content

        import re

        corrected_match = re.search(
            r"Corrected Text:\s*(.+?)(?=Corrections:|$)", content, re.DOTALL
        )
        corrections_match = re.search(r"Corrections:\s*(.+)$", content, re.DOTALL)

        corrected_text = (
            corrected_match.group(1).strip() if corrected_match else ocr_text
        )
        corrections = corrections_match.group(1).strip() if corrections_match else ""

        return OCRVerification(
            corrected_text=corrected_text,
            corrections_made=[corrections] if corrections else [],
            confidence=0.95 if corrected_text != ocr_text else 0.85,
        )

    def extract_entities_llm(
        self,
        text: str,
        context: str,
        entity_types: List[str],
        drawing_id: str = "unknown",
    ) -> List[Entity]:
        """
        Use case: Extract entities using LLM.

        Args:
            text: Text to extract from
            context: Context information
            entity_types: List of entity types to extract
            drawing_id: Drawing identifier

        Returns:
            List of Entity objects
        """
        # Get prompt
        template = self.prompt_library.get_prompt("entity_extraction")
        prompt = self.prompt_library.render_prompt(
            template,
            {"text": text, "context": context, "entity_types": ", ".join(entity_types)},
        )

        # Call LLM
        response = self.call_llm(
            prompt=prompt,
            image=None,
            use_case=UseCaseType.ENTITY_EXTRACTION,
            drawing_id=drawing_id,
        )

        # Parse JSON response
        entities = []
        try:
            import json

            data = json.loads(response.content)

            from ..models.data_structures import EntityType
            from ..utils.file_utils import generate_unique_id
            from ..utils.geometry_utils import BoundingBox

            for entity_data in data.get("entities", []):
                entity = Entity(
                    entity_id=generate_unique_id("ENT"),
                    entity_type=EntityType[entity_data["type"]],
                    value=entity_data["value"],
                    original_text=entity_data["value"],
                    normalized_value={"raw": entity_data["value"]},
                    confidence=entity_data.get("confidence", 0.9),
                    extraction_method="llm",
                    source_text_id="",
                    bbox=None,
                )
                entities.append(entity)

        except Exception as e:
            logger.error(f"Failed to parse LLM entity response: {e}")

        return entities

    def _get_provider(self, provider_name: str) -> Optional[LLMProvider]:
        """Get provider by name."""
        return self._providers.get(provider_name.lower())

    def _prepare_image(self, image: np.ndarray) -> bytes:
        """Convert numpy image to bytes for API."""
        import cv2

        # Encode as JPEG
        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            raise ValueError("Failed to encode image")

        return encoded.tobytes()
