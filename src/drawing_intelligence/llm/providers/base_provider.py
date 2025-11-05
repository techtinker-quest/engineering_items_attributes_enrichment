"""
Base provider interface for LLM integrations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """
    Standardized LLM response.

    Attributes:
        content: Response text content
        tokens_used: Token usage information
        model_used: Model identifier
        provider: Provider name
        success: Whether call succeeded
        error_message: Optional error message
    """

    content: str
    tokens_used: "TokenUsage"
    model_used: str
    provider: str
    success: bool
    error_message: Optional[str] = None


@dataclass
class TokenUsage:
    """
    Token usage tracking.

    Attributes:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        image_count: Number of images in request
    """

    input_tokens: int
    output_tokens: int
    image_count: int = 0


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All provider implementations must inherit from this class.
    """

    @abstractmethod
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
        Make API call to LLM provider.

        Args:
            prompt: Text prompt
            image: Optional image bytes
            model: Model identifier
            max_tokens: Maximum response tokens
            temperature: Temperature (0.0-1.0)
            response_format: Optional format specification (e.g., JSON mode)

        Returns:
            LLMResponse with content and metadata

        Raises:
            Exception: If API call fails
        """
        pass

    @abstractmethod
    def get_available_models(self) -> list:
        """
        Get list of available models for this provider.

        Returns:
            List of model identifiers
        """
        pass

    @abstractmethod
    def validate_credentials(self) -> bool:
        """
        Validate API credentials.

        Returns:
            True if credentials are valid
        """
        pass
