"""Configuration Validator Module.

This module provides comprehensive validation for system configuration settings.
It validates paths, LLM integration, model files, and database configuration
using a modular, profile-based approach.

Example:
    >>> from drawing_intelligence.cli.config_validator import ConfigValidator
    >>> validator = ConfigValidator(config, profile="strict")
    >>> result = validator.validate_all()
    >>> if not result:
    ...     print(f"Validation failed: {result.errors}")
"""

import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ..utils.config_loader import SystemConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Constants and Enums
# ============================================================================


class ValidationProfile(Enum):
    """Validation strictness profiles."""

    PERMISSIVE = "permissive"  # Warnings only
    STANDARD = "standard"  # Standard validation
    STRICT = "strict"  # Strict validation with all checks


class ProviderName(Enum):
    """Supported LLM provider names."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


# File size constants (MB)
MIN_MODEL_SIZE_MB = 1
MAX_MODEL_SIZE_MB = 200
MIN_DISK_SPACE_MB = 1000

# API key constants
MIN_API_KEY_LENGTH = 10
MIN_GOOGLE_KEY_LENGTH = 20

# Known insecure/fake API key patterns
FAKE_KEY_PATTERNS = [
    "sk-test",
    "sk-fake",
    "sk-demo",
    "your-api-key",
    "replace-me",
    "xxxxx",
]


# ============================================================================
# Custom Exceptions
# ============================================================================


class ConfigurationValidationError(Exception):
    """Base exception for configuration validation errors."""

    def __init__(self, message: str, config_key: Optional[str] = None):
        self.config_key = config_key
        super().__init__(message)


class CriticalConfigError(ConfigurationValidationError):
    """Raised for critical configuration errors that prevent startup."""

    pass


class PathConfigError(ConfigurationValidationError):
    """Raised for path-related configuration errors."""

    pass


class APIKeyError(ConfigurationValidationError):
    """Raised for API key validation errors."""

    pass


# ============================================================================
# Data Structures
# ============================================================================


@dataclass(frozen=True)
class ValidationResult:
    """Immutable container for configuration validation results.

    Attributes:
        is_valid: True if configuration passes all validation checks.
        errors: List of critical error messages that prevent system operation.
        warnings: List of non-critical warnings that don't prevent operation.
        info: List of informational messages about the validation process.
    """

    is_valid: bool
    errors: Tuple[str, ...] = field(default_factory=tuple)
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    info: Tuple[str, ...] = field(default_factory=tuple)

    def __bool__(self) -> bool:
        """Allow using ValidationResult in boolean context."""
        return self.is_valid

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "info": list(self.info),
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


# ============================================================================
# Base Validator
# ============================================================================


class BaseValidator(ABC):
    """Abstract base class for configuration validators."""

    def __init__(self, config: "SystemConfig", profile: ValidationProfile):
        self.config = config
        self.profile = profile

    @abstractmethod
    def validate(self) -> Tuple[List[str], List[str]]:
        """Validate configuration section.

        Returns:
            Tuple of (errors, warnings).
        """
        pass

    def _get_section(self, section_name: str, required: bool = True) -> Optional[any]:
        """Get configuration section with unified error handling.

        Args:
            section_name: Name of the configuration section.
            required: Whether the section is required.

        Returns:
            Configuration section object or None if not required and missing.

        Raises:
            CriticalConfigError: If required section is missing.
        """
        if not hasattr(self.config, section_name):
            if required:
                raise CriticalConfigError(
                    f"Missing required configuration section: '{section_name}'",
                    config_key=section_name,
                )
            return None
        return getattr(self.config, section_name)


# ============================================================================
# Path Validator
# ============================================================================


class PathValidator(BaseValidator):
    """Validates file system path configurations."""

    def validate(self) -> Tuple[List[str], List[str]]:
        """Validate all path configurations.

        Returns:
            Tuple of (errors, warnings).
        """
        errors: List[str] = []
        warnings: List[str] = []

        try:
            paths = self._get_section("paths")
        except CriticalConfigError as e:
            return [str(e)], []

        # Required paths that should be created if missing
        auto_create_paths = {
            "data_dir": "Data directory",
            "output_dir": "Output directory",
            "temp_dir": "Temporary directory",
            "log_dir": "Log directory",
        }

        for path_key, description in auto_create_paths.items():
            path_errors, path_warnings = self._validate_and_create_dir(
                paths, path_key, description, auto_create=True
            )
            errors.extend(path_errors)
            warnings.extend(path_warnings)

        # Required paths that must exist
        required_paths = {
            "models_dir": "Models directory",
        }

        for path_key, description in required_paths.items():
            path_errors, path_warnings = self._validate_and_create_dir(
                paths, path_key, description, auto_create=False
            )
            errors.extend(path_errors)
            warnings.extend(path_warnings)

        return errors, warnings

    def _validate_and_create_dir(
        self,
        paths_obj: any,
        path_key: str,
        description: str,
        auto_create: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """Validate a directory path and optionally create it.

        Args:
            paths_obj: Paths configuration object.
            path_key: Key name in paths configuration.
            description: Human-readable description of the path.
            auto_create: Whether to create the directory if missing.

        Returns:
            Tuple of (errors, warnings).
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not hasattr(paths_obj, path_key):
            errors.append(f"Missing 'paths.{path_key}' configuration")
            return errors, warnings

        path_value = getattr(paths_obj, path_key)
        path = Path(path_value)

        # Check for path traversal vulnerabilities
        try:
            path = path.resolve()
        except (OSError, RuntimeError) as e:
            errors.append(f"{description} path resolution failed '{path_value}': {e}")
            return errors, warnings

        # Check if path exists
        if not path.exists():
            if auto_create:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created {description.lower()}: {path}")
                except PermissionError:
                    errors.append(
                        f"Permission denied creating {description.lower()}: "
                        f"'{path}'"
                    )
                    return errors, warnings
                except OSError as e:
                    errors.append(f"Cannot create {description.lower()} '{path}': {e}")
                    return errors, warnings
            else:
                errors.append(f"{description} does not exist: '{path}'")
                return errors, warnings

        # Validate it's a directory
        if not path.is_dir():
            errors.append(f"{description} is not a directory: '{path}'")
            return errors, warnings

        # Check permissions
        if not os.access(path, os.R_OK):
            errors.append(f"No read permission for {description.lower()}: '{path}'")
        if not os.access(path, os.W_OK):
            errors.append(f"No write permission for {description.lower()}: '{path}'")

        # Check disk space (strict mode only)
        if self.profile == ValidationProfile.STRICT:
            try:
                stat = shutil.disk_usage(path)
                free_mb = stat.free / (1024 * 1024)
                if free_mb < MIN_DISK_SPACE_MB:
                    warnings.append(
                        f"Low disk space for {description.lower()}: "
                        f"{free_mb:.0f}MB available "
                        f"(recommended minimum: {MIN_DISK_SPACE_MB}MB)"
                    )
            except OSError as e:
                warnings.append(
                    f"Cannot check disk space for {description.lower()}: {e}"
                )

        return errors, warnings


# ============================================================================
# LLM Validator
# ============================================================================


class LLMValidator(BaseValidator):
    """Validates LLM integration configuration."""

    # Registry of provider-specific API key validators
    PROVIDER_VALIDATORS: Dict[ProviderName, Callable[[str], Tuple[bool, str]]] = {}

    @classmethod
    def register_provider(
        cls, provider: ProviderName
    ) -> Callable[[Callable], Callable]:
        """Decorator to register provider-specific validators."""

        def decorator(func: Callable[[str], Tuple[bool, str]]) -> Callable:
            cls.PROVIDER_VALIDATORS[provider] = func
            return func

        return decorator

    def validate(self) -> Tuple[List[str], List[str]]:
        """Validate LLM configuration and API keys.

        Returns:
            Tuple of (errors, warnings).
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Check if LLM config exists
        try:
            llm_config = self._get_section("llm_integration", required=False)
        except CriticalConfigError as e:
            warnings.append(str(e))
            return errors, warnings

        if llm_config is None:
            warnings.append("Missing 'llm_integration' configuration")
            return errors, warnings

        # Check if LLM is enabled
        if not getattr(llm_config, "enabled", False):
            warnings.append("LLM integration is disabled")
            return errors, warnings

        # Validate primary provider
        primary_errors, primary_warnings = self._validate_provider(
            llm_config, "primary_provider", required=True
        )
        errors.extend(primary_errors)
        warnings.extend(primary_warnings)

        # Validate fallback provider (optional)
        fallback_errors, fallback_warnings = self._validate_provider(
            llm_config, "fallback_provider", required=False
        )
        errors.extend(fallback_errors)
        warnings.extend(fallback_warnings)

        # Ensure fallback is different from primary
        if (
            hasattr(llm_config, "primary_provider")
            and hasattr(llm_config, "fallback_provider")
            and llm_config.fallback_provider
        ):
            primary_name = getattr(llm_config.primary_provider, "name", "").lower()
            fallback_name = getattr(llm_config.fallback_provider, "name", "").lower()
            if primary_name == fallback_name:
                warnings.append(
                    f"Fallback provider '{fallback_name}' is the same as "
                    f"primary provider. Consider using a different provider "
                    f"for better redundancy."
                )

        # Validate cost controls
        cost_errors, cost_warnings = self._validate_cost_controls(llm_config)
        errors.extend(cost_errors)
        warnings.extend(cost_warnings)

        return errors, warnings

    def _validate_provider(
        self, llm_config: any, provider_key: str, required: bool
    ) -> Tuple[List[str], List[str]]:
        """Validate a single provider configuration.

        Args:
            llm_config: LLM configuration object.
            provider_key: Key for provider (e.g., 'primary_provider').
            required: Whether this provider is required.

        Returns:
            Tuple of (errors, warnings).
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not hasattr(llm_config, provider_key):
            if required:
                errors.append(f"Missing '{provider_key}' configuration")
            return errors, warnings

        provider = getattr(llm_config, provider_key)
        if provider is None:
            if required:
                errors.append(f"'{provider_key}' is None")
            return errors, warnings

        # Check API key environment variable
        if not hasattr(provider, "api_key_env"):
            errors.append(f"Missing '{provider_key}.api_key_env' configuration")
            return errors, warnings

        api_key_env = provider.api_key_env
        api_key = os.getenv(api_key_env)

        if not api_key:
            error_msg = f"API key not found in environment variable: '{api_key_env}'"
            if required:
                errors.append(error_msg)
            else:
                warnings.append(error_msg)
            return errors, warnings

        # Validate key format
        provider_name = getattr(provider, "name", "unknown").lower()
        is_valid, error_msg = self._validate_api_key_format(provider_name, api_key)
        if not is_valid:
            errors.append(f"Invalid API key format for '{provider_name}': {error_msg}")

        # Security checks for potentially fake/exposed keys
        if self.profile == ValidationProfile.STRICT:
            security_warnings = self._check_api_key_security(api_key)
            warnings.extend(security_warnings)

        return errors, warnings

    def _validate_cost_controls(self, llm_config: any) -> Tuple[List[str], List[str]]:
        """Validate cost control settings.

        Args:
            llm_config: LLM configuration object.

        Returns:
            Tuple of (errors, warnings).
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not hasattr(llm_config, "cost_controls"):
            warnings.append("Missing 'cost_controls' configuration")
            return errors, warnings

        controls = llm_config.cost_controls

        # Validate daily budget
        if hasattr(controls, "daily_budget_usd"):
            if controls.daily_budget_usd <= 0:
                errors.append(
                    f"Daily budget must be positive, got: "
                    f"{controls.daily_budget_usd}"
                )
        else:
            warnings.append("Missing 'cost_controls.daily_budget_usd'")

        # Validate per-drawing limit
        if hasattr(controls, "per_drawing_limit_usd"):
            if controls.per_drawing_limit_usd <= 0:
                errors.append(
                    f"Per-drawing limit must be positive, got: "
                    f"{controls.per_drawing_limit_usd}"
                )
        else:
            warnings.append("Missing 'cost_controls.per_drawing_limit_usd'")

        return errors, warnings

    def _validate_api_key_format(self, provider: str, api_key: str) -> Tuple[bool, str]:
        """Validate API key format using provider registry.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic').
            api_key: The API key string to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        # Basic length check
        if len(api_key) < MIN_API_KEY_LENGTH:
            return False, f"API key too short (minimum {MIN_API_KEY_LENGTH})"

        # Try to get provider-specific validator
        try:
            provider_enum = ProviderName(provider)
            if provider_enum in self.PROVIDER_VALIDATORS:
                return self.PROVIDER_VALIDATORS[provider_enum](api_key)
        except ValueError:
            # Unknown provider, do basic validation only
            logger.debug(f"No specific validator for provider: {provider}")

        return True, ""

    def _check_api_key_security(self, api_key: str) -> List[str]:
        """Check for potentially fake or exposed API keys.

        Args:
            api_key: API key to check.

        Returns:
            List of security warnings.
        """
        warnings: List[str] = []

        # Check for known fake patterns
        api_key_lower = api_key.lower()
        for pattern in FAKE_KEY_PATTERNS:
            if pattern in api_key_lower:
                warnings.append(
                    f"API key contains suspicious pattern '{pattern}'. "
                    f"Ensure this is not a placeholder."
                )
                break

        return warnings


# Register provider-specific validators
@LLMValidator.register_provider(ProviderName.OPENAI)
def validate_openai_key(api_key: str) -> Tuple[bool, str]:
    """Validate OpenAI API key format."""
    if not api_key.startswith("sk-"):
        return False, "OpenAI keys should start with 'sk-'"
    return True, ""


@LLMValidator.register_provider(ProviderName.ANTHROPIC)
def validate_anthropic_key(api_key: str) -> Tuple[bool, str]:
    """Validate Anthropic API key format."""
    if not api_key.startswith("sk-ant-"):
        return False, "Anthropic keys should start with 'sk-ant-'"
    return True, ""


@LLMValidator.register_provider(ProviderName.GOOGLE)
def validate_google_key(api_key: str) -> Tuple[bool, str]:
    """Validate Google API key format."""
    if len(api_key) < MIN_GOOGLE_KEY_LENGTH:
        return (
            False,
            f"Google API key seems too short " f"(minimum {MIN_GOOGLE_KEY_LENGTH})",
        )
    return True, ""


# ============================================================================
# Model Validator
# ============================================================================


class ModelValidator(BaseValidator):
    """Validates YOLO model file configuration."""

    def validate(self) -> Tuple[List[str], List[str]]:
        """Validate YOLO model files.

        Returns:
            Tuple of (errors, warnings).
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Check shape detection config
        try:
            detection_config = self._get_section("shape_detection", required=False)
        except CriticalConfigError as e:
            warnings.append(str(e))
            return errors, warnings

        if detection_config is None:
            warnings.append("Missing 'shape_detection' configuration")
            return errors, warnings

        # Validate model path
        model_errors, model_warnings = self._validate_model_file(detection_config)
        errors.extend(model_errors)
        warnings.extend(model_warnings)

        # Validate device configuration
        device_errors, device_warnings = self._validate_device(detection_config)
        errors.extend(device_errors)
        warnings.extend(device_warnings)

        return errors, warnings

    def _validate_model_file(
        self, detection_config: any
    ) -> Tuple[List[str], List[str]]:
        """Validate model file existence and properties.

        Args:
            detection_config: Shape detection configuration object.

        Returns:
            Tuple of (errors, warnings).
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not hasattr(detection_config, "model_path"):
            errors.append("Missing 'shape_detection.model_path' configuration")
            return errors, warnings

        model_path = Path(detection_config.model_path)

        # Check existence
        if not model_path.exists():
            errors.append(f"YOLO model file not found: '{model_path}'")
            return errors, warnings

        # Check if it's a file
        if not model_path.is_file():
            errors.append(f"YOLO model path is not a file: '{model_path}'")
            return errors, warnings

        # Check read permission
        if not os.access(model_path, os.R_OK):
            errors.append(f"No read permission for model file: '{model_path}'")
            return errors, warnings

        # Check file size
        try:
            size_mb = model_path.stat().st_size / (1024 * 1024)
            if size_mb < MIN_MODEL_SIZE_MB:
                warnings.append(
                    f"YOLO model file seems small: {size_mb:.1f}MB "
                    f"(expected >{MIN_MODEL_SIZE_MB}MB). "
                    f"Verify this is a complete model file."
                )
            elif size_mb > MAX_MODEL_SIZE_MB:
                warnings.append(
                    f"YOLO model file seems large: {size_mb:.1f}MB "
                    f"(expected <{MAX_MODEL_SIZE_MB}MB). "
                    f"Verify this is the correct model."
                )
        except OSError as e:
            warnings.append(f"Cannot check model file size: {e}")

        return errors, warnings

    def _validate_device(self, detection_config: any) -> Tuple[List[str], List[str]]:
        """Validate device configuration for model inference.

        Args:
            detection_config: Shape detection configuration object.

        Returns:
            Tuple of (errors, warnings).
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not hasattr(detection_config, "device"):
            warnings.append("Missing 'shape_detection.device' configuration")
            return errors, warnings

        device = detection_config.device

        # Validate CUDA availability if configured
        if device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    warnings.append(
                        "CUDA device configured but not available. "
                        "Will fall back to CPU. Consider changing "
                        "'shape_detection.device' to 'cpu'."
                    )
            except ImportError:
                warnings.append(
                    "PyTorch not installed. Cannot verify CUDA availability. "
                    "Install PyTorch if you intend to use GPU acceleration."
                )

        return errors, warnings


# ============================================================================
# Database Validator
# ============================================================================


class DatabaseValidator(BaseValidator):
    """Validates database configuration."""

    def validate(self) -> Tuple[List[str], List[str]]:
        """Validate database configuration.

        Returns:
            Tuple of (errors, warnings).
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Check database config
        try:
            db_config = self._get_section("database")
        except CriticalConfigError as e:
            return [str(e)], []

        # Validate database path
        path_errors, path_warnings = self._validate_database_path(db_config)
        errors.extend(path_errors)
        warnings.extend(path_warnings)

        # Validate backup configuration
        backup_errors, backup_warnings = self._validate_backup_config(db_config)
        errors.extend(backup_errors)
        warnings.extend(backup_warnings)

        return errors, warnings

    def _validate_database_path(self, db_config: any) -> Tuple[List[str], List[str]]:
        """Validate database file path and permissions.

        Args:
            db_config: Database configuration object.

        Returns:
            Tuple of (errors, warnings).
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not hasattr(db_config, "path"):
            errors.append("Missing 'database.path' configuration")
            return errors, warnings

        db_path = Path(db_config.path)

        # Ensure parent directory exists
        if not db_path.parent.exists():
            try:
                db_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created database directory: {db_path.parent}")
            except PermissionError:
                errors.append(
                    f"Permission denied creating database directory: "
                    f"'{db_path.parent}'"
                )
                return errors, warnings
            except OSError as e:
                errors.append(
                    f"Cannot create database directory '{db_path.parent}': " f"{e}"
                )
                return errors, warnings

        # Check write permissions if database exists
        if db_path.exists():
            if not db_path.is_file():
                errors.append(f"Database path exists but is not a file: '{db_path}'")
                return errors, warnings

            if not os.access(db_path, os.W_OK):
                errors.append(f"No write permission for database file: '{db_path}'")

            if not os.access(db_path, os.R_OK):
                errors.append(f"No read permission for database file: '{db_path}'")

        # Check parent directory permissions
        if not os.access(db_path.parent, os.W_OK):
            errors.append(
                f"No write permission for database directory: " f"'{db_path.parent}'"
            )

        return errors, warnings

    def _validate_backup_config(self, db_config: any) -> Tuple[List[str], List[str]]:
        """Validate backup configuration.

        Args:
            db_config: Database configuration object.

        Returns:
            Tuple of (errors, warnings).
        """
        errors: List[str] = []
        warnings: List[str] = []

        backup_enabled = getattr(db_config, "backup_enabled", False)

        if backup_enabled:
            if not hasattr(db_config, "backup_frequency_hours"):
                warnings.append(
                    "Backup enabled but 'backup_frequency_hours' not "
                    "configured. Using default frequency."
                )

            # Validate backup directory if specified
            if hasattr(db_config, "backup_dir"):
                backup_dir = Path(db_config.backup_dir)
                if backup_dir.exists():
                    if not os.access(backup_dir, os.W_OK):
                        errors.append(
                            f"No write permission for backup directory: "
                            f"'{backup_dir}'"
                        )
                else:
                    warnings.append(
                        f"Backup directory does not exist: '{backup_dir}'. "
                        f"It will be created when needed."
                    )

        return errors, warnings


# ============================================================================
# Main Config Validator
# ============================================================================


class ConfigValidator:
    """Main configuration validator coordinating all validation modules.

    Attributes:
        config: The SystemConfig object to validate.
        profile: Validation strictness profile.
    """

    def __init__(
        self,
        config: "SystemConfig",
        profile: str = "standard",
    ) -> None:
        """Initialize the configuration validator.

        Args:
            config: SystemConfig object containing all configuration settings.
            profile: Validation profile ('permissive', 'standard', 'strict').

        Raises:
            ValueError: If profile is invalid.
        """
        self.config = config
        try:
            self.profile = ValidationProfile(profile)
        except ValueError:
            valid_profiles = [p.value for p in ValidationProfile]
            raise ValueError(
                f"Invalid profile '{profile}'. " f"Must be one of: {valid_profiles}"
            )

        logger.info(f"ConfigValidator initialized with profile: {self.profile.value}")

    def validate_all(self) -> ValidationResult:
        """Validate all configuration sections.

        Performs comprehensive validation using modular validators for each
        configuration area. Aggregates all errors, warnings, and info
        messages into a single ValidationResult.

        Returns:
            ValidationResult containing validation status and all messages.
        """
        logger.info(
            f"Starting configuration validation " f"(profile: {self.profile.value})..."
        )

        all_errors: List[str] = []
        all_warnings: List[str] = []
        info: List[str] = []

        # Create validators
        validators = [
            ("Paths", PathValidator(self.config, self.profile)),
            ("LLM Integration", LLMValidator(self.config, self.profile)),
            ("Model Files", ModelValidator(self.config, self.profile)),
            ("Database", DatabaseValidator(self.config, self.profile)),
        ]

        # Run each validator
        for name, validator in validators:
            logger.debug(f"Validating {name}...")
            try:
                errors, warnings = validator.validate()
                all_errors.extend(errors)
                all_warnings.extend(warnings)

                # Log section results
                if errors:
                    logger.error(
                        f"{name} validation failed with {len(errors)} error(s)"
                    )
                elif warnings:
                    logger.warning(
                        f"{name} validation passed with " f"{len(warnings)} warning(s)"
                    )
                else:
                    logger.info(f"{name} validation passed")

            except Exception as e:
                error_msg = f"Unexpected error validating {name}: {e}"
                logger.exception(error_msg)
                all_errors.append(error_msg)

        # Add summary info
        info.append(f"Configuration validation completed")
        info.append(
            f"Profile: {self.profile.value}, "
            f"Errors: {len(all_errors)}, "
            f"Warnings: {len(all_warnings)}"
        )

        is_valid = len(all_errors) == 0

        logger.info(
            f"Validation {'PASSED' if is_valid else 'FAILED'} - "
            f"{len(all_errors)} errors, {len(all_warnings)} warnings"
        )

        return ValidationResult(
            is_valid=is_valid,
            errors=tuple(all_errors),
            warnings=tuple(all_warnings),
            info=tuple(info),
        )

    def print_validation_report(
        self, result: ValidationResult, use_json: bool = False
    ) -> None:
        """Print formatted validation report to console.

        Displays a human-readable report with sections for information,
        warnings, and errors, along with a final validation status.
        Optionally outputs in JSON format.

        Args:
            result: ValidationResult object to format and display.
            use_json: If True, output JSON format instead of human-readable.
        """
        if use_json:
            print(result.to_json())
            return

        separator = "=" * 70
        print(f"\n{separator}")
        print("CONFIGURATION VALIDATION REPORT")
        print(f"Profile: {self.profile.value.upper()}")
        print(separator)

        # Print info
        if result.info:
            print("\n[INFO] Information:")
            for msg in result.info:
                print(f"   {msg}")

        # Print warnings
        if result.warnings:
            print("\n[WARN] Warnings:")
            for i, msg in enumerate(result.warnings, 1):
                print(f"   {i}. {msg}")

        # Print errors
        if result.errors:
            print("\n[ERROR] Errors:")
            for i, msg in enumerate(result.errors, 1):
                print(f"   {i}. {msg}")

        # Summary
        print(f"\n{'-' * 70}")
        if result.is_valid:
            print("[OK] Configuration is VALID")
            if result.warnings:
                print(f"     Note: {len(result.warnings)} warning(s) found")
        else:
            print("[FAIL] Configuration is INVALID")
            print(f"       Found {len(result.errors)} error(s)")
            print("\n       Action Required:")
            print("       - Fix all errors listed above")
            print("       - Re-run validation to confirm")
        print(f"{separator}\n")


# ============================================================================
# Validation Helper Functions
# ============================================================================


def validate_config(
    config: "SystemConfig",
    profile: str = "standard",
    print_report: bool = True,
    json_output: bool = False,
) -> ValidationResult:
    """Convenience function to validate configuration.

    Args:
        config: SystemConfig object to validate.
        profile: Validation profile ('permissive', 'standard', 'strict').
        print_report: Whether to print the validation report.
        json_output: Whether to use JSON format for report.

    Returns:
        ValidationResult object.

    Example:
        >>> from drawing_intelligence.utils.config_loader import Config
        >>> config = Config.load()
        >>> result = validate_config(config, profile="strict")
        >>> if not result:
        ...     sys.exit(1)
    """
    validator = ConfigValidator(config, profile=profile)
    result = validator.validate_all()

    if print_report:
        validator.print_validation_report(result, use_json=json_output)

    return result
