"""
Configuration Validator Module

Validates system configuration for completeness and correctness.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import os


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result from configuration validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    info: List[str]

    def __post_init__(self):
        """Initialize empty lists if None."""
        self.errors = self.errors or []
        self.warnings = self.warnings or []
        self.info = self.info or []


class ConfigValidator:
    """Validate system configuration."""

    def __init__(self, config):
        """
        Initialize configuration validator.

        Args:
            config: SystemConfig object
        """
        self.config = config
        logger.info("ConfigValidator initialized")

    def validate_all(self) -> ValidationResult:
        """
        Validate all configuration sections.

        Returns:
            ValidationResult with errors/warnings
        """
        logger.info("Validating configuration...")

        errors = []
        warnings = []
        info = []

        # Validate paths
        path_errors = self.validate_paths()
        errors.extend(path_errors)

        # Validate LLM config
        llm_errors, llm_warnings = self.validate_llm_config()
        errors.extend(llm_errors)
        warnings.extend(llm_warnings)

        # Validate model files
        model_errors, model_warnings = self.validate_model_files()
        errors.extend(model_errors)
        warnings.extend(model_warnings)

        # Validate database
        db_errors, db_warnings = self.validate_database()
        errors.extend(db_errors)
        warnings.extend(db_warnings)

        # General info
        info.append(f"Configuration file validated")
        info.append(f"Found {len(errors)} errors, {len(warnings)} warnings")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid, errors=errors, warnings=warnings, info=info
        )

    def validate_paths(self) -> List[str]:
        """
        Validate all path configurations.

        Returns:
            List of error messages
        """
        errors = []

        # Check if paths config exists
        if not hasattr(self.config, "paths"):
            errors.append("Missing 'paths' configuration section")
            return errors

        paths = self.config.paths

        # Validate data directory
        if hasattr(paths, "data_dir"):
            data_dir = Path(paths.data_dir)
            if not data_dir.exists():
                try:
                    data_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created data directory: {data_dir}")
                except Exception as e:
                    errors.append(f"Cannot create data directory {data_dir}: {e}")
        else:
            errors.append("Missing 'paths.data_dir' configuration")

        # Validate models directory
        if hasattr(paths, "models_dir"):
            models_dir = Path(paths.models_dir)
            if not models_dir.exists():
                errors.append(f"Models directory does not exist: {models_dir}")
        else:
            errors.append("Missing 'paths.models_dir' configuration")

        # Validate output directory
        if hasattr(paths, "output_dir"):
            output_dir = Path(paths.output_dir)
            if not output_dir.exists():
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created output directory: {output_dir}")
                except Exception as e:
                    errors.append(f"Cannot create output directory {output_dir}: {e}")
        else:
            errors.append("Missing 'paths.output_dir' configuration")

        # Validate temp directory
        if hasattr(paths, "temp_dir"):
            temp_dir = Path(paths.temp_dir)
            if not temp_dir.exists():
                try:
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created temp directory: {temp_dir}")
                except Exception as e:
                    errors.append(f"Cannot create temp directory {temp_dir}: {e}")

        # Validate log directory
        if hasattr(paths, "log_dir"):
            log_dir = Path(paths.log_dir)
            if not log_dir.exists():
                try:
                    log_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created log directory: {log_dir}")
                except Exception as e:
                    errors.append(f"Cannot create log directory {log_dir}: {e}")

        return errors

    def validate_llm_config(self) -> tuple[List[str], List[str]]:
        """
        Validate LLM configuration and API keys.

        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []

        # Check if LLM config exists
        if not hasattr(self.config, "llm_integration"):
            warnings.append("Missing 'llm_integration' configuration")
            return errors, warnings

        llm_config = self.config.llm_integration

        # Check if LLM is enabled
        if not llm_config.enabled:
            warnings.append("LLM integration is disabled")
            return errors, warnings

        # Validate primary provider
        if hasattr(llm_config, "primary_provider"):
            primary = llm_config.primary_provider

            # Check API key environment variable
            if hasattr(primary, "api_key_env"):
                api_key = os.getenv(primary.api_key_env)
                if not api_key:
                    errors.append(
                        f"API key not found in environment: {primary.api_key_env}"
                    )
                else:
                    # Validate key format
                    is_valid, error_msg = self._validate_api_key_format(
                        primary.name, api_key
                    )
                    if not is_valid:
                        errors.append(
                            f"Invalid API key format for {primary.name}: {error_msg}"
                        )
            else:
                errors.append("Missing 'primary_provider.api_key_env'")
        else:
            errors.append("Missing 'primary_provider' configuration")

        # Validate fallback provider (optional)
        if hasattr(llm_config, "fallback_provider") and llm_config.fallback_provider:
            fallback = llm_config.fallback_provider

            if hasattr(fallback, "api_key_env"):
                api_key = os.getenv(fallback.api_key_env)
                if not api_key:
                    warnings.append(
                        f"Fallback API key not found: {fallback.api_key_env}"
                    )

        # Validate cost controls
        if hasattr(llm_config, "cost_controls"):
            controls = llm_config.cost_controls

            if hasattr(controls, "daily_budget_usd"):
                if controls.daily_budget_usd <= 0:
                    errors.append("Daily budget must be positive")
            else:
                warnings.append("Missing 'cost_controls.daily_budget_usd'")

            if hasattr(controls, "per_drawing_limit_usd"):
                if controls.per_drawing_limit_usd <= 0:
                    errors.append("Per-drawing limit must be positive")
            else:
                warnings.append("Missing 'cost_controls.per_drawing_limit_usd'")
        else:
            warnings.append("Missing 'cost_controls' configuration")

        return errors, warnings

    def validate_model_files(self) -> tuple[List[str], List[str]]:
        """
        Validate YOLO model files exist.

        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []

        # Check shape detection config
        if not hasattr(self.config, "shape_detection"):
            warnings.append("Missing 'shape_detection' configuration")
            return errors, warnings

        detection_config = self.config.shape_detection

        # Check model path
        if hasattr(detection_config, "model_path"):
            model_path = Path(detection_config.model_path)

            if not model_path.exists():
                errors.append(f"YOLO model file not found: {model_path}")
            elif not model_path.is_file():
                errors.append(f"YOLO model path is not a file: {model_path}")
            else:
                # Check file size (YOLO models are typically 5-50MB)
                size_mb = model_path.stat().st_size / (1024 * 1024)
                if size_mb < 1:
                    warnings.append(f"YOLO model file seems small: {size_mb:.1f}MB")
                elif size_mb > 200:
                    warnings.append(f"YOLO model file seems large: {size_mb:.1f}MB")
        else:
            errors.append("Missing 'shape_detection.model_path' configuration")

        # Check device configuration
        if hasattr(detection_config, "device"):
            device = detection_config.device
            if device == "cuda":
                try:
                    import torch

                    if not torch.cuda.is_available():
                        warnings.append("CUDA device configured but not available")
                except ImportError:
                    warnings.append("PyTorch not installed, cannot verify CUDA")

        return errors, warnings

    def validate_database(self) -> tuple[List[str], List[str]]:
        """
        Validate database configuration.

        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []

        # Check database config
        if not hasattr(self.config, "database"):
            errors.append("Missing 'database' configuration")
            return errors, warnings

        db_config = self.config.database

        # Check database path
        if hasattr(db_config, "path"):
            db_path = Path(db_config.path)

            # Check if parent directory exists
            if not db_path.parent.exists():
                try:
                    db_path.parent.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created database directory: {db_path.parent}")
                except Exception as e:
                    errors.append(f"Cannot create database directory: {e}")

            # Check write permissions
            if db_path.exists():
                if not os.access(db_path, os.W_OK):
                    errors.append(f"No write permission for database: {db_path}")
        else:
            errors.append("Missing 'database.path' configuration")

        # Check backup configuration
        if hasattr(db_config, "backup_enabled") and db_config.backup_enabled:
            if not hasattr(db_config, "backup_frequency_hours"):
                warnings.append("Backup enabled but frequency not configured")

        return errors, warnings

    def _validate_api_key_format(self, provider: str, api_key: str) -> tuple[bool, str]:
        """
        Validate API key format for provider.

        Args:
            provider: Provider name
            api_key: API key to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Basic length check
        if len(api_key) < 10:
            return False, "API key too short"

        # Provider-specific checks
        if provider == "openai":
            if not api_key.startswith("sk-"):
                return False, "OpenAI keys should start with 'sk-'"

        elif provider == "anthropic":
            if not api_key.startswith("sk-ant-"):
                return False, "Anthropic keys should start with 'sk-ant-'"

        elif provider == "google":
            # Google API keys vary in format
            if len(api_key) < 20:
                return False, "Google API key seems too short"

        return True, ""

    def print_validation_report(self, result: ValidationResult) -> None:
        """
        Print formatted validation report to console.

        Args:
            result: ValidationResult to print
        """
        print("\n" + "=" * 70)
        print("CONFIGURATION VALIDATION REPORT")
        print("=" * 70)

        # Print info
        if result.info:
            print("\nℹ️  Information:")
            for msg in result.info:
                print(f"   {msg}")

        # Print warnings
        if result.warnings:
            print("\n⚠️  Warnings:")
            for msg in result.warnings:
                print(f"   {msg}")

        # Print errors
        if result.errors:
            print("\n❌ Errors:")
            for msg in result.errors:
                print(f"   {msg}")

        # Summary
        print("\n" + "-" * 70)
        if result.is_valid:
            print("✅ Configuration is VALID")
        else:
            print("❌ Configuration is INVALID")
            print(f"   Found {len(result.errors)} error(s)")
        print("=" * 70 + "\n")
