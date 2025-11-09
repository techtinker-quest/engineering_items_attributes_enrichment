"""Configuration loading and validation for the drawing intelligence system.

This module provides utilities to load system configuration from YAML files,
resolve relative paths to absolute paths based on project root, and validate
that all required filesystem paths exist.

Typical usage example:
    config = Config.load()
    errors = Config.validate(config)
    if errors:
        raise ConfigurationError(f"Configuration invalid: {errors}")

Author: CLAUDE + Sandeep A (01Nov2025)
Refactored: [Current Date] - Added path resolution automation and pathlib usage
"""

import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional


class SystemConfig:
    """Container for system configuration parameters.

    Attributes:
        paths: Dictionary containing filesystem paths (data_dir, models_dir, etc.).
        database: Dictionary containing database configuration (path, etc.).
        shape_detection: Dictionary containing shape detection model configuration.
        entity_extraction: Dictionary containing entity extraction configuration.
        batch_processing: Dictionary containing batch processing configuration.
        logging: Dictionary containing logging configuration.
    """

    def __init__(self, **config_dict: Dict[str, Any]) -> None:
        """Initialize SystemConfig from configuration dictionary.

        Args:
            **config_dict: Configuration dictionary with required keys:
                paths, database, shape_detection, entity_extraction,
                batch_processing, logging.

        Raises:
            KeyError: If any required configuration section is missing.
        """
        required_keys = [
            "paths",
            "database",
            "shape_detection",
            "entity_extraction",
            "batch_processing",
            "logging",
        ]

        missing_keys = [key for key in required_keys if key not in config_dict]
        if missing_keys:
            raise KeyError(f"Missing required configuration sections: {missing_keys}")

        self.paths: Dict[str, str] = config_dict["paths"]
        self.database: Dict[str, Any] = config_dict["database"]
        self.shape_detection: Dict[str, Any] = config_dict["shape_detection"]
        self.entity_extraction: Dict[str, Any] = config_dict["entity_extraction"]
        self.batch_processing: Dict[str, Any] = config_dict["batch_processing"]
        self.logging: Dict[str, Any] = config_dict["logging"]


class Config:
    """Static utility class for loading and validating configuration files.

    This class provides methods to load configuration from YAML files and
    validate that all required paths and settings are properly configured.
    """

    # Define all configuration keys that contain relative paths
    # These will be automatically resolved to absolute paths during loading
    _RELATIVE_PATH_KEYS = [
        "paths.data_dir",
        "paths.models_dir",
        "paths.output_dir",
        "paths.temp_dir",
        "database.path",
        "entity_extraction.oem_dictionary_path",
        "shape_detection.model_path",
        "batch_processing.checkpointing.batch_checkpoint_dir",
        "logging.log_dir",
    ]

    @staticmethod
    def _resolve_nested_path(
        config_dict: Dict[str, Any], key_path: str, project_root: Path
    ) -> None:
        """Resolve a nested config path to absolute path in-place.

        Navigates through nested dictionary structure using dot notation
        and converts relative paths to absolute paths based on project root.

        Args:
            config_dict: Configuration dictionary to modify in-place.
            key_path: Dot-separated path to the key (e.g., "paths.data_dir").
            project_root: Project root directory for resolving relative paths.

        Raises:
            KeyError: If any key in the path doesn't exist in config_dict.
        """
        keys = key_path.split(".")
        current = config_dict

        # Navigate to parent dictionary
        for key in keys[:-1]:
            current = current[key]

        # Resolve the final path value
        final_key = keys[-1]
        relative_path = current[final_key]
        current[final_key] = str(project_root / relative_path)

    @staticmethod
    def load(config_path: Optional[str] = "config/system_config.yaml") -> SystemConfig:
        """Load system configuration from a YAML file.

        Reads configuration from the specified YAML file and resolves all
        relative paths to absolute paths based on the project root directory.
        All paths defined in _RELATIVE_PATH_KEYS are automatically resolved.

        Args:
            config_path: Relative path to the configuration YAML file from
                project root. Defaults to "config/system_config.yaml".

        Returns:
            SystemConfig object containing the loaded and resolved configuration.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            yaml.YAMLError: If the configuration file is not valid YAML.
            KeyError: If required configuration keys are missing.
            ValueError: If the configuration file doesn't contain a dictionary.
        """
        # Resolve paths relative to project root (4 levels up from this file)
        project_root = Path(__file__).parent.parent.parent.parent
        config_file_path = project_root / config_path

        if not config_file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file_path}")

        try:
            with open(config_file_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Failed to parse configuration file: {config_file_path}"
            ) from e

        if not isinstance(config_dict, dict):
            raise ValueError("Configuration file must contain a YAML dictionary")

        # Automatically resolve all relative paths defined in _RELATIVE_PATH_KEYS
        for path_key in Config._RELATIVE_PATH_KEYS:
            try:
                Config._resolve_nested_path(config_dict, path_key, project_root)
            except KeyError as e:
                raise KeyError(
                    f"Missing required configuration path: {path_key}"
                ) from e

        return SystemConfig(**config_dict)

    @staticmethod
    def validate(config: SystemConfig) -> List[str]:
        """Validate that all configured paths exist.

        Checks that all critical filesystem paths in the configuration exist
        and are accessible. Distinguishes between files and directories and
        validates accordingly.

        Args:
            config: SystemConfig object to validate.

        Returns:
            List of error messages for any missing or inaccessible paths.
            Empty list if all paths are valid.
        """
        errors: List[str] = []

        # Define paths to validate with their expected types
        # Format: (config_key, path_value, should_be_dir, must_exist)
        paths_to_check = [
            (
                "paths.data_dir",
                config.paths["data_dir"],
                True,  # should_be_dir
                True,  # must_exist
            ),
            (
                "paths.models_dir",
                config.paths["models_dir"],
                True,
                True,
            ),
            (
                "paths.output_dir",
                config.paths["output_dir"],
                True,
                True,
            ),
            (
                "paths.temp_dir",
                config.paths["temp_dir"],
                True,
                True,
            ),
            (
                "shape_detection.model_path",
                config.shape_detection["model_path"],
                False,  # should be a file
                True,
            ),
            (
                "entity_extraction.oem_dictionary_path",
                config.entity_extraction["oem_dictionary_path"],
                False,
                True,
            ),
            (
                "batch_processing.checkpointing.batch_checkpoint_dir",
                config.batch_processing["checkpointing"]["batch_checkpoint_dir"],
                True,
                True,
            ),
            (
                "logging.log_dir",
                config.logging["log_dir"],
                True,
                True,
            ),
            (
                "database.path",
                config.database["path"],
                False,
                False,  # Database file may not exist initially
            ),
        ]

        for path_key, path_value, should_be_dir, must_exist in paths_to_check:
            path_obj = Path(path_value)

            if not path_obj.exists():
                if must_exist:
                    errors.append(f"Path not found for {path_key}: {path_value}")
            elif should_be_dir and not path_obj.is_dir():
                errors.append(
                    f"Expected directory for {path_key}, "
                    f"but found file: {path_value}"
                )
            elif not should_be_dir and path_obj.is_dir():
                errors.append(
                    f"Expected file for {path_key}, "
                    f"but found directory: {path_value}"
                )

        return errors
