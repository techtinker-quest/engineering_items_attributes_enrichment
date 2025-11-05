# src/drawing_intelligence/utils/config_loader.py
# Module to load and validate system configuration from a YAML file
# Includes path resolution relative to project root
# Defines SystemConfig class to hold configuration parameters
# Defines Config class with static methods to load and validate configuration
# Uses PyYAML for YAML parsing
# Handles paths for data directories, model files, logging, and database
# Validates existence of critical paths and returns errors if any are missing
# Author: GROK AI + Sandeep A (01Nov2025)


import os
import yaml
from pathlib import Path


class SystemConfig:
    def __init__(self, **config_dict):
        self.paths = config_dict["paths"]
        self.database = config_dict["database"]
        self.shape_detection = config_dict["shape_detection"]
        self.entity_extraction = config_dict["entity_extraction"]
        self.batch_processing = config_dict["batch_processing"]
        self.logging = config_dict["logging"]
        # Add other fields as needed


class Config:
    @staticmethod
    def load(config_path: str = "config/system_config.yaml") -> SystemConfig:
        # Resolve paths relative to project root
        project_root = Path(__file__).parent.parent.parent.parent
        with open(project_root / config_path) as f:
            config_dict = yaml.safe_load(f)
        # Resolve relative paths
        config_dict["paths"]["data_dir"] = str(
            project_root / config_dict["paths"]["data_dir"]
        )
        config_dict["paths"]["models_dir"] = str(
            project_root / config_dict["paths"]["models_dir"]
        )
        config_dict["paths"]["output_dir"] = str(
            project_root / config_dict["paths"]["output_dir"]
        )
        config_dict["paths"]["temp_dir"] = str(
            project_root / config_dict["paths"]["temp_dir"]
        )
        config_dict["database"]["path"] = str(
            project_root / config_dict["database"]["path"]
        )
        config_dict["entity_extraction"]["oem_dictionary_path"] = str(
            project_root / config_dict["entity_extraction"]["oem_dictionary_path"]
        )
        config_dict["shape_detection"]["model_path"] = str(
            project_root / config_dict["shape_detection"]["model_path"]
        )
        config_dict["batch_processing"]["checkpointing"]["batch_checkpoint_dir"] = str(
            project_root
            / config_dict["batch_processing"]["checkpointing"]["batch_checkpoint_dir"]
        )
        config_dict["logging"]["log_dir"] = str(
            project_root / config_dict["logging"]["log_dir"]
        )
        return SystemConfig(**config_dict)

    @staticmethod
    def validate(config: SystemConfig) -> list:
        """Validate configuration and return list of errors"""
        errors = []
        for path_key, path_value in [
            ("shape_detection.model_path", config.shape_detection["model_path"]),
            (
                "entity_extraction.oem_dictionary_path",
                config.entity_extraction["oem_dictionary_path"],
            ),
            ("database.path", config.database["path"]),
            ("paths.data_dir", config.paths["data_dir"]),
            ("paths.output_dir", config.paths["output_dir"]),
            ("paths.temp_dir", config.paths["temp_dir"]),
            (
                "batch_processing.checkpointing.batch_checkpoint_dir",
                config.batch_processing["checkpointing"]["batch_checkpoint_dir"],
            ),
            ("logging.log_dir", config.logging["log_dir"]),
        ]:
            if not os.path.exists(path_value):
                errors.append(f"Path not found for {path_key}: {path_value}")
        return errors
