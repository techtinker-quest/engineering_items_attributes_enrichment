# test_config.py
from src.drawing_intelligence.utils.config_loader import Config

config = Config.load()
print(config.paths["data_dir"])
print(config.shape_detection["model_path"])
print(config.entity_extraction["oem_dictionary_path"])
print(config.batch_processing["checkpointing"]["batch_checkpoint_dir"])
print(config.logging["log_dir"])
print(Config.validate(config))
