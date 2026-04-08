import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def load_config(path=CONFIG_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()

# Convenient exports for easy access
PATHS = config.get("paths", {})
TRAINING_DEFAULTS = config.get("training_defaults", {})
API_CONFIG = config.get("api", {})

DATASET_DIR = PATHS.get("dataset_dir", "/datasets")
CKPT_DIR = PATHS.get("ckpt_dir", "/app/F5-TTS/ckpts")
GENERATED_DIR = PATHS.get("generated_dir", "/app/generated")
TRAIN_LOG_FILE = PATHS.get("train_log_file", "/app/generated/training_log.txt")
TRAIN_STATUS_FILE = PATHS.get("train_status_file", "/app/generated/training_status.txt")
F5_TTS_SRC = PATHS.get("f5_tts_src", "/app/F5-TTS/src")
F5_TTS_TRAINING_SCRIPT = PATHS.get("f5_tts_training_script", "/app/F5-TTS/training_personal_TTS.py")
