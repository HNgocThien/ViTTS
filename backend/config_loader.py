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
MODELS = config.get("models", {})

DATASET_DIR = PATHS.get("dataset_dir", "/shared_storage/datasets")
TRAIN_DIR = PATHS.get("train_dir", "/shared_storage/outputs/train")
INFER_DIR = PATHS.get("infer_dir", "/shared_storage/outputs/infer")
TRAIN_LOG_FILE = PATHS.get("train_log_file", "/shared_storage/outputs/train/training_log.txt")
TRAIN_STATUS_FILE = PATHS.get("train_status_file", "/shared_storage/outputs/train/training_status.txt")
INFER_LOG_FILE = PATHS.get("infer_log_file", "/shared_storage/outputs/infer/inference_debug.log")
INFER_STATUS_FILE = PATHS.get("infer_status_file", "/shared_storage/outputs/infer/inference_status.txt")
PROMPTS_CSV = PATHS.get("prompts_csv", "/shared_storage/prompts/vietnamese_train.csv")
