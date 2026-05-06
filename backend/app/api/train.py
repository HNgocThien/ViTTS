from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import sys
import subprocess
import asyncio
from config_loader import (
    DATASET_DIR,
    TRAIN_DIR,
    TRAIN_LOG_FILE,
    TRAIN_STATUS_FILE,
    TRAINING_DEFAULTS,
    MODELS
)

router = APIRouter()

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)

class TrainConfig(BaseModel):
    dataset: str
    base_model: str = TRAINING_DEFAULTS.get("base_model", "F5TTS-Base")
    epoch: int = TRAINING_DEFAULTS.get("epoch", 10)
    batch_size: int = TRAINING_DEFAULTS.get("batch_size", 4)
    learning_rate: float = TRAINING_DEFAULTS.get("learning_rate", 7.5e-5)
    num_warmup_updates: int = TRAINING_DEFAULTS.get("num_warmup_updates", 500)

@router.post("/")
async def start_training(config: TrainConfig, background_tasks: BackgroundTasks):
    model_cfg = MODELS.get(config.base_model)
    if not model_cfg:
        return {"error": f"Model configuration not found for {config.base_model}"}

    dataset_path = os.path.join(DATASET_DIR, config.dataset)
    metadata_file = f"{config.dataset}-metadata.txt"
    ckpt_path = os.path.join(model_cfg["ckpt_parent"], config.dataset)
    os.makedirs(ckpt_path, exist_ok=True)

    with open(TRAIN_LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"[INFO] Training initialized for dataset: {config.dataset}\n")
        f.write(f"[INFO] Base model: {config.base_model} (Type: {model_cfg.get('type')})\n")
        f.write(f"[INFO] Epochs: {config.epoch} | Batch size: {config.batch_size}\n")
        f.write(f"[INFO] Learning rate: {config.learning_rate} | Warmup: {config.num_warmup_updates}\n")
    
    with open(TRAIN_STATUS_FILE, "w") as f:
        f.write("running")

    cmd = [
        sys.executable,
        model_cfg.get("train_script"),
        "--data_dir", dataset_path,
        "--metadata_file", metadata_file,
        "--epochs", str(config.epoch),
        "--batch_size", str(config.batch_size),
        "--checkpoint_dir", ckpt_path,
        "--learning_rate", str(config.learning_rate),
        "--num_warmup_updates", str(config.num_warmup_updates),
        "--model_type", model_cfg.get("type", "f5tts")
    ]

    def run_training():
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{model_cfg['src']}:{env.get('PYTHONPATH', '')}"
        with open(TRAIN_LOG_FILE, "a", encoding="utf-8") as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )
            for line in process.stdout:
                log_f.write(line)
                log_f.flush()
            process.wait()
            if process.returncode == 0:
                log_f.write("[SUCCESS] Training completed successfully!\n")
                with open(TRAIN_STATUS_FILE, "w") as sf:
                    sf.write("done")
            else:
                log_f.write(f"[ERROR] Training failed with exit code {process.returncode}\n")
                with open(TRAIN_STATUS_FILE, "w") as sf:
                    sf.write("failed")

    background_tasks.add_task(run_training)
    return {"status": "Training started", "config": config.dict()}

@router.get("/train_status")
async def train_status():
    try:
        with open(TRAIN_STATUS_FILE, "r") as f:
            status = f.read().strip()
    except FileNotFoundError:
        status = "idle"
    return {"status": status}

@router.get("/train_stream")
async def train_stream(request: Request):
    async def log_generator():
        waited = 0
        while not os.path.exists(TRAIN_LOG_FILE):
            if await request.is_disconnected():
                return
            yield "data: Waiting for training to start...\n\n"
            await asyncio.sleep(1)
            waited += 1
            if waited > 30:
                yield "data: [TIMEOUT] Log file not found.\n\n"
                return

        with open(TRAIN_LOG_FILE, "r", encoding="utf-8") as f:
            while True:
                if await request.is_disconnected():
                    break
                line = f.readline()
                if line:
                    yield f"data: {line.rstrip()}\n\n"
                else:
                    try:
                        with open(TRAIN_STATUS_FILE, "r") as sf:
                            status = sf.read().strip()
                        if status in ("done", "failed"):
                            remaining = f.read()
                            for rl in remaining.splitlines():
                                yield f"data: {rl}\n\n"
                            yield f"data: [STATUS] {status.upper()}\n\n"
                            break
                    except FileNotFoundError:
                        pass
                    await asyncio.sleep(0.3)

    return StreamingResponse(log_generator(), media_type="text/event-stream")
