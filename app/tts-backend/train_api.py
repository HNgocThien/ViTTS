from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import sys
import subprocess
import asyncio
from config_loader import (
    API_CONFIG,
    DATASET_DIR,
    CKPT_DIR,
    TRAIN_LOG_FILE,
    TRAIN_STATUS_FILE,
    F5_TTS_SRC,
    F5_TTS_TRAINING_SCRIPT,
    TRAINING_DEFAULTS
)

# Add F5-TTS to path
sys.path.insert(0, F5_TTS_SRC)

app = FastAPI(title="TTS-Backend Train API", version=API_CONFIG.get("version", "1.1.0"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(TRAIN_LOG_FILE), exist_ok=True)

class TrainConfig(BaseModel):
    dataset: str
    base_model: str = TRAINING_DEFAULTS.get("base_model", "F5TTS-Base")
    epoch: int = TRAINING_DEFAULTS.get("epoch", 10)
    batch_size: int = TRAINING_DEFAULTS.get("batch_size", 4)
    learning_rate: float = TRAINING_DEFAULTS.get("learning_rate", 7.5e-5)
    num_warmup_updates: int = TRAINING_DEFAULTS.get("num_warmup_updates", 500)

@app.get("/")
async def root():
    return {"message": "TTS Train Backend is running", "version": API_CONFIG.get("version", "1.1.0")}

@app.post("/train")
async def start_training(config: TrainConfig, background_tasks: BackgroundTasks):
    dataset_path = os.path.join(DATASET_DIR, config.dataset)
    metadata_file = f"{config.dataset}-metadata.txt"
    ckpt_path = os.path.join(CKPT_DIR, config.dataset)

    with open(TRAIN_LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"[INFO] Training initialized for dataset: {config.dataset}\n")
        f.write(f"[INFO] Base model: {config.base_model}\n")
        f.write(f"[INFO] Epochs: {config.epoch} | Batch size: {config.batch_size}\n")
        f.write(f"[INFO] Learning rate: {config.learning_rate} | Warmup: {config.num_warmup_updates}\n")
    
    with open(TRAIN_STATUS_FILE, "w") as f:
        f.write("running")

    cmd = [
        sys.executable,
        F5_TTS_TRAINING_SCRIPT,
        "--data_dir", dataset_path,
        "--metadata_file", metadata_file,
        "--epochs", str(config.epoch),
        "--batch_size", str(config.batch_size),
        "--checkpoint_dir", ckpt_path,
        "--learning_rate", str(config.learning_rate),
        "--num_warmup_updates", str(config.num_warmup_updates),
    ]

    def run_training():
        with open(TRAIN_LOG_FILE, "a", encoding="utf-8") as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
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

@app.get("/train/status")
async def train_status():
    try:
        with open(TRAIN_STATUS_FILE, "r") as f:
            status = f.read().strip()
    except FileNotFoundError:
        status = "idle"
    return {"status": status}

@app.get("/train_stream")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=API_CONFIG.get("train_port", 8000))
