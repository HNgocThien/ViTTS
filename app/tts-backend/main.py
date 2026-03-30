from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import os
import sys
import subprocess
import asyncio
import uuid

# Add F5-TTS to path
sys.path.insert(0, "/app/F5-TTS/src")

app = FastAPI(title="TTS-Backend API", version="1.0.0")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASET_DIR = "/datasets"
CKPT_DIR = "/app/ckpts"
GENERATED_DIR = "/app/generated"
TRAIN_LOG_FILE = "/app/generated/training_log.txt"
TRAIN_STATUS_FILE = "/app/generated/training_status.txt"  # "running" | "done" | "failed"

# Ensure directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)


# ─── Models ───────────────────────────────────────────────────────────────────

class TrainConfig(BaseModel):
    dataset: str
    base_model: str = "F5TTS-Base"
    epoch: int = 10
    batch_size: int = 4
    learning_rate: float = 7.5e-5
    num_warmup_updates: int = 500

class GenerateConfig(BaseModel):
    text: str
    checkpoint: Optional[str] = None
    ref_audio: Optional[str] = None   # relative path inside dataset, e.g. "wavs/sample.wav"
    ref_text: Optional[str] = None    # transcription for the ref audio (for conditioning)
    dataset: Optional[str] = None     # which dataset the ref audio belongs to


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "TTS Backend is running", "version": "1.0.0"}


# ─── Datasets ─────────────────────────────────────────────────────────────────

@app.get("/datasets")
async def list_datasets():
    try:
        datasets = [
            d for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d))
        ]
        return {"datasets": datasets}
    except Exception:
        return {"datasets": []}


@app.get("/datasets/{dataset_name}/ref_audios")
async def list_ref_audios(dataset_name: str):
    """
    List WAV files inside a dataset's wavs/ subfolder.
    Returns paths relative to the dataset root so the client can display them.
    """
    wav_dir = os.path.join(DATASET_DIR, dataset_name, "wavs")
    if not os.path.isdir(wav_dir):
        # Fallback: scan the dataset root itself
        wav_dir = os.path.join(DATASET_DIR, dataset_name)

    try:
        wavs = [
            f"wavs/{f}" if os.path.isdir(os.path.join(DATASET_DIR, dataset_name, "wavs")) else f
            for f in os.listdir(wav_dir)
            if f.lower().endswith(".wav")
        ]
        return {"ref_audios": sorted(wavs)}
    except Exception as e:
        return {"ref_audios": [], "error": str(e)}


@app.get("/datasets/{dataset_name}/audio/{audio_path:path}")
async def serve_dataset_audio(dataset_name: str, audio_path: str):
    """Serve a raw audio file from a dataset for preview in the UI."""
    file_path = os.path.join(DATASET_DIR, dataset_name, audio_path)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/wav")


# ─── Checkpoints ──────────────────────────────────────────────────────────────

@app.get("/checkpoints")
async def list_checkpoints():
    try:
        ckpts = []
        for root, dirs, files in os.walk(CKPT_DIR):
            for f in files:
                if f.endswith(".pt") or f.endswith(".safetensors"):
                    rel = os.path.relpath(os.path.join(root, f), CKPT_DIR)
                    ckpts.append(rel.replace("\\", "/"))
        return {"checkpoints": ckpts}
    except Exception:
        return {"checkpoints": []}


# ─── Training ─────────────────────────────────────────────────────────────────

@app.post("/train")
async def start_training(config: TrainConfig, background_tasks: BackgroundTasks):
    dataset_path = os.path.join(DATASET_DIR, config.dataset)
    metadata_file = f"{config.dataset}-metadata.txt"
    ckpt_path = os.path.join(CKPT_DIR, config.dataset)

    # Reset log files
    os.makedirs(os.path.dirname(TRAIN_LOG_FILE), exist_ok=True)
    with open(TRAIN_LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"[INFO] Training initialized for dataset: {config.dataset}\n")
        f.write(f"[INFO] Base model: {config.base_model}\n")
        f.write(f"[INFO] Epochs: {config.epoch} | Batch size: {config.batch_size}\n")
        f.write(f"[INFO] Learning rate: {config.learning_rate} | Warmup: {config.num_warmup_updates}\n")
    with open(TRAIN_STATUS_FILE, "w") as f:
        f.write("running")

    cmd = [
        sys.executable,
        "/app/training_personal_TTS.py",
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
    """SSE endpoint that tails the training log file in real-time."""
    async def log_generator():
        # Wait until log file exists (training might not yet have started)
        waited = 0
        while not os.path.exists(TRAIN_LOG_FILE):
            if await request.is_disconnected():
                return
            yield "data: Waiting for training to start...\n\n"
            await asyncio.sleep(1)
            waited += 1
            if waited > 30:
                yield "data: [TIMEOUT] Log file not found after 30 seconds.\n\n"
                return

        with open(TRAIN_LOG_FILE, "r", encoding="utf-8") as f:
            while True:
                if await request.is_disconnected():
                    break
                line = f.readline()
                if line:
                    formatted = line.rstrip("\n")
                    yield f"data: {formatted}\n\n"
                else:
                    # Check if training finished
                    try:
                        with open(TRAIN_STATUS_FILE, "r") as sf:
                            status = sf.read().strip()
                        if status in ("done", "failed"):
                            # Drain any remaining lines
                            remaining = f.read()
                            for rl in remaining.splitlines():
                                yield f"data: {rl}\n\n"
                            yield f"data: [STATUS] {status.upper()}\n\n"
                            break
                    except FileNotFoundError:
                        pass
                    await asyncio.sleep(0.3)

    return StreamingResponse(log_generator(), media_type="text/event-stream")


# ─── Inference / Generation ───────────────────────────────────────────────────

def _find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Walk ckpt_dir and return the most recently modified .pt file."""
    best = None
    best_mtime = -1
    for root, _, files in os.walk(ckpt_dir):
        for f in files:
            if f.endswith(".pt") or f.endswith(".safetensors"):
                full = os.path.join(root, f)
                mtime = os.path.getmtime(full)
                if mtime > best_mtime:
                    best_mtime = mtime
                    best = full
    return best


@app.post("/generate")
async def generate_voice(config: GenerateConfig):
    """
    Run F5-TTS inference.

    Strategy:
      1. Resolve checkpoint path (explicit > latest in ckpts dir).
      2. Resolve ref_audio path if provided.
      3. Call f5-tts_infer-cli (the CLI installed with F5-TTS pip package).
      4. If the CLI is unavailable, try the Python API directly.
      5. Return path to the generated WAV file.
    """
    # ── 1. Resolve checkpoint ──────────────────────────────────────────────
    ckpt_path = None
    if config.checkpoint:
        candidate = os.path.join(CKPT_DIR, config.checkpoint)
        if os.path.exists(candidate):
            ckpt_path = candidate
        elif os.path.exists(config.checkpoint):
            ckpt_path = config.checkpoint

    if ckpt_path is None:
        ckpt_path = _find_latest_checkpoint(CKPT_DIR)

    # ── 2. Resolve ref_audio ───────────────────────────────────────────────
    ref_audio_path = None
    if config.ref_audio and config.dataset:
        candidate = os.path.join(DATASET_DIR, config.dataset, config.ref_audio)
        if os.path.exists(candidate):
            ref_audio_path = candidate
    elif config.ref_audio and os.path.exists(config.ref_audio):
        ref_audio_path = config.ref_audio

    # ── 3. Output file ─────────────────────────────────────────────────────
    out_filename = f"output_{uuid.uuid4().hex[:8]}.wav"
    out_path = os.path.join(GENERATED_DIR, out_filename)

    # ── 4. Try F5-TTS Python API ───────────────────────────────────────────
    try:
        from f5_tts.api import F5TTS

        tts = F5TTS(
            model_type="F5TTS",
            ckpt_file=ckpt_path,
        )

        wav, sr, spec = tts.infer(
            ref_file=ref_audio_path,
            ref_text=config.ref_text or "",
            gen_text=config.text,
            file_wave=out_path,
            seed=-1,
        )
        return {"status": "Generation complete", "audio_path": f"/audio/{out_filename}"}

    except ImportError:
        pass  # F5-TTS Python API not available, try CLI
    except Exception as e:
        # Log and fall through to CLI attempt
        error_msg = str(e)

    # ── 5. Try f5-tts CLI ──────────────────────────────────────────────────
    try:
        cli_cmd = [
            "f5-tts_infer-cli",
            "--model", "F5TTS",
            "--gen_text", config.text,
            "--output_dir", GENERATED_DIR,
            "--output_file", out_filename,
        ]
        if ckpt_path:
            cli_cmd += ["--ckpt_file", ckpt_path]
        if ref_audio_path:
            cli_cmd += ["--ref_audio", ref_audio_path]
            if config.ref_text:
                cli_cmd += ["--ref_text", config.ref_text]

        result = subprocess.run(cli_cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and os.path.exists(out_path):
            return {"status": "Generation complete", "audio_path": f"/audio/{out_filename}"}
        else:
            raise RuntimeError(result.stderr or "CLI exited with non-zero code")

    except FileNotFoundError:
        pass  # CLI not installed

    # ── 6. Fallback: write a placeholder silent WAV ────────────────────────
    # (so the UI doesn't break in dev / pre-GPU environments)
    _write_silent_wav(out_path)
    return {
        "status": "Generation complete (placeholder - no GPU model loaded)",
        "audio_path": f"/audio/{out_filename}",
        "warning": "F5-TTS runtime not available. Returned silent placeholder."
    }


def _write_silent_wav(path: str, duration_s: float = 1.0, sample_rate: int = 24000):
    """Write a minimal silent PCM WAV file."""
    import struct
    num_samples = int(sample_rate * duration_s)
    data_size = num_samples * 2  # 16-bit mono
    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)


# ─── Audio serving ────────────────────────────────────────────────────────────

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    file_path = os.path.join(GENERATED_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Audio file not found")


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
