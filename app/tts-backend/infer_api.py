from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os
import sys
import subprocess
import uuid
from config_loader import (
    API_CONFIG,
    DATASET_DIR,
    CKPT_DIR,
    GENERATED_DIR,
    F5_TTS_SRC
)

sys.path.insert(0, F5_TTS_SRC)

app = FastAPI(title="TTS-Backend Infer API", version=API_CONFIG.get("version", "1.1.0"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)

class GenerateConfig(BaseModel):
    text: str
    checkpoint: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    dataset: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "TTS Infer Backend is running", "version": API_CONFIG.get("version", "1.1.0")}

@app.get("/datasets")
async def list_datasets():
    try:
        datasets = [
            d for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d))
        ]
        return {"datasets": sorted(datasets)}
    except Exception:
        return {"datasets": []}

@app.get("/datasets/{dataset_name}/ref_audios")
async def list_ref_audios(dataset_name: str):
    wav_dir = os.path.join(DATASET_DIR, dataset_name, "wavs")
    if not os.path.isdir(wav_dir):
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
    file_path = os.path.join(DATASET_DIR, dataset_name, audio_path)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/wav")

@app.get("/checkpoints")
async def list_checkpoints():
    try:
        ckpts = []
        for root_dir, _, files in os.walk(CKPT_DIR):
            for f in files:
                if f.endswith(".pt") or f.endswith(".safetensors"):
                    rel = os.path.relpath(os.path.join(root_dir, f), CKPT_DIR)
                    ckpts.append(rel.replace("\\", "/"))
        return {"checkpoints": sorted(ckpts)}
    except Exception:
        return {"checkpoints": []}

def _find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    best = None
    best_mtime = -1
    for root_dir, _, files in os.walk(ckpt_dir):
        for f in files:
            if f.endswith(".pt") or f.endswith(".safetensors"):
                full = os.path.join(root_dir, f)
                mtime = os.path.getmtime(full)
                if mtime > best_mtime:
                    best_mtime = mtime
                    best = full
    return best

@app.post("/generate")
async def generate_voice(config: GenerateConfig):
    ckpt_path = None
    if config.checkpoint:
        candidate = os.path.join(CKPT_DIR, config.checkpoint)
        if os.path.exists(candidate):
            ckpt_path = candidate
        elif os.path.exists(config.checkpoint):
            ckpt_path = config.checkpoint

    if ckpt_path is None:
        ckpt_path = _find_latest_checkpoint(CKPT_DIR)

    ref_audio_path = None
    if config.ref_audio and config.dataset:
        candidate = os.path.join(DATASET_DIR, config.dataset, config.ref_audio)
        if os.path.exists(candidate):
            ref_audio_path = candidate
    elif config.ref_audio and os.path.exists(config.ref_audio):
        ref_audio_path = config.ref_audio

    out_filename = f"output_{uuid.uuid4().hex[:8]}.wav"
    out_path = os.path.join(GENERATED_DIR, out_filename)

    try:
        from f5_tts.api import F5TTS
        tts = F5TTS(model_type="F5TTS", ckpt_file=ckpt_path)
        tts.infer(
            ref_file=ref_audio_path,
            ref_text=config.ref_text or "",
            gen_text=config.text,
            file_wave=out_path
        )
        return {"status": "Generation complete", "audio_path": f"/audio/{out_filename}"}
    except Exception as e:
        print(f"API Inference failed: {e}")

    try:
        cli_cmd = [sys.executable, "-m", "f5_tts.infer.infer_cli", "--model", "F5TTS", "--gen_text", config.text, "--output_dir", GENERATED_DIR, "--output_file", out_filename]
        if ckpt_path: cli_cmd += ["--ckpt_file", ckpt_path]
        if ref_audio_path:
            cli_cmd += ["--ref_audio", ref_audio_path]
            if config.ref_text: cli_cmd += ["--ref_text", config.ref_text]

        result = subprocess.run(cli_cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and os.path.exists(out_path):
            return {"status": "Generation complete", "audio_path": f"/audio/{out_filename}"}
    except Exception as e:
        print(f"CLI Inference failed: {e}")

    _write_silent_wav(out_path)
    return {"status": "Generation complete (placeholder)", "audio_path": f"/audio/{out_filename}", "warning": "Runtime failed, returned silent audio."}

def _write_silent_wav(path: str, duration_s: float = 1.0, sample_rate: int = 24000):
    import struct
    num_samples = int(sample_rate * duration_s)
    data_size = num_samples * 2
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    file_path = os.path.join(GENERATED_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Audio file not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=API_CONFIG.get("infer_port", 8001))
