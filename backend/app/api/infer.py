import os
import sys
import subprocess
import uuid
import asyncio

from fastapi import APIRouter, HTTPException, File, UploadFile, BackgroundTasks, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from config_loader import (
    DATASET_DIR,
    INFER_DIR,
    INFER_LOG_FILE,
    INFER_STATUS_FILE,
    MODELS,
)

router = APIRouter()

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(INFER_DIR, exist_ok=True)
TMP_REF_DIR = os.path.join(os.path.dirname(INFER_DIR), "tmp_ref")
os.makedirs(TMP_REF_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────

def log_inference(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(INFER_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def reset_inference_log():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(INFER_LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"[{timestamp}] Inference log reset for new request.\n")


if not os.path.exists(INFER_LOG_FILE):
    log_inference("Inference logging initialized.")


# ─────────────────────────────────────────────
# Request schema
# ─────────────────────────────────────────────

class GenerateConfig(BaseModel):
    text: str
    checkpoint: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    model: Optional[str] = "F5TTS-Base"
    dataset: Optional[str] = None


# ─────────────────────────────────────────────
# Dataset / checkpoint utilities
# ─────────────────────────────────────────────

@router.get("/datasets")
async def list_datasets():
    try:
        datasets = [
            d for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d))
        ]
        return {"datasets": sorted(datasets)}
    except Exception:
        return {"datasets": []}


@router.post("/upload_ref")
async def upload_reference_audio(audio: UploadFile = File(...)):
    """Upload a temporary reference audio and clean up old ones."""
    try:
        import time
        now = time.time()
        for f in os.listdir(TMP_REF_DIR):
            fpath = os.path.join(TMP_REF_DIR, f)
            if os.path.isfile(fpath) and (now - os.path.getmtime(fpath)) > 300:
                try:
                    os.remove(fpath)
                except Exception:
                    pass

        filename = f"ref_{uuid.uuid4().hex[:8]}.wav"
        save_path = os.path.join(TMP_REF_DIR, filename)
        with open(save_path, "wb") as f:
            f.write(await audio.read())
        return {"success": True, "ref_at": save_path}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/datasets/{dataset_name}/ref_audios")
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


@router.get("/datasets/{dataset_name}/audio/{audio_path:path}")
async def serve_dataset_audio(dataset_name: str, audio_path: str):
    file_path = os.path.join(DATASET_DIR, dataset_name, audio_path)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/wav")


@router.get("/checkpoints")
async def list_checkpoints():
    try:
        ckpts = []
        unique_ckpt_dirs = set(m["ckpt_parent"] for m in MODELS.values())
        for ckpt_dir in unique_ckpt_dirs:
            if not os.path.exists(ckpt_dir):
                continue
            for root_dir, _, files in os.walk(ckpt_dir):
                for f in files:
                    if f.endswith(".pt") or f.endswith(".safetensors"):
                        rel = os.path.relpath(os.path.join(root_dir, f), ckpt_dir)
                        ckpts.append(rel.replace("\\", "/"))
        return {"checkpoints": sorted(list(set(ckpts)))}
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


# ─────────────────────────────────────────────
# Architecture detection
# ─────────────────────────────────────────────

ARCH_CACHE = {}

def detect_architecture_from_ckpt(ckpt_path: str) -> dict:
    """Detect architecture and required tokenizer from checkpoint weights."""
    import torch
    import gc

    result = {"arch": "F5TTS_Base", "tokenizer": "pinyin"}

    if not ckpt_path or not os.path.exists(ckpt_path):
        return result

    try:
        mtime = os.path.getmtime(ckpt_path)
        if ckpt_path in ARCH_CACHE and ARCH_CACHE[ckpt_path]["mtime"] == mtime:
            log_inference(f"Using cached architecture detection for {ckpt_path}")
            return ARCH_CACHE[ckpt_path]["result"]
    except Exception:
        pass

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get("ema_model_state_dict", checkpoint),
        )

        for key in state_dict.keys():
            if "transformer_blocks.0.attn_norm.linear.weight" in key:
                dim = state_dict[key].shape[0] // 6
                result["arch"] = "F5TTS_Personal" if dim == 768 else "F5TTS_Base"
                break

        for key in state_dict.keys():
            if "text_embed.text_embed.weight" in key:
                vocab_size = state_dict[key].shape[0]
                if vocab_size == 2546:
                    result["tokenizer"] = "pinyin"
                elif vocab_size == 257:
                    result["tokenizer"] = "byte"
                else:
                    result["tokenizer"] = "custom"
                break

        del state_dict
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()

        ARCH_CACHE[ckpt_path] = {"mtime": os.path.getmtime(ckpt_path), "result": result}

    except Exception as e:
        log_inference(f"Checkpoint detection failed: {e}")

    return result


# ─────────────────────────────────────────────
# Inference endpoints
# ─────────────────────────────────────────────

@router.post("/generate")
async def generate_voice(config: GenerateConfig, background_tasks: BackgroundTasks):
    reset_inference_log()
    log_inference(f"Generation request: {config.json()}")

    with open(INFER_STATUS_FILE, "w") as f:
        f.write("running")

    # ── Model resolution ──────────────────────
    selected_model_id = config.model or "F5TTS-Base"
    model_meta = MODELS.get(selected_model_id, next(iter(MODELS.values())) if MODELS else None)

    if not model_meta:
        log_inference("ERROR: No valid model configuration found.")
        with open(INFER_STATUS_FILE, "w") as f:
            f.write("failed")
        return {"status": "Error", "warning": "Server configuration missing model registry."}

    # ── Checkpoint resolution ─────────────────
    ckpt_path = None
    ckpt_parent = model_meta["ckpt_parent"]

    if config.checkpoint:
        for m in MODELS.values():
            candidate = os.path.join(m["ckpt_parent"], config.checkpoint)
            if os.path.exists(candidate):
                ckpt_path = candidate
                break
        if not ckpt_path and os.path.exists(config.checkpoint):
            ckpt_path = config.checkpoint

    if ckpt_path is None:
        ckpt_path = _find_latest_checkpoint(ckpt_parent)
        log_inference(f"Using auto-detected latest checkpoint: {ckpt_path}")

    # ── Tokenizer / arch detection ────────────
    if ckpt_path:
        detected = detect_architecture_from_ckpt(ckpt_path)
        detected_tokenizer = detected["tokenizer"]
        log_inference(
            f"Auto-detected: arch={detected['arch']}, tokenizer={detected_tokenizer} from {ckpt_path}"
        )
    else:
        detected_tokenizer = "pinyin"
        log_inference("No checkpoint found – using default tokenizer: pinyin")

    # ── Reference audio ───────────────────────
    ref_audio_path = None
    if config.ref_audio and config.dataset:
        candidate = os.path.join(DATASET_DIR, config.dataset, config.ref_audio)
        if os.path.exists(candidate):
            ref_audio_path = candidate
    elif config.ref_audio and os.path.exists(config.ref_audio):
        ref_audio_path = config.ref_audio

    # ── Output path ───────────────────────────
    out_filename = f"output_{uuid.uuid4().hex[:8]}.wav"
    out_path = os.path.join(INFER_DIR, out_filename)

    # ── CLI command ───────────────────────────
    personal_infer_script = model_meta.get("infer_script")
    cli_cmd = [
        sys.executable, personal_infer_script,
        "--gen_text", config.text,
        "--output_dir", INFER_DIR,
        "--output_file", out_filename,
        "--tokenizer", detected_tokenizer,
        "--device", "cuda",
    ]
    if ckpt_path:
        cli_cmd += ["--ckpt_file", ckpt_path]
    if ref_audio_path:
        cli_cmd += ["--ref_audio", ref_audio_path]
        if config.ref_text:
            cli_cmd += ["--ref_text", config.ref_text]

    # ── Background task ───────────────────────
    def run_inference():
        env = os.environ.copy()
        env["PYTHONPATH"] = model_meta["src"] + (
            os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else ""
        )

        with open(INFER_LOG_FILE, "a", encoding="utf-8") as log_f:
            try:
                log_f.write("[INFO] Launching inference subprocess...\n")
                log_f.flush()

                process = subprocess.Popen(
                    cli_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                )
                for line in process.stdout:
                    log_f.write(line)
                    log_f.flush()
                process.wait()

                if process.returncode == 0 and os.path.exists(out_path):
                    log_f.write(f"[SUCCESS] Audio saved: {out_filename}\n")
                    with open(INFER_STATUS_FILE, "w") as sf:
                        sf.write(f"done:{out_filename}")
                else:
                    log_f.write(f"[ERROR] Inference failed (exit code {process.returncode})\n")
                    with open(INFER_STATUS_FILE, "w") as sf:
                        sf.write("failed")
            except Exception as e:
                log_f.write(f"[CRITICAL] Runtime error: {e}\n")
                with open(INFER_STATUS_FILE, "w") as sf:
                    sf.write("failed")

    background_tasks.add_task(run_inference)
    return {
        "status": "Processing",
        "audio_url": f"/api/infer/audio/{out_filename}",
        "filename": out_filename,
    }


@router.get("/infer_status")
async def get_infer_status():
    """Returns current inference status: idle | running | done | failed."""
    try:
        if not os.path.exists(INFER_STATUS_FILE):
            return {"status": "idle", "filename": None, "audio_url": None}
        with open(INFER_STATUS_FILE, "r") as f:
            raw = f.read().strip()
        if raw.startswith("done:"):
            filename = raw.split(":", 1)[1]
            return {
                "status": "done",
                "filename": filename,
                "audio_url": f"/api/infer/audio/{filename}",
            }
        return {"status": raw, "filename": None, "audio_url": None}
    except Exception:
        return {"status": "error", "filename": None, "audio_url": None}


@router.get("/infer_stream")
async def infer_stream():
    """Server-Sent Events stream of the inference log file."""

    async def log_generator():
        if not os.path.exists(INFER_LOG_FILE):
            yield "data: Waiting for inference to start...\n\n"

        with open(INFER_LOG_FILE, "r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line.rstrip()}\n\n"
                else:
                    # Check if still running
                    try:
                        if os.path.exists(INFER_STATUS_FILE):
                            with open(INFER_STATUS_FILE, "r") as sf:
                                status = sf.read().strip()
                            if status != "running":
                                break
                    except Exception:
                        break
                    await asyncio.sleep(0.4)

    return StreamingResponse(log_generator(), media_type="text/event-stream")


# ─────────────────────────────────────────────
# Audio serving
# ─────────────────────────────────────────────

@router.get("/audio/{filename}")
async def get_audio(filename: str):
    file_path = os.path.join(INFER_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Audio file not found")
