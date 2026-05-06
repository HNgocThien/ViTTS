from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
import os
import csv
from config_loader import DATASET_DIR, API_CONFIG, PROMPTS_CSV

router = APIRouter()

def _load_prompts():
    prompts = []
    if os.path.exists(PROMPTS_CSV):
        with open(PROMPTS_CSV, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts:
                    prompts.append(parts[0])
    return prompts

PROMPTS = _load_prompts()

@router.get("/datasets")
async def list_datasets():
    """List all available datasets in the dataset directory."""
    if not os.path.exists(DATASET_DIR):
        return {"datasets": []}
    
    datasets = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    return {"datasets": datasets}

@router.get("/prompt")
async def get_prompt(uuid: str, id: Optional[int] = None):
    """Get a specific prompt or the next unrecorded prompt if id is None."""
    user_dataset_dir = os.path.join(DATASET_DIR, uuid)
    wav_dir = os.path.join(user_dataset_dir, "wavs")
    
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir, exist_ok=True)
    
    # Calculate default next prompt
    completed_count = len([f for f in os.listdir(wav_dir) if f.endswith(".wav")])
    
    # Determine target ID
    target_id = id if id is not None else (completed_count + 1)
    
    if target_id < 1:
        target_id = 1
        
    if target_id > len(PROMPTS):
        return {"success": False, "message": "Index out of range"}
        
    next_prompt = PROMPTS[target_id - 1]
    
    # Check if audio already exists for this prompt
    wav_filename = f"{target_id:04d}_{uuid}.wav"
    wav_path = os.path.join(wav_dir, wav_filename)
    audio_url = None
    if os.path.exists(wav_path):
        # Return path relative to INFER_URL
        audio_url = f"/datasets/{uuid}/audio/wavs/{wav_filename}"
        
    return {
        "success": True, 
        "data": {
            "prompt": next_prompt, 
            "prompt_id": target_id, 
            "total": len(PROMPTS),
            "audio_url": audio_url
        }
    }

@router.post("/audio")
async def upload_audio(uuid: str = Form(...), prompt: str = Form(...), prompt_id: int = Form(...), audio: UploadFile = File(...)):
    """Save the uploaded audio direttamente into the training-ready dataset structure."""
    user_dataset_dir = os.path.join(DATASET_DIR, uuid)
    wav_dir = os.path.join(user_dataset_dir, "wavs")
    metadata_file = os.path.join(user_dataset_dir, f"{uuid}-metadata.txt")
    
    os.makedirs(wav_dir, exist_ok=True)
    
    # Sanitize prefix padded number
    wav_filename = f"{prompt_id:04d}_{uuid}.wav"
    wav_path = os.path.join(wav_dir, wav_filename)
    
    content = await audio.read()
    with open(wav_path, "wb") as f:
        f.write(content)
        
    # Update metadata: remove existing entry for this wav if it exists, then append
    new_entry = f"wavs/{wav_filename}|{prompt}\n"
    lines = []
    if os.path.exists(metadata_file):
        with open(metadata_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    
    # Filter out any existing line for this wav_filename
    lines = [line for line in lines if not line.startswith(f"wavs/{wav_filename}|")]
    lines.append(new_entry)
    
    with open(metadata_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return {"success": True, "message": "Audio saved successfully"}

@router.post("/undo")
async def undo_last(uuid: str = Form(...)):
    """Remove the last recorded audio file and its metadata entry."""
    user_dataset_dir = os.path.join(DATASET_DIR, uuid)
    wav_dir = os.path.join(user_dataset_dir, "wavs")
    metadata_file = os.path.join(user_dataset_dir, f"{uuid}-metadata.txt")
    
    if not os.path.exists(metadata_file):
        return {"success": False, "message": "No recordings to undo"}
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if not lines:
        return {"success": False, "message": "No recordings to undo"}
    
    # Remove the last line
    last_line = lines.pop()
    
    # Identify wav file to delete
    # Format: wavs/0001_userid.wav|text
    try:
        wav_rel_path = last_line.split("|")[0]
        wav_path = os.path.join(user_dataset_dir, wav_rel_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)
    except Exception as e:
        print(f"Error deleting wav file: {e}")
        
    # Write back the remaining metadata
    with open(metadata_file, "w", encoding="utf-8") as f:
        f.writelines(lines)
        
    return {"success": True, "message": "Last recording removed"}
