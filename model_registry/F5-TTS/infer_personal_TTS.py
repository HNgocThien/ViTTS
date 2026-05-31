import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# 1. Setup paths to include F5-TTS source
F5_TTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if F5_TTS_PATH not in sys.path:
    sys.path.append(F5_TTS_PATH)

from f5_tts.model.backbones.dit import DiT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_checkpoint,
    infer_process,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    device as default_device
)
from f5_tts.model import CFM
from f5_tts.model.utils import get_tokenizer

def detect_configuration(ckpt_path):
    """Detect architecture and tokenizer from checkpoint weights."""
    import gc
    print(f"Detecting configuration from {ckpt_path}...")
    
    # Defaults to V0 (F5TTS_Base) config
    arch = "f5tts" # maps to DiT
    tokenizer_type = "pinyin"
    model_cfg = {
        "dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2,
        "text_dim": 512, "conv_layers": 4,
        "text_mask_padding": False, "pe_attn_head": 1
    }

    if not ckpt_path or not os.path.exists(ckpt_path):
        return arch, tokenizer_type, model_cfg

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("ema_model_state_dict", checkpoint))
        
        # 1. Detect Architecture (Base vs Lite/Others)
        for key in state_dict.keys():
            if "attn_norm.linear.weight" in key or "norm.linear.weight" in key:
                if "transformer_blocks" in key: # DiT
                    arch = "f5tts"
                    shape = state_dict[key].shape[0]
                    dim = shape // 6
                    if dim == 768:
                        model_cfg["dim"] = 768
                    elif dim == 1024:
                        model_cfg["dim"] = 1024
                break
        
        # 2. Detect Tokenizer
        for key in state_dict.keys():
            if "text_embed.text_embed.weight" in key:
                vocab_size = state_dict[key].shape[0]
                if vocab_size == 257:
                    tokenizer_type = "byte"
                elif vocab_size == 2546:
                    tokenizer_type = "pinyin"
                else:
                    tokenizer_type = "custom"
                break

        del state_dict
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()
                
    except Exception as e:
        print(f"Warning: Configuration detection failed: {e}. Using defaults.")
        
    return arch, tokenizer_type, model_cfg

def load_custom_model(ckpt_path, vocab_file="", tokenizer="auto", device=default_device):
    """Refined model loader that handles personal DiT config."""
    from importlib.resources import files
    
    arch_type, detected_tokenizer, model_cfg = detect_configuration(ckpt_path)
    
    # Use detected tokenizer if user didn't explicitly override with something non-default
    if tokenizer == "auto":
        tokenizer = detected_tokenizer

    # Auto-detect vocab file in the checkpoint directory if not provided and using custom tokenizer
    if not vocab_file and tokenizer == "custom" and ckpt_path:
        candidate_vocab = os.path.join(os.path.dirname(ckpt_path), "vocab.txt")
        if os.path.exists(candidate_vocab):
            vocab_file = candidate_vocab
            print(f"Auto-detected vocab file in checkpoint directory: {vocab_file}")

    # CRITICAL FIX for base model inference:
    # F5-TTS expects tokenizer="custom" and a direct path to vocab.txt for pre-trained models
    # that use the 2546-length vocabulary.
    if tokenizer == "pinyin":
        tokenizer = "custom"
        if not vocab_file:
            vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))

    print(f"Initializing model: arch={arch_type}, tokenizer={tokenizer}, device={device}")
    
    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)
    
    transformer = DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100)

    model = CFM(
        transformer=transformer,
        mel_spec_kwargs=dict(
            n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=100,
            target_sample_rate=24000, mel_spec_type="vocos"
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    model = load_checkpoint(model, ckpt_path, device, use_ema=True)
    return model

def main():
    parser = argparse.ArgumentParser(description="F5-TTS Personal Inference Script")
    parser.add_argument("--ref_audio", type=str, required=True, help="Reference audio file")
    parser.add_argument("--ref_text", type=str, default="", help="Subtitle/transcript for reference audio")
    parser.add_argument("--gen_text", type=str, required=True, help="Text to generate")
    parser.add_argument("--ckpt_file", type=str, help="Path to checkpoint (local or relative to ckpts/)")
    parser.add_argument("--vocab_file", type=str, help="Path to vocab.txt file")
    parser.add_argument("--tokenizer", type=str, default="auto", choices=["auto", "byte", "pinyin", "custom"], help="Tokenizer type")
    parser.add_argument("--output_dir", type=str, default="tests", help="Output directory")
    parser.add_argument("--output_file", type=str, help="Output filename")
    parser.add_argument("--remove_silence", action="store_true", help="Remove silence from output")
    parser.add_argument("--vocoder_name", type=str, default="vocos", choices=["vocos", "bigvgan"], help="Vocoder type")
    parser.add_argument("--nfe_step", type=int, default=32, help="Number of denoising steps")
    parser.add_argument("--cfg_strength", type=float, default=2.0, help="CFG strength")
    parser.add_argument("--device", type=str, default=default_device, help="Device to run on")

    args = parser.parse_args()

    # 1. Resolve Checkpoint Path
    ckpt_path = args.ckpt_file
    if not ckpt_path:
        # Check personal dir first
        personal_ckpt = os.path.join("ckpts", "personal_tts_vn", "model_last.pt")
        base_ckpt = os.path.join("ckpts", "base", "model_1200000.pt")
        if os.path.exists(personal_ckpt):
            ckpt_path = personal_ckpt
        elif os.path.exists(base_ckpt):
            ckpt_path = base_ckpt
        else:
            print("Error: No checkpoint found in default locations. Please specify --ckpt_file")
            return

    if not os.path.exists(ckpt_path):
        # Try relative to ckpts/ if not found
        candidate = os.path.join("ckpts", ckpt_path)
        if os.path.exists(candidate):
            ckpt_path = candidate
        else:
            print(f"Error: Checkpoint not found at {ckpt_path}")
            return

    # 2. Setup Output
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = args.output_file or f"personal_tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    out_path = Path(args.output_dir) / out_file

    # 3. Load Model & Vocoder
    # Auto-detect vocab if not provided
    vocab_file = args.vocab_file
    if not vocab_file:
        candidate_vocab = os.path.join(os.path.dirname(ckpt_path), "vocab.txt")
        if os.path.exists(candidate_vocab):
            vocab_file = candidate_vocab
            print(f"Auto-detected vocab file at: {vocab_file}")

    vocoder = load_vocoder(vocoder_name=args.vocoder_name, device=args.device)
    model = load_custom_model(ckpt_path, vocab_file=vocab_file, tokenizer=args.tokenizer, device=args.device)

    # 4. Preprocess Reference
    print(f"Processing reference: {args.ref_audio}")
    ref_audio, ref_text = preprocess_ref_audio_text(args.ref_audio, args.ref_text)

    # 5. Inference
    print(f"Generating audio for: {args.gen_text}")
    audio_segment, final_sample_rate, _ = infer_process(
        ref_audio, ref_text, args.gen_text, model, vocoder,
        nfe_step=args.nfe_step, cfg_strength=args.cfg_strength, device=args.device
    )

    # 6. Save
    if audio_segment is not None:
        import torchaudio
        # audio_segment is numpy array, need to convert to torch tensor [channels, samples]
        audio_tensor = torch.from_numpy(audio_segment).unsqueeze(0)
        torchaudio.save(str(out_path), audio_tensor, final_sample_rate)
        
        if args.remove_silence:
            remove_silence_for_generated_wav(str(out_path))
        print(f"Success! Audio saved to: {out_path}")
    else:
        print("Error: Inference failed to generate audio.")

if __name__ == "__main__":
    main()
