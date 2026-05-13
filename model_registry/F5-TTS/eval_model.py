import os
import sys
import torch
import time
import pandas as pd
import numpy as np
import torchaudio
from pathlib import Path
from datetime import datetime

# Setup paths
F5_TTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if F5_TTS_PATH not in sys.path:
    sys.path.append(F5_TTS_PATH)

from f5_tts.infer.utils_infer import (
    load_vocoder,
    infer_process,
    preprocess_ref_audio_text,
    device as default_device
)
from infer_personal_TTS import load_custom_model

def calculate_metrics(gen_text, audio_path, latency, audio_duration, whisper_model=None):
    """Calculate WER and CER using Whisper ASR."""
    rtf = latency / audio_duration if audio_duration > 0 else 0
    wer, cer = None, None
    
    if whisper_model and os.path.exists(audio_path):
        try:
            from jiwer import wer as calc_wer, cer as calc_cer
            # ASR Transcription
            result = whisper_model.transcribe(str(audio_path), language="vi")
            pred_text = result["text"].strip().lower()
            ref_text = gen_text.strip().lower()
            
            wer = calc_wer(ref_text, pred_text)
            cer = calc_cer(ref_text, pred_text)
        except Exception as e:
            print(f"Error calculating WER/CER: {e}")
            
    return rtf, wer, cer

def main():
    # --- CONFIGURATION ---
    REF_AUDIO = "../../shared_storage/datasets/thien_dataset/wavs/0001_thien_dataset.wav"
    REF_TEXT = "Hôm nay, tôi bắt đầu ghi âm cho dự án tốt nghiệp của mình."
    CKPT_FILE = "ckpts/vietnamese/model_last.pt"
    OUTPUT_DIR = "../../shared_storage/outputs/evaluation"
    TEST_SENTENCES = [
        "Xin chào, đây là bài kiểm tra đánh giá chất lượng mô hình tiếng Việt.",
        "Hệ thống chuyển đổi văn bản thành giọng nói hoạt động rất ổn định.",
        "Trí tuệ nhân tạo đang thay đổi cách chúng ta tương tác với máy tính hàng ngày.",
        "Nhiệt độ hôm nay tại Hà Nội là ba mươi hai độ xê, trời nắng nóng.",
        "Bạn có nghe rõ giọng nói của tôi không? Tôi hy vọng kết quả sẽ tốt."
    ]
    # ---------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading models on {device}...")
    vocoder = load_vocoder(device=device)
    model = load_custom_model(CKPT_FILE, device=device)
    
    # Load Whisper for WER/CER if possible
    whisper_model = None
    try:
        import whisper
        print("Loading Whisper for WER/CER evaluation...")
        whisper_model = whisper.load_model("base", device=device)
    except ImportError:
        print("Whisper or jiwer not installed. Skipping WER/CER.")

    results = []
    
    # Preprocess Reference
    ref_audio, ref_text = preprocess_ref_audio_text(REF_AUDIO, REF_TEXT)

    for i, text in enumerate(TEST_SENTENCES):
        print(f"\n[{i+1}/{len(TEST_SENTENCES)}] Evaluating: {text[:50]}...")
        
        # Start timing
        start_time = time.time()
        
        # Inference
        audio_segment, final_sample_rate, _ = infer_process(
            ref_audio, ref_text, text, model, vocoder,
            nfe_step=32, cfg_strength=2.0, device=device
        )
        
        latency = time.time() - start_time
        
        # Save audio to calculate duration and for MOS
        out_filename = f"eval_{i+1}.wav"
        out_path = Path(OUTPUT_DIR) / out_filename
        
        if audio_segment is not None:
            audio_tensor = torch.from_numpy(audio_segment).unsqueeze(0)
            torchaudio.save(str(out_path), audio_tensor, final_sample_rate)
            
            audio_duration = audio_segment.shape[0] / final_sample_rate
            rtf, wer, cer = calculate_metrics(text, out_path, latency, audio_duration, whisper_model)
            
            print(f"  > Latency: {latency:.2f}s | Audio Len: {audio_duration:.2f}s | RTF: {rtf:.4f}")
            if wer is not None:
                print(f"  > WER: {wer:.2%}, CER: {cer:.2%}")
            
            results.append({
                "ID": i+1,
                "Text": text,
                "Audio_Path": out_filename,
                "Latency_sec": round(latency, 3),
                "Duration_sec": round(audio_duration, 3),
                "RTF": round(rtf, 4),
                "WER": wer,
                "CER": cer
            })
        else:
            print("  > Inference Failed!")

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[DONE] Results saved to {csv_path}")

if __name__ == "__main__":
    main()
