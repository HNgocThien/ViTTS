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

def calculate_similarity(ref_audio_path, gen_audio_path, similarity_model, device):
    """Calculate Speaker Similarity (SIM-O) using ECAPA-TDNN."""
    if not similarity_model or not os.path.exists(ref_audio_path) or not os.path.exists(gen_audio_path):
        return None
    try:
        import torchaudio
        import torch.nn.functional as F
        from pydub import AudioSegment
        import io
        
        def load_audio(path):
            try:
                return torchaudio.load(str(path))
            except Exception:
                audio = AudioSegment.from_file(str(path))
                wav_io = io.BytesIO()
                audio.export(wav_io, format="wav")
                wav_io.seek(0)
                return torchaudio.load(wav_io)

        # Load audio files
        wav1, sr1 = load_audio(gen_audio_path)
        wav2, sr2 = load_audio(ref_audio_path)
        
        # Move to device and resample to 16kHz if necessary (SpeechBrain ECAPA-TDNN expects 16kHz)
        if sr1 != 16000:
            resampler = torchaudio.transforms.Resample(sr1, 16000).to(device)
            wav1 = resampler(wav1.to(device))
        else:
            wav1 = wav1.to(device)
            
        if sr2 != 16000:
            resampler = torchaudio.transforms.Resample(sr2, 16000).to(device)
            wav2 = resampler(wav2.to(device))
        else:
            wav2 = wav2.to(device)
            
        # Ensure single channel (mono)
        if wav1.shape[0] > 1:
            wav1 = wav1.mean(dim=0, keepdim=True)
        if wav2.shape[0] > 1:
            wav2 = wav2.mean(dim=0, keepdim=True)
            
        # Extract embeddings
        with torch.no_grad():
            emb1 = similarity_model.encode_batch(wav1.squeeze(0).unsqueeze(0))
            emb2 = similarity_model.encode_batch(wav2.squeeze(0).unsqueeze(0))
            
            # Squeeze to 1D vectors
            emb1 = emb1.squeeze()
            emb2 = emb2.squeeze()
            
            # Calculate cosine similarity
            similarity = F.cosine_similarity(emb1, emb2, dim=0).item()
            
        return similarity
    except Exception as e:
        print(f"Error calculating speaker similarity: {e}")
        return None

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
    import argparse
    import gc
    parser = argparse.ArgumentParser(description="Evaluate F5-TTS model on Vietnamese test set")
    parser.add_argument("--ref_audio", type=str, default="../../shared_storage/datasets/thien_dataset/wavs/0001_thien_dataset.wav", help="Reference audio file")
    parser.add_argument("--ref_text", type=str, default="Hôm nay, tôi bắt đầu ghi âm cho dự án tốt nghiệp của mình.", help="Subtitle/transcript for reference audio")
    parser.add_argument("--ckpt_file", type=str, default="ckpts/vietnamese/model_last.pt", help="Path to checkpoint (local or relative to ckpts/)")
    parser.add_argument("--output_dir", type=str, default="../../shared_storage/outputs/evaluation", help="Output directory")
    parser.add_argument("--whisper_model", type=str, default="base", choices=["tiny", "base", "small", "medium", "large-v3", "turbo"], help="Whisper model size for ASR evaluation")
    parser.add_argument("--device", type=str, help="Device for F5-TTS model (cpu or cuda)")
    parser.add_argument("--eval_device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run evaluation models on (Whisper & SpeechBrain) to prevent GPU OOM")
    
    args = parser.parse_args()

    REF_AUDIO = args.ref_audio
    REF_TEXT = args.ref_text
    CKPT_FILE = args.ckpt_file
    OUTPUT_DIR = args.output_dir
    WHISPER_SIZE = args.whisper_model
    
    # Resolve Checkpoint Path
    if not os.path.exists(CKPT_FILE):
        candidate = os.path.join("ckpts", CKPT_FILE)
        if os.path.exists(candidate):
            CKPT_FILE = candidate
        else:
            print(f"Warning: Checkpoint not found at {CKPT_FILE}, trying default locations...")
            personal_ckpt = os.path.join("ckpts", "personal_tts_vn", "model_last.pt")
            base_ckpt = os.path.join("ckpts", "base", "model_1200000.pt")
            if os.path.exists(personal_ckpt):
                CKPT_FILE = personal_ckpt
            elif os.path.exists(base_ckpt):
                CKPT_FILE = base_ckpt
            else:
                print("Error: No checkpoint found. Please specify correct --ckpt_file")
                return

    # Auto-detect model type (pretrained vs fine-tuned) to separate output folder
    ckpt_name = os.path.basename(CKPT_FILE).lower()
    is_pretrained = "base" in ckpt_name or "vietnamese" in CKPT_FILE.lower() or "1200000" in ckpt_name
    
    sub_dir = "pretrained" if is_pretrained else "finetuned"
    TARGET_OUTPUT_DIR = os.path.join(OUTPUT_DIR, sub_dir)
    os.makedirs(TARGET_OUTPUT_DIR, exist_ok=True)
    
    print(f"Detected Model Type: {'Pretrained (Base)' if is_pretrained else 'Fine-tuned'}")
    print(f"All generated files and CSV tables will be saved to: {TARGET_OUTPUT_DIR}")

    # 20 Vietnamese test sentences covering the 4 categories from Chapter 4
    TEST_SENTENCES = [
        # 1. Câu ngắn / Mệnh lệnh (Command) - 5 câu
        "Xin chào, rất vui được gặp bạn hôm nay.",
        "Hãy mở cửa sổ ra cho thoáng một chút nhé.",
        "Hôm nay thời tiết thật đẹp, trời nắng và không có gió.",
        "Nhớ uống đủ nước mỗi ngày để giữ gìn sức khỏe nhé.",
        "Bạn có thể giúp tôi mang cái túi này lên lầu không?",

        # 2. Câu hỏi / Ngữ điệu / Cảm xúc (Intonation & Emotion) - 5 câu
        "Bạn đã ăn cơm chưa? Nếu chưa thì mình ra ngoài ăn cùng nhau nhé.",
        "Trời ơi, hôm nay đường phố đông quá, mình đi trễ mất thôi.",
        "Thật không thể tin được, kết quả lại tốt đến vậy, tôi rất vui.",
        "Tại sao bạn lại quyết định chọn ngành công nghệ thông tin vậy?",
        "Ôi, chiếc bánh này ngon quá, tôi chưa bao giờ ăn thứ ngon như vậy.",

        # 3. Đoạn văn dài (Long-form narration) - 5 câu
        "Mỗi buổi sáng khi thức dậy, tôi thường dành khoảng ba mươi phút để tập một vài động tác thể dục nhẹ nhàng, sau đó tự tay pha một tách cà phê nóng rồi ngồi đọc tin tức trên điện thoại trước khi chuẩn bị trang phục để đi làm, bởi vì thói quen này giúp tôi có được sự tập trung cao nhất cho cả ngày dài bận rộn.",
        "Mùa hè năm ngoái, gia đình tôi đã có một chuyến đi du lịch vô cùng đáng nhớ tại thành phố biển Nha Trang, nơi chúng tôi không chỉ được đắm mình trong làn nước biển trong xanh mát rượi mà còn được thưởng thức rất nhiều món hải sản tươi ngon độc đáo và tham gia các hoạt động vui chơi giải trí đầy thú vị trên đảo.",
        "Việc tự học một ngôn ngữ mới hay một kỹ năng công nghệ trong thời đại số hiện nay đòi hỏi mỗi người phải có tính tự giác cực kỳ cao, sự kiên trì vượt qua những khó khăn ban đầu khi gặp phải kiến thức phức tạp, cũng như việc chủ động phân bổ quỹ thời gian hợp lý hàng ngày để thực hành liên tục không ngừng nghỉ.",
        "Thành phố Hồ Chí Minh là một trung tâm kinh tế năng động và náo nhiệt bậc nhất cả nước, nơi mà dòng người qua lại luôn tấp nập từ sáng sớm cho đến đêm muộn, với những tòa nhà cao tầng hiện đại mọc lên san sát bên cạnh các khu chợ truyền thống mang đậm nét văn hóa đặc trưng lâu đời của người dân Nam Bộ.",
        "Tôi luôn duy trì thói quen đọc sách khoảng một tiếng vào mỗi buổi tối trước khi đi ngủ, bởi vì những trang sách không chỉ mang lại cho tôi nguồn tri thức vô tận về lịch sử và thế giới xung quanh, mà còn là liều thuốc tinh thần tuyệt vời giúp tôi giải tỏa mọi căng thẳng sau những giờ làm việc mệt mỏi.",

        # 4. Yếu tố phức tạp (Numbers, dates, special characters) - 5 câu
        "Hội thảo khoa học về trí tuệ nhân tạo sẽ chính thức khai mạc vào lúc 08:30 ngày 15/10/2026 tại hội trường lớn.",
        "Doanh số bán hàng của cửa hàng trực tuyến trong tháng 5 vừa qua đã đạt mức tăng trưởng kỷ lục là 18,5%.",
        "Chúng tôi mua bộ thiết bị thu âm chuyên nghiệp này với tổng chi phí là 12.450.000 đồng trên sàn thương mại điện tử.",
        "Mọi thắc mắc về tài liệu hướng dẫn xin gửi về hòm thư điện tử sv@kltn-tts.edu.vn để được ban quản trị phản hồi sớm nhất.",
        "Thời tiết Hà Nội hôm nay khá oi bức với nhiệt độ trung bình từ 35°C đến 38°C, độ ẩm đo được là 75%."
    ]


    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    eval_device = args.eval_device
    
    print(f"Loading F5-TTS models from {CKPT_FILE} on {device}...")
    vocoder = load_vocoder(device=device)
    model = load_custom_model(CKPT_FILE, device=device)
    
    # Load Whisper for WER/CER
    whisper_model = None
    try:
        import whisper
        print(f"Loading Whisper model '{WHISPER_SIZE}' for ASR evaluation on {eval_device}...")
        whisper_model = whisper.load_model(WHISPER_SIZE, device=eval_device)
    except Exception as e:
        print(f"Whisper or jiwer not installed. Skipping WER/CER. Error: {e}")
        
    # Load SpeechBrain ECAPA-TDNN for SIM-O
    similarity_model = None
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        print(f"Loading SpeechBrain ECAPA-TDNN for Speaker Similarity (SIM-O) on {eval_device}...")
        similarity_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            run_opts={"device": eval_device}
        )
    except Exception as e:
        print(f"SpeechBrain not installed or failed to load. Skipping Speaker Similarity (SIM-O). Error: {e}")
        print("To compute Speaker Similarity, install it via: pip install speechbrain")

    # Clear GPU Cache to release VRAM before starting inference
    if "cuda" in device:
        torch.cuda.empty_cache()
        gc.collect()

    results = []
    
    # Preprocess Reference
    ref_audio, ref_text = preprocess_ref_audio_text(REF_AUDIO, REF_TEXT)

    print(f"\nEvaluating on {len(TEST_SENTENCES)} sentences. Outputs will be saved to {TARGET_OUTPUT_DIR}\n" + "="*80)
    for i, text in enumerate(TEST_SENTENCES):
        print(f"\n[{i+1}/{len(TEST_SENTENCES)}] Evaluating: {text}")
        
        # Start timing
        start_time = time.time()
        
        # Inference (runs on GPU device)
        audio_segment, final_sample_rate, _ = infer_process(
            ref_audio, ref_text, text, model, vocoder,
            nfe_step=32, cfg_strength=2.0, device=device
        )
        
        latency = time.time() - start_time
        
        # Save audio
        out_filename = f"eval_{i+1}.wav"
        out_path = Path(TARGET_OUTPUT_DIR) / out_filename
        
        if audio_segment is not None:
            audio_tensor = torch.from_numpy(audio_segment).unsqueeze(0)
            torchaudio.save(str(out_path), audio_tensor, final_sample_rate)
            
            audio_duration = audio_segment.shape[0] / final_sample_rate
            # Run evaluations on eval_device (CPU by default)
            rtf, wer, cer = calculate_metrics(text, out_path, latency, audio_duration, whisper_model)
            sim_o = calculate_similarity(REF_AUDIO, out_path, similarity_model, eval_device)
            
            print(f"  > Latency: {latency:.2f}s | Audio Len: {audio_duration:.2f}s | RTF: {rtf:.4f}")
            if wer is not None:
                print(f"  > WER: {wer:.2%}, CER: {cer:.2%}")
            if sim_o is not None:
                print(f"  > Speaker Similarity (SIM-O): {sim_o:.4f}")
            
            results.append({
                "ID": i+1,
                "Text": text,
                "Audio_Path": out_filename,
                "Latency_sec": round(latency, 3),
                "Duration_sec": round(audio_duration, 3),
                "RTF": round(rtf, 4),
                "WER": wer,
                "CER": cer,
                "Similarity": sim_o
            })
            
            # Periodically clean cache to keep VRAM clean
            if "cuda" in device and i % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        else:
            print("  > Inference Failed!")

    # Save to CSV and Print Summary
    if results:
        df = pd.DataFrame(results)
        ckpt_stem = os.path.basename(CKPT_FILE).replace(".pt", "")
        csv_path = os.path.join(TARGET_OUTPUT_DIR, f"evaluation_results_{ckpt_stem}.csv")
        df.to_csv(csv_path, index=False)
        print("\n" + "="*80 + "\n[EVALUATION SUMMARY]")
        print(f"Model Checkpoint: {CKPT_FILE}")
        print(f"Mean Latency: {df['Latency_sec'].mean():.3f} s")
        print(f"Mean RTF: {df['RTF'].mean():.4f}")
        if df['WER'].notna().any():
            print(f"Mean WER: {df['WER'].mean():.2%}")
            print(f"Mean CER: {df['CER'].mean():.2%}")
        if df['Similarity'].notna().any():
            print(f"Mean Speaker Similarity (SIM-O): {df['Similarity'].mean():.4f}")
        print(f"Results table and generated audios saved to: {TARGET_OUTPUT_DIR}")
    else:
        print("No evaluation results were generated.")

if __name__ == "__main__":
    main()
