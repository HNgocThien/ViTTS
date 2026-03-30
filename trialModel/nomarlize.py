import unicodedata
import re

input_file = r"d:\THIEN_PROJECT\KLTN_TTS\mimic-recording-studio\backend\prompts\vietnamese_train.csv"
output_file = r"d:\THIEN_PROJECT\KLTN_TTS\mimic-recording-studio\backend\prompts\vietnamese_train_fixed.csv"

def normalize_text(text):
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    
    text = text.strip()
    text = " ".join(text.split())
    
    # Viết hoa đầu câu
    text = text.capitalize()
    
    # Chuẩn hóa dấu câu
    text = re.sub(r"[!]{2,}", "!", text)
    text = re.sub(r"[?]{2,}", "?", text)
    
    return text

with open(input_file, "r", encoding="utf-8") as f_in, \
     open(output_file, "w", encoding="utf-8") as f_out:
    
    for line in f_in:
        if not line.strip():
            continue
        
        parts = line.strip().split("\t")
        
        if len(parts) < 1:
            continue
        
        text = parts[0]
        
        # normalize
        text = normalize_text(text)
        
        # tính lại length chuẩn
        length = len(text)
        
        f_out.write(f"{text}\t{length}\n")

print("Done! File fixed saved as vietnamese_tts_fixed.csv")