import os
import sys
import torch
import argparse
from torch.utils.data import Dataset

# 1. Setup paths to include F5-TTS source
F5_TTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if F5_TTS_PATH not in sys.path:
    sys.path.append(F5_TTS_PATH)

from f5_tts.model import CFM, Trainer
from f5_tts.model.backbones.dit import DiT
from f5_tts.model.utils import get_tokenizer, list_str_to_tensor
from f5_tts.model.modules import MelSpec

# 2. Define Custom Dataset for Mimic Recording Studio data
class MimicDataset(Dataset):
    def __init__(self, data_dir, metadata_file, target_sample_rate=24000, n_mel_channels=100, **mel_spec_kwargs):
        self.data_dir = data_dir
        self.samples = []
        
        # Load metadata
        metadata_path = os.path.join(data_dir, metadata_file)
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if "|" in line:
                    parts = line.strip().split("|")
                    if len(parts) >= 2:
                        audio_file = parts[0]
                        text = parts[1]
                        duration = float(parts[2]) if len(parts) > 2 else 0.0
                        self.samples.append({
                            "audio_path": os.path.join(data_dir, audio_file),
                            "text": text,
                            "duration": duration
                        })
        
        # Setup Mel Spectrogram
        self.mel_spectrogram = MelSpec(
            target_sample_rate=target_sample_rate,
            n_mel_channels=n_mel_channels,
            **mel_spec_kwargs
        )
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.samples)

    def get_frame_len(self, index):
        # Estimated frame length for dynamic batching
        return self.samples[index]["duration"] * self.target_sample_rate / 256

    def __getitem__(self, index):
        sample = self.samples[index]
        audio_path = sample["audio_path"]
        text = sample["text"]

        import torchaudio
        audio, sr = torchaudio.load(audio_path)
        
        # Mix to mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            
        # Resample
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            audio = resampler(audio)
            
        # Compute Mel
        mel_spec = self.mel_spectrogram(audio).squeeze(0) # [n_mel, n_frames]
        
        return {
            "mel_spec": mel_spec,
            "text": text
        }

def train_personal_data(data_dir, metadata_file, epochs, batch_size, checkpoint_dir,
                        learning_rate=7.5e-5, num_warmup_updates=500, vocab_file=None):
    print("Initializing configuration...")
    # Configuration
    DATA_DIR = data_dir
    METADATA_FILE = metadata_file
    CHECKPOINT_DIR = checkpoint_dir
    
    # Model Hyperparameters (Official V0 Base Configuration)
    model_cfg = {
        "dim": 1024,  
        "depth": 22, 
        "heads": 16, 
        "ff_mult": 2,
        "text_dim": 512,
        "conv_layers": 4,
        "text_mask_padding": False, # 🔥 CRITICAL: Must be false for V0 Base backward-compatibility
        "pe_attn_head": 1,          # 🔥 CRITICAL: Needed for V0 architecture
        "checkpoint_activations": True  # Saves VRAM by recomputing activations
    }
    mel_cfg = {
        "target_sample_rate": 24000,
        "n_mel_channels": 100,
        "hop_length": 256,
        "win_length": 1024,
        "n_fft": 1024
    }

    # 1. Setup Tokenizer
    print("Setting up tokenizer...")
    if vocab_file and os.path.exists(vocab_file):
        # Automatically copy vocab to checkpoint directory for portability
        import shutil
        os.makedirs(checkpoint_dir, exist_ok=True)
        dest_vocab = os.path.join(checkpoint_dir, "vocab.txt")
        if not os.path.exists(dest_vocab) or not os.path.samefile(vocab_file, dest_vocab):
            shutil.copy(vocab_file, dest_vocab)
            print(f"Copied vocab file to checkpoint directory: {checkpoint_dir}")

        print(f"Using custom vocab file: {vocab_file}")
        vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer="custom")
    else:
        print("Using default byte tokenizer (UTF-8)")
        vocab_char_map, vocab_size = get_tokenizer(None, tokenizer="byte")

    # 2. Initialize Model (DiT + CFM)
    print(f"Initializing model backend: DiT...")
    transformer = DiT(
        **model_cfg,
        text_num_embeds=vocab_size,
        mel_dim=mel_cfg["n_mel_channels"]
    )
    
    model = CFM(
        transformer=transformer,
        mel_spec_kwargs=mel_cfg,
        vocab_char_map=vocab_char_map
    )

    # 3. Initialize Trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model,
        epochs=epochs,
        learning_rate=learning_rate,
        num_warmup_updates=num_warmup_updates,
        save_per_updates=500,  # Save more frequently for safety
        checkpoint_path=CHECKPOINT_DIR,
        batch_size_per_gpu=1,  # 🔥 CRITICAL: 4GB VRAM limit
        grad_accumulation_steps=8, # Effective batch size = 8
        batch_size_type="sample",
        max_grad_norm=1.0,
        bnb_optimizer=True,  # 🔥 CRITICAL: Uses 8-bit AdamW to save VRAM
        logger=None  
    )

    # 4. Load Dataset
    print("Loading dataset...")
    dataset = MimicDataset(DATA_DIR, METADATA_FILE, **mel_cfg)
    
    print(f"Dataset loaded with {len(dataset)} samples.")
    
    # 5. Handle Pretrained Weight Loading
    if not os.path.exists(os.path.join(CHECKPOINT_DIR, "model_last.pt")):
        # Path resolution for base model: check --pretrained_checkpoint or default
        base_weight_path = getattr(args, 'pretrained_checkpoint', None)
        if not base_weight_path:
            # Mặc định dùng model tiếng Việt trong thư mục ckpts/vietnamese/
            base_weight_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts", "vietnamese", "model_last.pt")
        
        # Tự động tải nếu không tồn tại
        if not os.path.exists(base_weight_path):
            print(f"Pretrained weight not found at {base_weight_path}")
            print("Downloading Vietnamese pretrained model from Hugging Face...")
            os.makedirs(os.path.dirname(base_weight_path), exist_ok=True)
            
            import urllib.request
            url = "https://huggingface.co/hynt/F5-TTS-Vietnamese-ViVoice/resolve/main/model_last.pt"
            try:
                def progress(count, block_size, total_size):
                    percent = int(count * block_size * 100 / total_size)
                    sys.stdout.write(f"\rDownloading: {percent}%")
                    sys.stdout.flush()
                
                urllib.request.urlretrieve(url, base_weight_path, reporthook=progress)
                print("\nDownload completed successfully.")
            except Exception as e:
                print(f"\nFailed to download: {e}")

        if os.path.exists(base_weight_path):
            print(f"PREPARING TO FINE-TUNE FROM: {base_weight_path}")
            
            # Surgery Loading: Handle potential size mismatch (e.g. pinyin vocab vs byte vocab)
            print("Performing 'Surgery' on checkpoint (loading matching layers only)...")
            checkpoint = torch.load(base_weight_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("ema_model_state_dict", checkpoint))
            
            # Filter state_dict
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            skipped_layers = []
            
            for k, v in state_dict.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        skipped_layers.append(k)
                else:
                    # Handle ema_model prefix if necessary
                    k_clean = k.replace("ema_model.", "")
                    if k_clean in model_state_dict:
                         if v.shape == model_state_dict[k_clean].shape:
                            filtered_state_dict[k_clean] = v
                         else:
                            skipped_layers.append(k_clean)

            if skipped_layers:
                print(f"NOTICE: Skipped matching {len(skipped_layers)} layers due to shape mismatch (e.g. embeddings): {skipped_layers[:3]}...")
            
            model.load_state_dict(filtered_state_dict, strict=False)
            print(f"Successfully loaded {len(filtered_state_dict)} layers from pretrained weights.")
            
            # Note: We don't copy the file to the checkpoint dir because we already manually loaded it.
            # The Trainer will save the first local checkpoint normally.
        else:
            print(f"WARNING: Pretrained weights not found at {base_weight_path}. Starting from scratch!")

    # 6. Start Training
    print("Starting fine-tuning...")
    trainer.train(dataset, num_workers=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F5-TTS Personal Training Script")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing audio files and metadata")
    parser.add_argument("--metadata_file", type=str, required=True, help="Name of the metadata file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--checkpoint_dir", type=str, default="ckpts/vietnamese", help="Directory to save checkpoints")
    parser.add_argument("--pretrained_checkpoint", type=str, help="Path to pretrained base model weights")
    parser.add_argument("--learning_rate", type=float, default=7.5e-5, help="Optimizer learning rate")
    parser.add_argument("--num_warmup_updates", type=int, default=500, help="Number of warmup update steps")
    parser.add_argument("--vocab_file", type=str, help="Path to custom vocab.txt")

    args = parser.parse_args()

    train_personal_data(
        data_dir=args.data_dir,
        metadata_file=args.metadata_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        learning_rate=args.learning_rate,
        num_warmup_updates=args.num_warmup_updates,
        vocab_file=args.vocab_file
    )
