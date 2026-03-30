import os
import sys
import torch
import argparse
from torch.utils.data import Dataset

# 1. Setup paths to include F5-TTS source
F5_TTS_PATH = os.path.join(os.getcwd(), "F5-TTS", "src")
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
                        learning_rate=7.5e-5, num_warmup_updates=500):
    print("Initializing configuration...")
    # Configuration
    DATA_DIR = data_dir
    METADATA_FILE = metadata_file
    CHECKPOINT_DIR = checkpoint_dir
    
    # Model Hyperparameters (Base Configuration)
    model_cfg = {
        "dim": 1024,
        "depth": 22,
        "heads": 16,
        "ff_mult": 2,
        "text_dim": 512,
        "conv_layers": 4
    }
    mel_cfg = {
        "target_sample_rate": 24000,
        "n_mel_channels": 100,
        "hop_length": 256,
        "win_length": 1024,
        "n_fft": 1024
    }

    # 1. Setup Tokenizer (Byte-level for UTF-8 support)
    print("Setting up tokenizer...")
    vocab_char_map, vocab_size = get_tokenizer(None, tokenizer="byte")

    # 2. Initialize Model (DiT + CFM)
    print("Initializing model...")
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
        save_per_updates=1000,
        checkpoint_path=CHECKPOINT_DIR,
        batch_size_per_gpu=batch_size,
        batch_size_type="sample",
        max_grad_norm=1.0,
        logger=None  # Set to "wandb" or "tensorboard" if configured
    )

    # 4. Load Dataset
    print("Loading dataset...")
    dataset = MimicDataset(DATA_DIR, METADATA_FILE, **mel_cfg)
    
    print(f"Dataset loaded with {len(dataset)} samples.")
    
    # 5. Start Training
    print("Starting training...")
    trainer.train(dataset, num_workers=0) # Use 0 workers for simpler debugging on Windows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F5-TTS Personal Training Script")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing audio files and metadata")
    parser.add_argument("--metadata_file", type=str, required=True, help="Name of the metadata file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--checkpoint_dir", type=str, default="/app/ckpts/personal_tts_vn", help="Directory to save checkpoints")
    parser.add_argument("--learning_rate", type=float, default=7.5e-5, help="Optimizer learning rate")
    parser.add_argument("--num_warmup_updates", type=int, default=500, help="Number of warmup update steps")

    args = parser.parse_args()

    train_personal_data(
        data_dir=args.data_dir,
        metadata_file=args.metadata_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        learning_rate=args.learning_rate,
        num_warmup_updates=args.num_warmup_updates,
    )
