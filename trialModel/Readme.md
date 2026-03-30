# Training F5-TTS on Personal Data

The goal is to create a Python script `training_personal_TTS.py` that utilizes the F5-TTS `Trainer` from `trainer.py` to train or finetune a model using audio data from Mimic Recording Studio.

## User Review Required

> [!IMPORTANT]
> The training data provided is very small (only 1 audio file). Training from scratch will not yield good results. This script is best used for **finetuning** an existing model or as a starting point for a larger dataset.

> [!NOTE]
> I will use the **byte-level tokenizer** (UTF-8) by default to support Vietnamese characters without needing a pre-defined vocabulary file.

## Proposed Changes

### [NEW] [training_personal_TTS.py](file:///d:/THIEN_PROJECT/KLTN_TTS/training_personal_TTS.py)

This script will:
1.  Setup the environment by adding `F5-TTS/src` to the Python path.
2.  Define `MimicDataset`: A PyTorch Dataset that parses the metadata file format `audio_file|text|duration`.
3.  Initialize the **DiT (Diffusion Transformer)** and **CFM (Conditional Flow Matching)** model architectures.
4.  Configure the **Trainer** with hyperparameters like learning rate, batch size, and checkpoint paths.
5.  Execute the training loop using the `accelerate` backend.

## Open Questions

1.  **Finetuning vs. Training from Scratch**: Do you have a pretrained checkpoint you want to start from? If so, please provide the path.
2.  **Hardware**: Is this intended to run on a GPU? The F5-TTS model is computationally intensive.

## Verification Plan

### Manual Verification
1.  Run the script: `python training_personal_TTS.py`.
2.  Verify that it successfully loads the data and starts the training updates (showing the progress bar).
3.  Check that checkpoints are being saved in the specified directory.
