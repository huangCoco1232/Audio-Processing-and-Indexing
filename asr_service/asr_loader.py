# asr_service/asr_loader.py

import torch
import whisper

print("Loading Whisper ASR model...")

# 你可选 tiny / base / small / medium / large-v3
MODEL_NAME = "small"

model = whisper.load_model(MODEL_NAME)

print(f"Whisper ASR model loaded, model = {MODEL_NAME}")