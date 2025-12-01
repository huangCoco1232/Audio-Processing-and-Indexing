# Audio Processing andIndexing Final Project

**Project:** Voice Mimicry Dialogue System with LLM  
**Team:** Group 20 (Leiden University)

## Overview
This project implements an end-to-end conversational AI system that mimics a specific persona (e.g., Squidward). It integrates three main components:

1. **Speech Recognition (ASR):** Converts user speech to text using **Whisper**.  
2. **Language Generation (LLM):** Generates character-specific responses using **Microsoft Phi-4**.  
3. **Voice Cloning (TTS):** Synthesizes speech using **CosyVoice2** (Zero-shot inference).

---

## Project Directory Structure




---


## Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.10+ installed. Install required packages:

```bash
pip install torch transformers fastapi uvicorn streamlit whisper-openai librosa soundfile pydub requests
```

Note: You must also install CosyVoice-specific dependencies in CosyVoice/requirements.txt.

### 2. Model Preparation

LLM: The system automatically downloads microsoft/Phi-4-mini-instruct on first backend run.

TTS: You must manually download CosyVoice2-0.5B and place it in:

```bash
CosyVoice/pretrained_models/
```

### 3. Generate Reference Audio

The TTS engine requires a 16kHz WAV file for zero-shot voice cloning.

Place squidward.mp3 into:
```bash
CosyVoice/example_audio/
```

Run conversion:
```bash
python audio_convert.py
```

This generates squidward_16k_clean.wav for use by the main system.

## How to Run

You must run both Backend and Frontend in separate terminals.

### Step 1: Start the LLM Backend (Terminal 1)

Handles text generation.

```bash
uvicorn llm_fastapi:app --host 0.0.0.0 --port 8000
```

Wait until you see:
“Application startup complete.”

### Step 2: Start the Web UI (Terminal 2)

Launch the interactive web interface.

```bash
streamlit run intergration.py
```

## User Guide

Open the URL shown in Terminal 2 (typically: http://localhost:8501
).

Click Record to start recording your question.

Click Stop to process:

Whisper transcribes speech.

Phi-4 LLM generates the reply.

CosyVoice synthesizes the cloned voice.

Listen to the auto-played response.