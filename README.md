<div align="center">
<h1>
 Voice Mimicry Dialogue System with LLM  
</h1>

**Project:** Audio Processing and Indexing Final Project
  
**Team:** Group 20 (Leiden University):Hanwen Huang, Zhiwei Ma, Yutao Liu, Zhengqi Lang
</div>

## <img src = "./gateway/squidward_avatar.png" width = "8%"> Overview

This project implements an end-to-end conversational AI system that mimics a specific persona (e.g., Squidward). It integrates three main components:

1. **Speech Recognition (ASR):** Converts user speech to text using **Whisper**.  
2. **Language Generation (LLM):** Generates character-specific responses using **Microsoft Phi-4**.  
3. **Voice Cloning (TTS):** Synthesizes speech using **CosyVoice2** (Zero-shot inference).



## <img src = "./gateway/squidward_avatar.png" width = "8%"> Project Directory Structure

This project uses a Streamlit-based webpage demo format, with the following structure:

```
- web_demo(Streamlit)
  -- Whisper
  -- LLM
  -- CosyVoice2

```


## <img src = "./gateway/squidward_avatar.png" width = "8%"> Installation

To install all the library, you need to pip install:
- streamlit and streamlitaudiorecorder
- openai-whisper

## <img src = "./gateway/squidward_avatar.png" width = "8%"> How to use

1. Create Individual Virtual Environments

You must manually create 4 environments under the project root:

```
env_asr/
env_llm/
env_tts/
env_app/
```

2. Install Dependencies for Each Environment

ASR Service

```
py -3.10 -m venv env_asr
env_asr\Scripts\activate
pip install torch==2.3.1
pip install openai-whisper fastapi uvicorn
deactivate
```

LLM Service

```
py -3.10 -m venv env_llm
env_llm\Scripts\activate
pip install torch==2.5.1
pip install transformers==4.49.0 fastapi uvicorn
deactivate
```

TTS Service

```
py -3.10 -m venv env_tts
env_tts\Scripts\activate
pip install torch==2.3.1 torchaudio==2.3.1
pip install transformers==4.51.3 librosa soundfile fastapi uvicorn
deactivate
```
(Some CosyVoice dependencies may require manual installation)


Gateway UI

```
py -3.10 -m venv env_app
env_app\Scripts\activate
pip install streamlit streamlit-audiorecorder requests
deactivate
```


3. Start Each Service
activate each environment in seperate terminal, and start each service
```
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
uvicorn app:app --host 0.0.0.0 --port 8002 --reload
uvicorn app:app --host 0.0.0.0 --port 8003 --reload
streamlit run app.py
```
