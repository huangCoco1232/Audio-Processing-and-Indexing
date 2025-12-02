# tts_service/app.py

import os
import torch
import torchaudio
from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
from cosy_loader import cosyvoice, prompt_speech_16k

app = FastAPI(title="CosyVoice2 Zero-Shot TTS API (Fixed Voice)")

# 默认固定 prompt_text，你也可以换成任何中文或英文
DEFAULT_PROMPT_TEXT = (
    "look spongebob i told you use your net and go fish! "
    "happy birthday, SpongeBob SquarePants! are you insane. "
    "Hahahaha,Hahahaha,Hahahaha! I'll ever get surrounded by such loser neighbors! "
    "Hahahaha!Hahahaha!Hahahaha! spongebob, can we lower the volume please?"
)

@app.post("/tts")
async def tts_endpoint(
    text: str = Form(...)
):
    # Zero-Shot 推理
    chunks = []
    for out in cosyvoice.inference_zero_shot(
        text,
        DEFAULT_PROMPT_TEXT,
        prompt_speech_16k,
        stream=False,
        text_frontend=False
    ):
        chunks.append(out["tts_speech"])

    # 合并 chunk 并输出 wav 文件
    full_audio = torch.cat(chunks, dim=-1)
    output_path = "output.wav"
    torchaudio.save(output_path, full_audio, cosyvoice.sample_rate)

    return FileResponse(output_path, media_type="audio/wav", filename="tts.wav")