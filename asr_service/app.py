# asr_service/app.py

import os
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from asr_loader import model
import whisper
import time

app = FastAPI(title="Whisper ASR Service")

@app.post("/asr")
async def asr_endpoint(file: UploadFile = File(...)):

    t_start = time.time()
    # 生成临时文件名
    temp_filename = f"temp_{uuid.uuid4().hex}.wav"

    # 保存上传的音频
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    # Whisper 自动检测格式
    result = model.transcribe(temp_filename, fp16=False)

    # 删除临时文件
    os.remove(temp_filename)

    t_end = time.time()

    print(f"Finished in {t_end - t_start} seconds.")

    return JSONResponse({"text": result["text"]})