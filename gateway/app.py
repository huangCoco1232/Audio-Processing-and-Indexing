# gateway/app.py

import os
import time
import json
import requests
import streamlit as st
from io import BytesIO
from audiorecorder import audiorecorder  # streamlit-audiorecorder

# ===== 配置 =====
ASR_URL = "http://localhost:8001/asr"
LLM_URL = "http://localhost:8002/llm"
TTS_URL = "http://localhost:8003/tts"

st.set_page_config(page_title="Voice Chat Gateway")
st.title("🎤 语音助手（ASR → LLM → TTS）")

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clear_chat_history():
    if "messages" in st.session_state:
        del st.session_state.messages


def init_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []
        with st.chat_message("assistant", avatar='🤖'):
            st.markdown("你好！请按下录音按钮开始讲话，我会帮你完成 ASR → LLM → TTS 全流程 😊")

    # 重新渲染历史对话
    for msg in st.session_state.messages:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    return st.session_state.messages


def call_asr(audio_bytes):
    """调用 ASR 服务"""
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    resp = requests.post(ASR_URL, files=files)
    return resp.json().get("text", "")


def call_llm(text):
    """调用 LLM 服务"""
    payload = {"text": text}
    resp = requests.post(LLM_URL, json=payload)
    return resp.json().get("reply", "")


def call_tts(text):
    """调用 TTS 服务"""
    data = {"text": text}
    resp = requests.post(TTS_URL, data=data)
    return resp.content  # wav bytes


def main():
    messages = init_chat_history()

    st.markdown("### 🎙️ 按下按钮开始录音")
    audio = audiorecorder("开始录音", "正在录音... 点击停止")

    if len(audio) > 0:
        # audio 是一个 AudioSegment
        buf = BytesIO()
        audio.export(buf, format="wav")
        audio_bytes = buf.getvalue()

        # 避免重复处理
        if st.session_state.get("last_audio_len", 0) == len(audio_bytes):
            return
        st.session_state["last_audio_len"] = len(audio_bytes)

        # 前端播放用户录音
        st.markdown("#### 🔊 你刚刚录的音频：")
        st.audio(audio_bytes, format="audio/wav")

        # 保存音频文件（可调试）
        ts = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(OUTPUT_DIR, f"{ts}.wav")
        with open(save_path, "wb") as f:
            f.write(audio_bytes)

        st.success(f"录音已保存到: `{save_path}`")

        # =============== 1. Whisper ASR ===============
        with st.spinner("正在识别语音 (ASR)..."):
            text = call_asr(audio_bytes)

        if not text:
            st.error("ASR 没识别到内容，请再试一次。")
            return

        messages.append({"role": "user", "content": text})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(text)

        # =============== 2. LLM 聊天 ===============
        with st.spinner("正在生成 LLM 回复..."):
            reply = call_llm(text)

        messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(reply)

        # =============== 3. TTS 合成 ===============
        with st.spinner("正在语音合成 (TTS)..."):
            tts_audio = call_tts(reply)

        st.markdown("#### 🗣️ 合成语音回复：")
        st.audio(tts_audio, format="audio/wav")

    st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()