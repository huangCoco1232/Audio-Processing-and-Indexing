"""
intergration.py是在web.py的基础上修改的。
架构设计（手上没有模型源还需要在有模型源的前提下调试。如果可以用就代替web.py
代码使用了相对路径 current_dir 来定位 CosyVoice 文件夹。 因此intergration.py需要放在和 CosyVoice 文件夹 同级 的目录下。
root/
├── CosyVoice/
├── llm_fastapi.py
├── model_loader.py
└── intergration.py
--------------------------------------------------------------------------------
1. 启动 LLM 服务：
   - 在后台运行 llm_fastapi.py。
   - 目的：作为独立服务运行，避免阻塞 Streamlit 主进程，防止界面卡顿。

2. 前端改造 (web.py)：
   A. 初始化 TTS：
      - 在 web.py 启动时加载 CosyVoice 模型。
      - 关键点：利用 @st.cache_resource 装饰器，确保模型只加载一次，避免重复加载消耗资源。

   B. 请求 LLM：
      - 使用 Python requests 库向 llm_fastapi 发送 HTTP POST 请求。
      - 流程：前端输入 -> 发送请求 -> 获取文本回复。

   C. 生成语音：
      - 将 LLM 返回的文本内容传递给 CosyVoice 模型进行推理。

   D. 播放：
      - 获取推理后的音频数据，使用 st.audio 组件在前端播放。
--------------------------------------------------------------------------------
"""

import os
import time
import json
import requests
import torch
import streamlit as st
import whisper
from audiorecorder import audiorecorder
from io import BytesIO
import sys
import torchaudio

# =================配置区域=================
# 设置 CosyVoice 路径 (参考 new.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
cosyvoice_root = os.path.join(current_dir, "CosyVoice")
sys.path.insert(0, cosyvoice_root)
sys.path.insert(0, os.path.join(cosyvoice_root, "third_party", "Matcha-TTS"))
sys.path.insert(0, os.path.join(cosyvoice_root, "third_party", "AcademiCodec"))

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# LLM API 地址 (假设 llm_fastapi.py 运行在本地 8000 端口)
LLM_API_URL = "http://localhost:8000/chat"

# 录音保存与 TTS 输出目录
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 章鱼哥参考音频路径 (参考 audio_convert.py)
REF_AUDIO_PATH = os.path.join(cosyvoice_root, "example_audio", "squidward_16k_clean.wav")
# 章鱼哥参考文本 (参考 new.py，用于 Zero-shot 提示)
REF_TEXT_PROMPT = (
    "look spongebob i told you use your net and go fish! "
    "happy birthday, SpongeBob SquarePants! are you insane. "
    "Hahahaha,Hahahaha,Hahahaha! I'll ever get surrounded by such loser neighbors! "
    "Hahahaha!Hahahaha!Hahahaha! spongebob, can we lower the volume please?"
)

st.set_page_config(page_title="Squidward Voice Bot", layout="wide")
st.title("🐙 Squidward Voice Chat (Phi-4 + CosyVoice2)")


# =================模型加载=================

@st.cache_resource
def init_asr():
    """加载 Whisper ASR 模型"""
    print("Loading Whisper...")
    return whisper.load_model("small")


@st.cache_resource
def init_tts():
    """加载 CosyVoice2 TTS 模型 (只加载一次)"""
    print("Loading CosyVoice2...")
    model_path = os.path.join(cosyvoice_root, "pretrained_models", "CosyVoice2-0.5B")
    # 注意：根据你的显存情况，fp16 可以设为 True
    cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, fp16=False)

    # 预加载参考音频
    if os.path.exists(REF_AUDIO_PATH):
        prompt_speech_16k = load_wav(REF_AUDIO_PATH, 16000)
    else:
        st.error(f"未找到参考音频: {REF_AUDIO_PATH}")
        prompt_speech_16k = None

    return cosyvoice, prompt_speech_16k


# =================功能函数=================

def get_llm_response(user_text):
    """调用 llm_fastapi 接口"""
    try:
        payload = {"messages": [user_text]}
        response = requests.post(LLM_API_URL, json=payload)
        if response.status_code == 200:
            return response.json().get("reply", "")
        else:
            return f"Error: LLM API returned {response.status_code}"
    except Exception as e:
        return f"Error connecting to LLM: {str(e)}"


def generate_audio(tts_model, prompt_speech, text_to_say, output_filename):
    """使用 CosyVoice 生成音频"""
    if not tts_model or not prompt_speech:
        return None

    all_chunks = []
    # 使用 zero_shot 推理
    for out in tts_model.inference_zero_shot(
            text_to_say,
            REF_TEXT_PROMPT,
            prompt_speech,
            stream=False
    ):
        all_chunks.append(out["tts_speech"])

    if all_chunks:
        full_audio = torch.cat(all_chunks, dim=-1)
        save_path = os.path.join(OUTPUT_DIR, output_filename)
        torchaudio.save(save_path, full_audio, tts_model.sample_rate)
        return save_path
    return None


# =================主程序=================

def main():
    # 1. 初始化模型
    asr_model = init_asr()
    tts_model, prompt_speech = init_tts()

    # 2. 初始化聊天记录
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # 初始问候
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Oh, great. Another neighbor. What do you want? (I'm listening...)"
        })

    # 3. 显示历史消息
    for message in st.session_state.messages:
        avatar = '🧑‍💻' if message["role"] == "user" else '🐙'
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            # 如果历史消息里有音频路径，也显示出来（可选）
            if "audio" in message:
                st.audio(message["audio"])

    # 4. 录音部分
    st.markdown("---")
    audio = audiorecorder("Click to Record", "Recording...")

    if len(audio) > 0:
        # 简单的去重逻辑
        buf = BytesIO()
        audio.export(buf, format="wav")
        audio_bytes = buf.getvalue()

        current_len = len(audio_bytes)
        if st.session_state.get("last_audio_len") != current_len:
            st.session_state["last_audio_len"] = current_len

            # --- Step A: 保存用户录音 ---
            timestamp = time.strftime("%H%M%S")
            user_wav_path = os.path.join(OUTPUT_DIR, f"user_{timestamp}.wav")
            with open(user_wav_path, "wb") as f:
                f.write(audio_bytes)

            # --- Step B: ASR (Whisper) ---
            with st.spinner("Listening (Whisper)..."):
                result = asr_model.transcribe(user_wav_path, language="zh")  # 或 auto
                user_text = result.get("text", "").strip()

            if user_text:
                # 显示用户消息
                st.session_state.messages.append({"role": "user", "content": user_text})
                with st.chat_message("user", avatar='🧑‍💻'):
                    st.markdown(user_text)

                # --- Step C: LLM (Phi-4 via API) ---
                with st.spinner("Thinking (Phi-4)..."):
                    reply_text = get_llm_response(user_text)

                # --- Step D: TTS (CosyVoice2) ---
                tts_audio_path = None
                with st.spinner("Speaking (Squidward TTS)..."):
                    tts_filename = f"reply_{timestamp}.wav"
                    tts_audio_path = generate_audio(tts_model, prompt_speech, reply_text, tts_filename)

                # 显示助手消息
                msg_data = {"role": "assistant", "content": reply_text}
                if tts_audio_path:
                    msg_data["audio"] = tts_audio_path

                st.session_state.messages.append(msg_data)

                with st.chat_message("assistant", avatar='🐙'):
                    st.markdown(reply_text)
                    if tts_audio_path:
                        st.audio(tts_audio_path)
            else:
                st.warning("Sorry, please say it again.")

    # 清空按钮
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()