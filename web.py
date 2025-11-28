import os
import time
import json
import torch
import streamlit as st
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.generation.utils import GenerationConfig

import whisper
from audiorecorder import audiorecorder  # 来自 streamlit-audiorecorder
from io import BytesIO
st.set_page_config(page_title="Baichuan 2 语音 Demo")
st.title("Baichuan 2 语音版（暂用 Whisper 文本代替 LLM 输出）")

# 录音保存目录：当前目录下的 output 文件夹
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


############################
# （保留）Baichuan2 大模型 #
############################
@st.cache_resource
def init_llm():
    """
    这里保留原来的 Baichuan2 加载逻辑，
    但现在 main() 里暂时不调用，等你以后下载好模型再启用。
    """
    model = AutoModelForCausalLM.from_pretrained(
        "baichuan-inc/Baichuan2-13B-Chat",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.generation_config = GenerationConfig.from_pretrained(
        "baichuan-inc/Baichuan2-13B-Chat"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "baichuan-inc/Baichuan2-13B-Chat",
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer


#####################
# Whisper ASR 模型  #
#####################
@st.cache_resource
def init_asr():
    # 按你机器性能选 tiny/base/small/medium/large
    # small 是一个比较折中版本
    model = whisper.load_model("small")
    return model


def clear_chat_history():
    if "messages" in st.session_state:
        del st.session_state.messages


def init_chat_history():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，这是语音输入版界面。目前还没接入百川 LLM，先用 Whisper 把你说的话转成文字再展示给你 🥰")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages


def main():
    # 暂时只加载 Whisper，不加载 LLM，避免去下 13B 模型
    asr_model = init_asr()
    messages = init_chat_history()

    st.markdown("### 🎙️ 按下按钮开始录音")

    # audiorecorder 返回的是一个 AudioSegment 对象（内部有音频数据）
    audio = audiorecorder(
        "开始录音",          # 按钮文案：初始状态
        "正在录音... 点击停止"  # 按钮文案：录音时
    )

    # 录音结束后，audio 会变成非空
    if len(audio) > 0:
        # 为了避免每次刷新都重复处理同一段音频，我们简单做个“去重”
        buf = BytesIO()
        audio.export(buf, format="wav")
        audio_bytes = buf.getvalue()

        audio_len = len(audio_bytes)
        last_len = st.session_state.get("last_audio_len", 0)
        if audio_len == last_len:
            # 同一段音频重复 rerun，直接不再处理
            pass
        else:
            st.session_state["last_audio_len"] = audio_len

            # 1. 前端播放录音
            st.markdown("#### 🔊 你刚刚录的音频：")
            st.audio(audio_bytes, format="audio/wav")

            # 2. 保存到 ./output 目录
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{ts}.wav"
            save_path = os.path.join(OUTPUT_DIR, filename)
            with open(save_path, "wb") as f:
                f.write(audio_bytes)

            st.success(f"录音已保存到: `{save_path}`")

            # 3. 用 Whisper 识别
            with st.spinner("正在用 Whisper 识别语音..."):
                # 也可以不指定 language，让 Whisper 自动检测
                result = asr_model.transcribe(save_path, language="zh")
            text = result.get("text", "").strip()

            if not text:
                st.error("Whisper 没识别出内容，可以再录一遍，尽量靠近麦克风说得清晰一点。")
            else:


                ####加入llm的东西
                
                # 4. 把识别的文本当作“用户输入”
                messages.append({"role": "user", "content": text})
                with st.chat_message("user", avatar='🧑‍💻'):
                    st.markdown(text)

                # 5. 助手回复：暂时不用 LLM，就直接回显文本
                reply = f"（当前未接入百川 LLM，仅展示 Whisper 识别的文本）\n\n{text}"
                messages.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant", avatar='🤖'):
                    st.markdown(reply)

                print(json.dumps(messages, ensure_ascii=False), flush=True)

    # 清空对话按钮
    st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()
