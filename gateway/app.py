import os
import time
import requests
import streamlit as st
from io import BytesIO
from audiorecorder import audiorecorder

# ================= Configuration =================
# Backend Service URLs (Hardcoded for production)
DEFAULT_ASR_URL = "http://localhost:8001/asr"
DEFAULT_LLM_URL = "http://localhost:8002/llm"
DEFAULT_TTS_URL = "http://localhost:8003/tts"

# Directory configuration
OUTPUT_DIR = "output"
AVATAR_FILENAME = "squidward_avatar.png"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Page Setup
st.set_page_config(
    page_title="Group 20 - Voice Mimicry System",
    page_icon="🐙",
    layout="wide"
)

# ================= Helper Functions =================

def get_avatar_image():
    """
    Get the absolute path of the avatar image.
    Returns the emoji '🐙' if the file does not exist.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    avatar_path = os.path.join(current_dir, AVATAR_FILENAME)
    return avatar_path if os.path.exists(avatar_path) else "🐙"


def call_service(url, payload, file_mode=False):
    """
    Unified function to call ASR or LLM services.
    :param url: Service API URL
    :param payload: Audio bytes (for ASR) or Text string (for LLM)
    :param file_mode: Boolean, True for file upload (ASR), False for JSON (LLM)
    """
    try:
        if file_mode:
            # ASR: Upload audio file
            files = {"file": ("audio.wav", payload, "audio/wav")}
            resp = requests.post(url, files=files, timeout=15)
            return resp.json().get("text", "") if resp.status_code == 200 else None
        else:
            # LLM: Send text payload
            resp = requests.post(url, json={"text": payload}, timeout=60)
            return resp.json().get("reply", "") if resp.status_code == 200 else None
    except Exception as e:
        st.error(f"Connection Failed: {e}")
        return None


def call_tts_service(text):
    """
    Call the TTS service to generate audio.
    Expects binary audio content in response.
    """
    try:
        # TTS usually expects form data or json, here assuming form data based on previous context
        resp = requests.post(DEFAULT_TTS_URL, data={"text": text}, timeout=60)
        return resp.content if resp.status_code == 200 else None
    except Exception as e:
        st.error(f"TTS Failed: {e}")
        return None


def init_chat_history():
    """
    Initialize chat history with a system greeting if not present.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Oh, great. Another neighbor. What do you want?"
        })
    return st.session_state.messages


def clear_chat():
    """
    Clear chat history by removing the session state key.
    This forces a re-initialization on the next run.
    """
    if "messages" in st.session_state:
        del st.session_state.messages


# ================= Main UI =================

def main():
    # Title Section
    st.title("Group 20 - Voice Mimicry Dialogue System")
    st.subheader("Squidward Voice Chat")

    st.markdown("---")

    # Initialize Data
    messages = init_chat_history()
    bot_avatar = get_avatar_image()

    # Chat Display Area
    chat_container = st.container()
    with chat_container:
        for msg in messages:
            avatar = "🧑‍💻" if msg["role"] == "user" else bot_avatar
            with st.chat_message(msg["role"], avatar=avatar):
                st.write(msg["content"])
                if "audio" in msg:
                    st.audio(msg["audio"], format="audio/wav")

    st.markdown("###")

    # ================= Bottom Button Area =================

    # Custom CSS: Unify button styles and alignment
    st.markdown("""
    <style>
    /* Align buttons to the bottom of the container */
    div[data-testid="stHorizontalBlock"] > div {
        display: flex;
        align-items: flex-end;
    }

    /* Adjust AudioRecorder button style */
    .stAudioRecorder button {
        height: 38px !important;
        padding: 0.25rem 0.75rem !important;
        font-size: 14px !important;
        border-radius: 0.25rem !important;
    }

    /* Adjust Standard Streamlit button style to match */
    button[kind="secondary"] {
        height: 38px !important;
        padding: 0.25rem 0.75rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Layout: Recorder (Left) | Clear (Middle) | Spacer (Right)
    col_rec, col_clear, col_rest = st.columns([1.2, 1, 6])

    with col_rec:
        # Dynamic label: "Start" for first turn, "Continue" for subsequent turns
        start_label = "Start Recording" if len(messages) <= 1 else "Continue"
        audio = audiorecorder(start_label, "Stop")

    with col_clear:
        if st.button("Clear Chat", use_container_width=False):
            clear_chat()
            st.rerun()

    # ================= Main Processing Logic =================
    if len(audio) > 0:
        buf = BytesIO()
        audio.export(buf, format="wav")
        audio_bytes = buf.getvalue()

        # Prevent reprocessing the same audio segment upon app refresh
        if st.session_state.get("last_audio_len", 0) != len(audio_bytes):
            st.session_state["last_audio_len"] = len(audio_bytes)

            # Save user audio locally
            ts = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(OUTPUT_DIR, f"{ts}.wav")
            with open(save_path, "wb") as f:
                f.write(audio_bytes)

            # UI Feedback: Toast notification
            st.toast("Processing Pipeline...", icon="⏳")

            # Step 1: ASR (Speech to Text)
            user_text = call_service(DEFAULT_ASR_URL, audio_bytes, file_mode=True)

            if user_text:
                st.session_state.messages.append({"role": "user", "content": user_text})

                # Step 2: LLM (Text Generation)
                bot_reply = call_service(DEFAULT_LLM_URL, user_text, file_mode=False)

                if bot_reply:
                    # Step 3: TTS (Text to Speech)
                    tts_audio = call_tts_service(bot_reply)

                    # Append Assistant Response
                    msg_data = {"role": "assistant", "content": bot_reply}
                    if tts_audio:
                        msg_data["audio"] = tts_audio

                    st.session_state.messages.append(msg_data)
                    st.rerun()
            else:
                st.error("Could not recognize speech. Please try again.")


if __name__ == "__main__":
    main()