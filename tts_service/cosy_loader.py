# tts_service/cosy_loader.py

import os
import sys

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
cosyvoice_root = os.path.join(current_dir, "CosyVoice")

# 插入路径
sys.path.insert(0, cosyvoice_root)
sys.path.insert(0, os.path.join(cosyvoice_root, "third_party", "Matcha-TTS"))
sys.path.insert(0, os.path.join(cosyvoice_root, "third_party", "AcademiCodec"))

# 模型路径
model_path = os.path.join(cosyvoice_root, "pretrained_models", "CosyVoice2-0.5B")

# 固定 prompt audio（这里改成你自己的路径即可）
fixed_prompt_audio_path = os.path.join(
    cosyvoice_root, "example_audio", "squidward_16k_clean.wav"
)

from CosyVoice.cosyvoice.cli.cosyvoice import CosyVoice2
from CosyVoice.cosyvoice.utils.file_utils import load_wav

print("Loading CosyVoice2 model...")
cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, fp16=False)
print("CosyVoice2 loaded!")

print("Loading fixed prompt audio...")
prompt_speech_16k = load_wav(fixed_prompt_audio_path, 16000)
print("Prompt audio loaded!")
