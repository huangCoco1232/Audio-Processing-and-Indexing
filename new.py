import sys
import os
import time
import torch
import re
current_dir = os.path.dirname(os.path.abspath(__file__))
cosyvoice_root = os.path.join(current_dir, "CosyVoice")

sys.path.insert(0, cosyvoice_root)

sys.path.insert(0, os.path.join(cosyvoice_root, "third_party", "Matcha-TTS"))

sys.path.insert(0, os.path.join(cosyvoice_root, "third_party", "AcademiCodec"))

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
t_all_start = time.perf_counter()

model_path = os.path.join(cosyvoice_root, "pretrained_models", "CosyVoice2-0.5B")

t_load_start = time.perf_counter()
cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, fp16=False)
t_load_end = time.perf_counter()

prompt_audio_path = os.path.join(cosyvoice_root, "example_audio", "squidward_16k_clean.wav")
prompt_speech_16k = load_wav(prompt_audio_path, 16000)

text = (
    "Soapy's mind became cognisant of the fact that the time had come for him to "
    "resolve himself into a singular Committee of Ways and Means to provide against "
    "the coming rigour. And therefore he moved uneasily on his bench."
)

prompt_text = (
    "look spongebob i told you use your net and go fish! "
    "happy birthday, SpongeBob SquarePants! are you insane. "
    "Hahahaha,Hahahaha,Hahahaha! I'll ever get surrounded by such loser neighbors! "
    "Hahahaha!Hahahaha!Hahahaha! spongebob, can we lower the volume please?"
)

def split_en(text):
    parts = re.split(r'([。！？])', text)
    res = []
    for i in range(0, len(parts), 2):
        seg = parts[i].strip()
        if not seg:
            continue
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        res.append(seg + punct)
    return res or [text]

sentences = split_en(text)

all_chunks = []
t_infer_start = time.perf_counter()
for si, sent in enumerate(sentences):
    local_chunks = []
    for ci, out in enumerate(
        cosyvoice.inference_zero_shot(
            sent,
            prompt_text,
            prompt_speech_16k,
            stream=False,
            text_frontend=False,
        )
    ):
        local_chunks.append(out["tts_speech"])
        torchaudio.save(f"zero_shot_s{si}_c{ci}.wav", out["tts_speech"], cosyvoice.sample_rate)
    if local_chunks:
        all_chunks.append(torch.cat(local_chunks, dim=-1))

if all_chunks:
    full = torch.cat(all_chunks, dim=-1)
    torchaudio.save("zero_shot_full.wav", full, cosyvoice.sample_rate)

t_infer_end = time.perf_counter()
t_all_end = time.perf_counter()

print("Done! Saved zero_shot_s*_c*.wav and zero_shot_full.wav")
print(f"Deduction time:     {t_infer_end - t_infer_start:.3f} s")
print(f"Total script duration:   {t_all_end - t_all_start:.3f} s")