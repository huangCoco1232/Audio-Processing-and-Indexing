import sys
import os
import time

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

text = "Soapy's mind became cognisant of the fact that the time had come for him to resolve himself into a singular Committee of Ways and Means to provide against the coming rigour. And therefore he moved uneasily on his bench."

prompt_text = "look spongebob i told you use your net and go fish! happy birthday,SpongeBob!SquarePants!are you insane.Hahahaha, Hahahaha ,Hahahaha  I'll ever get surrounded by such loser neighbors!Hahahaha! Hahahaha! spongebob, can we lower the volume please?"

chunks = []
t_infer_start = time.perf_counter()
for i, out in enumerate(
    cosyvoice.inference_zero_shot(
        text,
        prompt_text,
        prompt_speech_16k,
        stream=False,
        text_frontend=False,
    )
):
    chunks.append(out["tts_speech"])
    torchaudio.save(f"zero_shot_chunk_{i}.wav", out["tts_speech"], cosyvoice.sample_rate)

import torch
full = torch.cat(chunks, dim=-1)
torchaudio.save("zero_shot_full.wav", full, cosyvoice.sample_rate)

t_infer_end = time.perf_counter()
t_all_end = time.perf_counter()

print("Done! Saved zero_shot_chunk_*.wav and zero_shot_full.wav")
print(f"Deduction time:     {t_infer_end - t_infer_start:.3f} s")
print(f"Total script duration:   {t_all_end - t_all_start:.3f} s")