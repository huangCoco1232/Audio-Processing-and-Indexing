import librosa
import soundfile as sf
from pydub import AudioSegment
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

input_mp3 = os.path.join(BASE_DIR, "CosyVoice", "example_audio", "squidward.mp3")
output_wav = os.path.join(BASE_DIR, "CosyVoice", "example_audio", "squidward_16k_clean.wav")
temp_wav = os.path.join(BASE_DIR, "temp.wav")

print("Input path:", input_mp3)

audio = AudioSegment.from_mp3(input_mp3)
audio.export(temp_wav, format="wav")

y, sr = librosa.load(temp_wav, sr=None, mono=True)

max_len = 29 * sr
if len(y) > max_len:
    y = y[:max_len]

y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)

sf.write(output_wav, y_16k, 16000)

os.remove(temp_wav)

print("finish convert:", output_wav)
