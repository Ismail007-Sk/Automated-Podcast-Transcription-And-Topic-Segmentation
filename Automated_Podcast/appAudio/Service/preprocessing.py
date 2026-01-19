"""
Audio Preprocessing Pipeline
Optimized for Speech / ASR (Whisper)
"""

from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
import torch
from silero_vad import get_speech_timestamps

# ===============================
# CONFIGURATION
# ===============================

TARGET_SR = 16000  # ASR standard

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "appAudio" / "Service" / "Dataset" / "Raw"
PREPROCESSED_DIR = BASE_DIR / "appAudio" / "Service" / "Dataset" / "Preprocessed"

PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# LOAD SILERO VAD MODEL
# ===============================
vad_model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False
)
(get_speech_timestamps, _, _, _, _) = utils

# ===============================
# CORE PREPROCESSING
# ===============================

def preprocess_audio(file_path):
    """
    Accepts string or Path
    Returns Path to processed audio
    """
    file_path = Path(file_path)   # ✅ FIX

    audio, sr = librosa.load(file_path, sr=TARGET_SR)

    if audio.ndim > 1:
        audio = librosa.to_mono(audio)

    audio = librosa.util.normalize(audio)
    audio = nr.reduce_noise(y=audio, sr=TARGET_SR)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    audio = apply_silero_vad(audio, TARGET_SR)

    output_file = PREPROCESSED_DIR / f"{file_path.stem}_processed.wav"
    sf.write(output_file, audio, TARGET_SR)

    return output_file   # ✅ IMPORTANT

# ===============================
# SILERO VAD
# ===============================

def apply_silero_vad(audio, sr):
    audio_tensor = torch.from_numpy(audio).float()

    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        vad_model,
        sampling_rate=sr
    )

    if not speech_timestamps:
        return audio

    return np.concatenate(
        [audio[s["start"]:s["end"]] for s in speech_timestamps]
    )

# ===============================
# BATCH MODE
# ===============================

def process_all():
    for file in RAW_DIR.iterdir():
        if file.suffix.lower() in [".wav", ".mp3", ".flac", ".m4a"]:
            preprocess_audio(file)


if __name__ == "__main__":
    process_all()
