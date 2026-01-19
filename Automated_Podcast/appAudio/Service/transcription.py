"""
Audio Transcription Module
Using OpenAI Whisper (small model)
"""

from pathlib import Path
import whisper
import json

# ===============================
# PATH CONFIGURATION
# ===============================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

PREPROCESSED_DIR = BASE_DIR / "appAudio" / "Service" / "Dataset" / "Preprocessed"
OUTPUT_DIR = BASE_DIR / "appAudio" / "Service" / "Output" / "Transcription"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



import torch
import gc
import os

# Force CUDA cleanup
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# Optional: reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"




# ===============================
# LOAD WHISPER MODEL (ONCE)
# ===============================

model = whisper.load_model("small",device="cuda" ) #medium  device="cpu" device="cuda"


# ===============================
# TRANSCRIPTION FUNCTION
# ===============================

def transcribe_audio(file_path):
    """
    Accepts string or Path
    Returns JSON path (for segmentation)
    """
    file_path = Path(file_path)   # ✅ FIX

    result = model.transcribe(
        str(file_path),
        fp16=False,
        language="en",
        temperature=0.0,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0
    )

    # TXT
    txt_file = OUTPUT_DIR / f"{file_path.stem}.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(result["text"].strip())

    # JSON (USED BY SEGMENTATION)
    json_file = OUTPUT_DIR / f"{file_path.stem}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return json_file   # ✅ IMPORTANT

# ===============================
# BATCH MODE
# ===============================

def transcribe_all():
    for file in PREPROCESSED_DIR.iterdir():
        if file.suffix.lower() in [".wav", ".mp3", ".flac", ".m4a"]:
            transcribe_audio(file)


if __name__ == "__main__":
    transcribe_all()
