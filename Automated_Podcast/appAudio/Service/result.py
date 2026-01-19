from jiwer import wer, cer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import os

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parents[2]
BASE = BASE_DIR / "appAudio" / "Service" /"Output"

TRANSa = os.path.join(BASE, "ActualData")
TRANSp = os.path.join(BASE, "Transcription")

SEGa = os.path.join(BASE, "ActualData")
SEGp = os.path.join(BASE, "T_segmentation")

# ---------------- FILES ----------------
GT_TXT = f"{TRANSa}/ground_truth.txt"
PRED_TXT = f"{TRANSp}/f1c856a5-33f5-4e7c-9bc3-850df73155d2_Bill_Gates_Podcast(128k)_processed.txt"

GT_TOPIC = f"{SEGa}/ground_truth_topics.txt"
PRED_TOPIC = f"{SEGp}/f1c856a5-33f5-4e7c-9bc3-850df73155d2_Bill_Gates_Podcast(128k)_processed_topics.txt"

# ---------------- UTILS ----------------
def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

# ---------------- METRICS ----------------
def transcription_eval():
    gt = read_text(GT_TXT)
    pred = read_text(PRED_TXT)
    return {
        "WER": round(wer(gt, pred), 3),
        "CER": round(cer(gt, pred), 3)
    }

def segmentation_eval():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    gt = read_lines(GT_TOPIC)
    pred = read_lines(PRED_TOPIC)

    gt_emb = model.encode(gt)
    pred_emb = model.encode(pred)

    sim = cosine_similarity(pred_emb, gt_emb).max(axis=1)

    return {
        "Topic_Coherence": round(float(np.mean(sim)), 3),
        "Boundary_Accuracy": round(float(np.mean(sim > 0.6)), 3)
    }

def usability_eval():
    pred = read_text(PRED_TXT)
    words = pred.split()

    return {
        "Readability_Word_Count": len(words),
        "Search_Readiness": "Good" if len(words) > 100 else "Poor"
    }

# ---------------- PERFORMANCE (MANUAL) ----------------
def performance_eval():
    # Manually set values (in seconds)
    audio_duration = 2100      # example: 30 minutes audio
    processing_time = 850      # example: 7 minutes processing

    rtf = processing_time / audio_duration

    return {
        "Audio_Duration_sec": audio_duration,
        "Processing_Time_sec": processing_time,
        "RTF": round(rtf, 3)
    }


# ---------------- RUN ----------------
if __name__ == "__main__":
    print("\n=== TRANSCRIPTION ===")
    print(transcription_eval())

    print("\n=== TOPIC SEGMENTATION ===")
    print(segmentation_eval())

    print("\n=== PERFORMANCE ===")
    print(performance_eval())

    print("\n=== USABILITY ===")
    print(usability_eval())

