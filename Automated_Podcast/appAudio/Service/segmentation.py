"""
Hybrid Unsupervised Semantic Topic Segmentation
with LLaMA-assisted Human-like Chapter Generation
"""

from pathlib import Path
import json
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama

# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TRANSCRIPT_DIR = BASE_DIR / "appAudio" / "Service" / "Output" / "Transcription"
SEGMENT_DIR = BASE_DIR / "appAudio" / "Service" / "Output" / "T_segmentation"
MODEL_PATH = BASE_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

SEGMENT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# MODELS (Lazy Loaded)
# =========================================================

_embedder = None
_llm = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def get_llm():
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=2048,
            n_gpu_layers=8,
            temperature=0.3,
            verbose=False
        )
    return _llm


# =========================================================
# UTILS
# =========================================================

def sec_to_mmss(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

# I use SentenceTransformer embeddings. 
# These models work best on raw, natural text. 
# Over-cleaning (stopwords, stemming) hurts semantic similarity. 
# LLaMA title generation also needs natural language context
#👉 So minimal preprocessing is a deliberate, good choice.
#Extra preprocessing (stopwords removal, stemming, heavy cleaning):
#❌ Removes context
#❌ Breaks meaning
#❌ Reduces semantic similarity quality

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


# =========================================================
# LLaMA TITLE GENERATION
# =========================================================

def generate_llama_title(text):
    llm = get_llm()

    prompt = f"""
Create a short podcast chapter title.

Rules:
- Max 6 words
- Natural language
- No punctuation

Text:
{text}

Title:
"""

    response = llm(prompt, max_tokens=16, stop=["\n"])
    title = response["choices"][0]["text"].strip()
    return title if title else "General Discussion"


# =========================================================
# CORE SEGMENTATION
# =========================================================

def segment_whisper_file(json_file: Path):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if len(segments) < 2:
        return

    #✔ Removes extra spaces, line breaks
    #✔ Makes text cleaner for embeddings
    #✔ Trims edges
    texts = [clean_text(s["text"]) for s in segments]

    starts = [s["start"] for s in segments]
    ends = [s["end"] for s in segments]

    embedder = get_embedder()
    embeddings = embedder.encode(texts)

    SIM_THRESHOLD = 0.55
    blocks = []
    block_start = 0
    current_emb = embeddings[0]

    for i in range(1, len(embeddings)):
        sim = cosine_similarity(
            current_emb.reshape(1, -1),
            embeddings[i].reshape(1, -1)
        )[0][0]

        if sim < SIM_THRESHOLD and i - block_start >= 5:
            blocks.append((block_start, i - 1))
            block_start = i
            current_emb = embeddings[i]
        else:
            current_emb = (current_emb + embeddings[i]) / 2

    blocks.append((block_start, len(texts) - 1))

    out_file = SEGMENT_DIR / f"{json_file.stem}_topics.txt"

    # with open(out_file, "w", encoding="utf-8") as f:
    #     f.write("00:00 - Introduction\n")
    #     for s, e in blocks:
    #         if starts[s] < 10:
    #             continue
    #         title = generate_llama_title(" ".join(texts[s:s+5]))
    #         f.write(f"{sec_to_mmss(starts[s])} - {title}\n")
    #     f.write(f"{sec_to_mmss(ends[-1])} - Conclusion\n")


    with open(out_file, "w", encoding="utf-8") as f:
        f.write("00:00 - Introduction\n")
        MIN_GAP_SEC = 80  # 1.2 minutes
        last_written_time = 0

        for s, e in blocks:
            start_sec = starts[s]

            if start_sec < 10:
                continue

            if start_sec - last_written_time < MIN_GAP_SEC:
                continue

            title = generate_llama_title(" ".join(texts[s:s+5]))
            f.write(f"{sec_to_mmss(start_sec)} - {title}\n")

            last_written_time = start_sec

        f.write(f"{sec_to_mmss(ends[-1])} - Conclusion\n")


# =========================================================
# DJANGO PIPELINE WRAPPER  ✅ IMPORTANT
# =========================================================

def segment_from_json(json_path):
    """
    Used by Django pipeline.
    Accepts string or Path.
    """
    json_path = Path(json_path)
    segment_whisper_file(json_path)


# =========================================================
# BATCH MODE
# =========================================================

def segment_all():
    for file in TRANSCRIPT_DIR.glob("*.json"):
        segment_whisper_file(file)


if __name__ == "__main__":
    segment_all()
