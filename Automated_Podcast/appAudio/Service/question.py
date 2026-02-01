from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from dotenv import load_dotenv
from groq import Groq

# --------------------------------------------------
# LOAD ENV
# --------------------------------------------------
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# --------------------------------------------------
# CONFIGURE GROQ CLIENT
# --------------------------------------------------
client = Groq(api_key=api_key)

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
TRANSCRIPT_DIR = BASE_DIR / "appAudio" / "Service" / "Output" / "Transcription"

# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------
# LOAD & CHUNK DATA
# --------------------------------------------------
MAX_CHARS_PER_CHUNK = 1200
chunks = []

if TRANSCRIPT_DIR.exists():
    for txt in TRANSCRIPT_DIR.glob("*.txt"):
        text = txt.read_text(encoding="utf-8")
        for p in text.split("\n\n"):
            p = p.strip()
            if p:
                chunks.append(p[:MAX_CHARS_PER_CHUNK])

# --------------------------------------------------
# CREATE EMBEDDINGS + FAISS
# --------------------------------------------------
if not chunks:
    index = None
else:
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

# --------------------------------------------------
# PUBLIC FUNCTION
# --------------------------------------------------
def answer_from_transcript(question: str) -> str:
    if not question:
        return "Please ask a question."

    if index is None:
        return "Transcript not available yet."

    # Embed question
    q_embed = embedder.encode([question], convert_to_numpy=True)

    # Retrieve most relevant chunk
    _, ids = index.search(q_embed, k=1)
    context = chunks[ids[0][0]]

    # Prompt
    prompt = f"""
You are answering from a podcast transcript.

Rules:
- Answer ONLY using the context below
- Keep the answer SHORT (2–4 lines)
- If the answer is not present, reply ONLY with:
  Not found in the document.

Context:
{context}

Question:
{question}

Answer:
"""

    # GROQ CALL
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # FREE & FAST
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.2
    )

    answer = response.choices[0].message.content.strip()

    if not answer:
        return "Not found in the document."

    return answer

# # --------------------------------------------------
# # LOCAL TEST
# # --------------------------------------------------
# if __name__ == "__main__":
#     print("Chunks loaded:", len(chunks))
#     print(answer_from_transcript("What is this podcast about?"))
