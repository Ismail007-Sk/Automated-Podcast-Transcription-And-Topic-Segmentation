# Automated Podcast Transcription and Topic Segmentation

## 🧩 Problem Statement

Podcasts are increasingly popular, but long audio episodes are difficult to navigate. Manual transcription and topic-wise timestamp creation is time-consuming and inefficient.  
This project builds an **automated system that converts podcast audio into structured, searchable, topic-wise knowledge using GenAI**.

---

## 🎯 Project Objective

To design and implement an **end-to-end automated pipeline** that transforms podcast audio into:
- Accurate text transcription  
- Semantically segmented topics with timestamps  
- Human-like chapter titles for easy navigation  
- Context-aware question answering from podcast transcripts  

---

## 🧩 Solution Overview

The system is a **Django-based backend application** that provides a one-click automated pipeline. It performs audio preprocessing, speech-to-text transcription, semantic topic segmentation, and **question answering**, executed using **Groq-powered GenAI**.

**User Workflow:**
1. Upload podcast audio (any format supported; `.wav` recommended)
2. Audio preprocessing (noise reduction, silence removal, VAD)
3. Speech-to-text transcription using Whisper
4. Semantic topic segmentation with AI-generated titles (Groq API)
5. Ask natural language questions from the generated transcription
6. Download final outputs:
   - `transcription.txt`
   - `topic_segmented_timestamps.txt`

---

## 🏗️ System Architecture

The system follows a modular, pipeline-based architecture:

- Frontend: Audio upload, question input, and result download
- Backend: Django-based orchestration
- Processing Modules:
  - Audio preprocessing
  - Transcription
  - Topic segmentation
  - Question answering
  - Evaluation

📌 The complete workflow diagram is available in the `assets/` folder.

---

## ⚙️ Tech Stack

| Layer | Technology |
|------|------------|
| Backend | Django |
| ASR | OpenAI Whisper (Small) |
| NLP | Sentence Transformers |
| GenAI | Groq API (llama-3.3-70b-versatile) |
| Audio Processing | Librosa, Silero VAD |
| Evaluation | JiWER, Scikit-learn |

---

## 🧩 Core Modules

| Module | Responsibility |
|------|---------------|
| `preprocessing/` | Audio cleaning, noise reduction, silence removal, VAD |
| `transcription/` | Speech-to-text transcription |
| `segmentation/` | Semantic topic segmentation & title generation (Groq-based) |
| `question.py` | Question answering from podcast transcription |
| `pipeline.py` | End-to-end pipeline orchestration |
| `evaluation.py` | Quality and performance evaluation |

---

## 📄 Outputs / Results

The system generates two user-downloadable files:
- **`transcription.txt`** – Complete podcast transcript  
- **`topic_segmented_timestamps.txt`** – Topic-wise timestamps with titles  

Additionally, users can **interactively query the podcast content** using natural language questions.

---
## 📊 Evaluation Metrics

### 🔹 Transcription Quality
- **WER:** 0.561  
- **CER:** 0.462  

### 🔹 Topic Segmentation Performance
- **Topic Coherence:** 0.849  
- **Semantic Boundary Alignment:** 1.0  
- **Total Predicted Topics:** 8  

### 🔹 GenAI Usage
- ASR Model: Whisper Small  
- LLM Model: llama-3.3-70b-versatile (via Groq API)  
- Purpose:
  - ASR output evaluation  
  - Topic title generation  

### 🔹 Safety Handling
- Hybrid execution (Local ASR + Cloud LLM)  
- No user data storage  
- Cloud API used: Groq API  

### 🔹 Cost Awareness
- ASR Model: Whisper Small  
- Reason: Low GPU usage, zero ASR API cost  
- Inference Cost: Low (local ASR) + API-based (LLM)  

### 🔹 Code Quality
- Modular pipeline design  
- Reusable components  
- Readable and maintainable code  

### 🔹 Documentation & Explainability
- Clear docstrings  
- Inline comments  
- Stepwise, explainable pipeline  
- Transparent evaluation metrics  

---

## 🚫 Not in Scope

- Real-time transcription  
- Speaker diarization  
- Live podcast streaming  

---

## 🌱 Stretch Goals

- Multilingual transcription support  
- Advanced Whisper models (medium / large)  
- Search functionality inside transcripts  

---

## 🔐 Safety & Privacy

- Secure API-based LLM usage  
- No persistent storage of user data  
- API keys managed via environment variables  

---

## 📘 Project Resources

All project resources are available in the **`assets/` folder**, including:
- 📘 **Full Project Report (PDF):**  
  [View Report](./Assest_Please_check/Automated-Podcast-Transcription-And-Topic-Segmentation(Infosys-Springboard).pdf)

- 📊 **Project Presentation (PPT):**  
  [View Presentation](./Assest_Please_check/Automated-Podcast-Transcription-And-Topic-Segmentation(Infosys-Springboard)PPT.pdf)

- 📊 **Project Workflow :**  
  [View Presentation](./Assest_Please_check/workflow.png)

📎 Project Video Explanation Google Drive (Demo & Resources):  
https://drive.google.com/file/d/1RAtWC6xAEqP-cBFJroqcZYl9zd2NP0ZY/view?usp=drivesdk

---

## 👥 Contributor

**Ismail Sk**  
ML / NLP / Backend Developer  

---

## 📜 Internship Context

This project was developed as an **individual project** under the  
**Infosys Springboard Internship 6.0** program.

---

⭐ If you find this project useful, consider starring the repository!
