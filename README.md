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

---

## 🧩 Solution Overview

The system is a **Django-based backend application** that provides a one-click automated pipeline. It performs audio preprocessing, speech-to-text transcription, and semantic topic segmentation, all executed **locally without cloud APIs**.

**User Workflow:**
1. Upload podcast audio (any format supported; `.wav` recommended)
2. Audio preprocessing (noise reduction, silence removal, VAD)
3. Speech-to-text transcription using Whisper
4. Semantic topic segmentation with AI-generated titles
5. Download final outputs:
   - `transcription.txt`
   - `topic_segmented_timestamps.txt`

---

## 🏗️ System Architecture

The system follows a modular, pipeline-based architecture:

- Frontend: Audio upload and result download
- Backend: Django-based orchestration
- Processing Modules:
  - Audio preprocessing
  - Transcription
  - Topic segmentation
  - Evaluation

📌 The complete workflow diagram is available in the `assets/` folder.

---

## ⚙️ Tech Stack

| Layer | Technology |
|------|------------|
| Backend | Django |
| ASR | OpenAI Whisper (Small) |
| NLP | Sentence Transformers |
| GenAI | Meta-LLaMA-3.1-8B (GGUF) |
| Audio Processing | Librosa, Silero VAD |
| Evaluation | JiWER, Scikit-learn |

---

## 🧩 Core Modules

| Module | Responsibility |
|------|---------------|
| `preprocessing/` | Audio cleaning, noise reduction, silence removal, VAD |
| `transcription/` | Speech-to-text transcription |
| `segmentation/` | Semantic topic segmentation & title generation |
| `pipeline.py` | End-to-end pipeline orchestration |
| `evaluation.py` | Quality and performance evaluation |

---

## 📄 Outputs / Results

The system generates two user-downloadable files:
- **`transcription.txt`** – Complete podcast transcript  
- **`topic_segmented_timestamps.txt`** – Topic-wise timestamps with titles  

These outputs make long podcasts easy to explore and reuse.

---

## 📊 Evaluation Metrics

### 🔹 Transcription Quality
- **WER:** 0.41  
- **CER:** 0.331  

### 🔹 Topic Segmentation Performance
- **Topic Coherence:** 0.547  
- **Boundary Accuracy:** 0.32  
- **Total Predicted Topics:** 25  

### 🔹 GenAI Usage
- ASR Model: Whisper Small  
- LLM Model: Meta-LLaMA-3.1-8B-Instruct (GGUF)  
- Purpose: Transcription and human-like topic title generation  

### 🔹 Safety Handling
- Fully local execution  
- No user data storage  
- No cloud API usage  

### 🔹 Cost Awareness
- Low GPU usage  
- Zero API cost  
- Minimal inference overhead  

### 🔹 Code Quality
- Modular pipeline design  
- Reusable and readable components  

### 🔹 Documentation & Explainability
- Clear docstrings and inline comments  
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

- Local-only processing  
- No external data leakage  
- User data remains private  

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
