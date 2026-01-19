# appAudio/Service/pipeline.py

from .preprocessing import preprocess_audio
from .transcription import transcribe_audio
from .segmentation import segment_from_json


def run_pipeline(audio_path):
    """
    One-click automated pipeline:
    audio → preprocessing → transcription → segmentation
    """

    # Step 1: Preprocessing
    clean_audio_path = preprocess_audio(audio_path)

    # Step 2: Transcription
    transcript_text = transcribe_audio(clean_audio_path)

    # Step 3: Segmentation
    segments = segment_from_json(transcript_text)

    return segments
