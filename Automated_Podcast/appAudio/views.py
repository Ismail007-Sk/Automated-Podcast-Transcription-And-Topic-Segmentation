from django.shortcuts import render
from django.conf import settings
import os, uuid, shutil, time, threading

from .Service.preprocessing import preprocess_audio
from .Service.transcription import transcribe_audio
from .Service.segmentation import segment_from_json


# --------------------------------------------------
# Base paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "appAudio", "Service", "Dataset", "Raw")
PREPROCESS_DIR = os.path.join(BASE_DIR, "appAudio", "Service", "Dataset", "Preprocessed")
TRANS_DIR = os.path.join(BASE_DIR, "appAudio", "Service", "Output", "Transcription")
SEG_DIR = os.path.join(BASE_DIR, "appAudio", "Service", "Output", "T_segmentation")

MEDIA_TRANS = os.path.join(settings.MEDIA_ROOT, "transcription")
MEDIA_SEG = os.path.join(settings.MEDIA_ROOT, "segmentation")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(MEDIA_TRANS, exist_ok=True)
os.makedirs(MEDIA_SEG, exist_ok=True)


# --------------------------------------------------
# Auto-clean helpers
# --------------------------------------------------
def clean_old_files(folder):
    """Delete all files inside a folder"""
    if not os.path.exists(folder):
        return

    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path):
            os.remove(path)


def delete_later(path, delay=300):
    """Delete a file after delay (seconds)"""
    def _delete():
        time.sleep(delay)
        if os.path.exists(path):
            os.remove(path)
    threading.Thread(target=_delete, daemon=True).start()


# --------------------------------------------------
# Main view
# --------------------------------------------------
def upload_audio(request):
    context = {}

    if request.method == "POST":
        action = request.POST.get("action")

        # ---------- CANCEL ----------
        if action == "cancel":
            return render(request, "upload.html", {
                "status": "❌ Cancelled by user"
            })

        # ---------- UPLOAD ----------
        if action == "upload":
            audio_file = request.FILES.get("audio")
            if not audio_file:
                return render(request, "upload.html", {
                    "status": "No file selected"
                })

            # 🔥 AUTO DELETE OLD RAW AUDIO
            clean_old_files(RAW_DIR)
            clean_old_files(PREPROCESS_DIR)
            clean_old_files(TRANS_DIR)
            clean_old_files(SEG_DIR)

            filename = f"{uuid.uuid4()}_{audio_file.name}"
            file_path = os.path.join(RAW_DIR, filename)

            with open(file_path, "wb+") as f:
                for chunk in audio_file.chunks():
                    f.write(chunk)

            return render(request, "upload.html", {
                "status": "✅ Upload completed",
                "uploaded_file": filename
            })

        # ---------- PROCESS ----------
        if action == "process":
            start_time = time.time()

            uploaded_file = request.POST.get("uploaded_file")
            file_path = os.path.join(RAW_DIR, uploaded_file)

            # preprocessing → transcription → segmentation
            clean_audio = preprocess_audio(file_path)
            json_path = transcribe_audio(clean_audio)
            segment_from_json(json_path)

            # output files
            txt_file = os.path.basename(json_path).replace(".json", ".txt")
            seg_file = os.path.basename(json_path).replace(".json", "_topics.txt")

            trans_src = os.path.join(TRANS_DIR, txt_file)
            seg_src = os.path.join(SEG_DIR, seg_file)

            trans_dst = os.path.join(MEDIA_TRANS, txt_file)
            seg_dst = os.path.join(MEDIA_SEG, seg_file)

            shutil.copy(trans_src, trans_dst)
            shutil.copy(seg_src, seg_dst)

            # file sizes
            txt_size = os.path.getsize(trans_dst) // 1024
            seg_size = os.path.getsize(seg_dst) // 1024

            # ⏱ auto delete downloadable files
            delete_later(trans_dst, delay=300)
            delete_later(seg_dst, delay=300)

            # 🧹 immediate cleanup of raw audio
            if os.path.exists(file_path):
                os.remove(file_path)

            context.update({
                "status": "✅ Processing completed",
                "uploaded_file": uploaded_file,
                "transcription_file": f"{settings.MEDIA_URL}transcription/{txt_file}",
                "segmentation_file": f"{settings.MEDIA_URL}segmentation/{seg_file}",
                "txt_size": txt_size,
                "seg_size": seg_size,
                "time_taken": round(time.time() - start_time, 2),
                "done": True
            })

            return render(request, "upload.html", context)

    return render(request, "upload.html")

def home(request):
    return render(request, "home.html")
