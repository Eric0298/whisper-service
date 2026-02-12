from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI(title="whisper-service", version="0.1.0")

MAX_BYTES = 25 * 1024 * 1024  # 25MB
ALLOWED_MIME = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/webm",
    "audio/ogg",
    "audio/mp4",
    "audio/x-m4a",
    "audio/aac",
}

# ✅ Config por entorno (Railway)
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")          # tiny | base | small...
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")      # int8 recomendado CPU
CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "2"))      # baja picos
NUM_WORKERS = int(os.getenv("WHISPER_WORKERS", "1"))          # 1 para no duplicar RAM

# ✅ Modelo global (se carga 1 vez)
model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type=COMPUTE_TYPE,
    cpu_threads=CPU_THREADS,
    num_workers=NUM_WORKERS,
)

@app.get("/", include_in_schema=False)
def root():
    return {"ok": True, "service": "whisper-service", "model": MODEL_SIZE}

@app.get("/health")
def health():
    return {"ok": True, "status": "healthy", "model": MODEL_SIZE}

@app.post("/transcribe/file")
async def transcribe_file(
    file: UploadFile = File(...),
    language: str = Form("es"),
    context: str = Form(""),
):
    if not file:
        raise HTTPException(status_code=400, detail="Falta archivo")

    if file.content_type and file.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=415, detail=f"Tipo no permitido: {file.content_type}")

    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Archivo vacío")

    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Archivo demasiado grande (máx 25MB)")

    suffix = os.path.splitext(file.filename or "")[1] or ".audio"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(
            tmp_path,
            language=language if language else None,
        )

        text = "".join(seg.text for seg in segments).strip()
        duration = float(getattr(info, "duration", 0) or 0)

        return {
            "ok": True,
            "text": text if text else "(sin texto)",
            "durationSec": round(duration),
            "language": language,
            "type": "file",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
