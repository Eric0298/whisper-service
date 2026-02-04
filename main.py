from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI(title="whisper-service", version="0.1.0")

# === CONFIG ===
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

# CPU mode (gratis)
model = WhisperModel("base", device="cpu", compute_type="int8")


# === HEALTHCHECKS ===

@app.get("/", include_in_schema=False)
def root():
    return {"ok": True, "service": "whisper-service"}


@app.get("/health")
def health():
    return {"ok": True, "status": "healthy"}


# === TRANSCRIPTION ===

@app.post("/transcribe/file")
async def transcribe_file(
    file: UploadFile = File(...),
    language: str = Form("es"),
    context: str = Form(""),
):
    if not file:
        raise HTTPException(status_code=400, detail="Falta archivo")

    if file.content_type and file.content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo no permitido: {file.content_type}",
        )

    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Archivo vacío")

    if len(content) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Archivo demasiado grande (máx 25MB)",
        )

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
