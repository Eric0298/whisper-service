from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI()

# CPU mode (gratis). Si algún día tienes GPU, se puede cambiar.
model = WhisperModel("base", device="cpu", compute_type="int8")

ALLOWED_MIME = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
    "audio/webm", "audio/ogg", "audio/mp4", "audio/x-m4a", "audio/aac"
}

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("es"),
    context: str = Form("")
):
    if file.content_type and file.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=415, detail=f"Tipo no permitido: {file.content_type}")

    # Guardar temporalmente el archivo
    suffix = os.path.splitext(file.filename or "")[1] or ".audio"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Whisper (faster-whisper) soporta "language" opcional
        segments, info = model.transcribe(
            tmp_path,
            language=language if language else None
        )

        text_parts = []
        for seg in segments:
            text_parts.append(seg.text)

        text = "".join(text_parts).strip()
        duration = float(getattr(info, "duration", 0) or 0)

        # "context" aquí todavía no se usa (whisper base no lo aprovecha como prompt real),
        # pero lo mantenemos por compatibilidad y para futuros proveedores.
        return {
            "ok": True,
            "text": text if text else "(sin texto)",
            "durationSec": duration,
            "language": language,
        }
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass
