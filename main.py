from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from faster_whisper import WhisperModel
import tempfile
import os
import shutil

app = FastAPI(title="whisper-service", version="0.1.0")

MAX_BYTES = 25 * 1024 * 1024

ALLOWED_MIME = {
    "audio/mpeg","audio/mp3","audio/wav","audio/x-wav","audio/webm","audio/ogg",
    "audio/mp4","audio/x-m4a","audio/aac",
}

model = WhisperModel("base", device="cpu", compute_type="int8")

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

    suffix = os.path.splitext(file.filename or "")[1] or ".audio"

    # ✅ Guardar a disco en streaming y contar tamaño sin reventar RAM
    size = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        while True:
            chunk = await file.read(1024 * 1024)  # 1MB
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_BYTES:
                tmp.close()
                try:
                    os.remove(tmp.name)
                except Exception:
                    pass
                raise HTTPException(status_code=413, detail="Archivo demasiado grande (máx 25MB)")
            tmp.write(chunk)

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
