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

model = WhisperModel("tiny", device="cpu", compute_type="int8")
# model = WhisperModel("base", device="cpu", compute_type="int8")


@app.get("/", include_in_schema=False)
def root():
    return {"ok": True, "service": "whisper-service"}


@app.get("/health")
def health():
    return {"ok": True, "status": "healthy"}


def format_paragraphs_from_segments(seg_list, max_paragraph_sec=15.0, pause_sec=1.0):
    """
    Junta segments en párrafos:
    - corta si el párrafo acumula >= max_paragraph_sec
    - corta si hay una pausa (gap) >= pause_sec
    Devuelve string con párrafos separados por doble salto de línea.
    """
    paragraphs = []
    buf = []
    start_t = None
    last_end = None

    def flush():
        nonlocal buf, start_t
        text = " ".join(" ".join(buf).split()).strip()
        if text:
            paragraphs.append(text)
        buf = []
        start_t = None

    for seg in seg_list:
        s = float(getattr(seg, "start", 0.0) or 0.0)
        e = float(getattr(seg, "end", 0.0) or 0.0)
        t = (getattr(seg, "text", "") or "").strip()
        if not t:
            last_end = e
            continue

        if start_t is None:
            start_t = s

        if last_end is not None:
            gap = s - float(last_end)
            if gap >= pause_sec:
                flush()
                start_t = s

        buf.append(t)

        if start_t is not None and (e - start_t) >= max_paragraph_sec:
            flush()

        last_end = e

    flush()
    return "\n\n".join(paragraphs).strip()


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

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name

            total = 0
            chunk_size = 1024 * 1024  # 1MB
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BYTES:
                    raise HTTPException(status_code=413, detail="Archivo demasiado grande (máx 25MB)")
                tmp.write(chunk)

        segments_iter, info = model.transcribe(
            tmp_path,
            language=language if language else None,
        )

        # ✅ Convertimos a lista para poder usarlo varias veces
        seg_list = list(segments_iter)

        # ✅ Texto crudo (igual que tu idea, pero con espacios bien)
        raw_text = " ".join((seg.text or "").strip() for seg in seg_list).strip()

        # ✅ Segments con timestamps
        segments_out = [
            {
                "id": i,
                "start": float(getattr(seg, "start", 0.0) or 0.0),
                "end": float(getattr(seg, "end", 0.0) or 0.0),
                "text": (getattr(seg, "text", "") or "").strip(),
            }
            for i, seg in enumerate(seg_list)
            if (getattr(seg, "text", "") or "").strip()
        ]

        # ✅ Texto “guionizado” básico por párrafos (sin IA extra)
        paragraph_text = format_paragraphs_from_segments(
            seg_list,
            max_paragraph_sec=15.0,
            pause_sec=1.0,
        )

        duration = float(getattr(info, "duration", 0) or 0)

        return {
            "ok": True,
            # Mantén text como antes (string). Si prefieres crudo, cambia esta línea a raw_text.
            "text": paragraph_text if paragraph_text else (raw_text if raw_text else "(sin texto)"),
            "rawText": raw_text,
            "durationSec": round(duration),
            "language": language,
            "type": "file",
            "segments": segments_out,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if tmp_path:
                os.remove(tmp_path)
        except Exception:
            pass
