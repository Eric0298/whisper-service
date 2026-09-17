from collections import defaultdict
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from faster_whisper import WhisperModel
import asyncio
import logging
import os
from pathlib import Path
import re
import secrets
import subprocess
import tempfile
import time

logger = logging.getLogger("whisper-service")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(value, maximum))


MAX_AUDIO_FILE_SIZE_MB = env_int("MAX_AUDIO_FILE_SIZE_MB", 10, 1, 250)
MAX_BYTES = MAX_AUDIO_FILE_SIZE_MB * 1024 * 1024
MAX_AUDIO_DURATION_SEC = env_int("MAX_AUDIO_DURATION_SEC", 180, 10, 7200)
RATE_LIMIT_PER_MINUTE = env_int("RATE_LIMIT_PER_MINUTE", 30, 1, 600)
SERVICE_TOKEN = os.getenv("WHISPER_SERVICE_TOKEN", "").strip()
MODEL_NAME = os.getenv("WHISPER_MODEL", "small").strip() or "small"
MODEL_DEVICE = os.getenv("WHISPER_DEVICE", "cpu").strip() or "cpu"
MODEL_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip() or "int8"
MODEL_CPU_THREADS = env_int("WHISPER_CPU_THREADS", 8, 0, 64)

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

ALLOWED_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".webm",
    ".ogg",
    ".m4a",
    ".mp4",
    ".aac",
}

app = FastAPI(title="whisper-service", version="0.2.0")
auth_scheme = HTTPBearer(auto_error=False)
rate_buckets: dict[str, list[float]] = defaultdict(list)
model: WhisperModel | None = None

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

def get_model():
    global model
    if model is None:
        logger.info(
            "loading whisper model %s (device=%s, compute_type=%s, cpu_threads=%d)",
            MODEL_NAME,
            MODEL_DEVICE,
            MODEL_COMPUTE_TYPE,
            MODEL_CPU_THREADS,
        )
        model = WhisperModel(
            MODEL_NAME,
            device=MODEL_DEVICE,
            compute_type=MODEL_COMPUTE_TYPE,
            cpu_threads=MODEL_CPU_THREADS,
        )
        logger.info("whisper model loaded")
    return model


@app.get("/", include_in_schema=False)
def root():
    return {"ok": True, "service": "whisper-service"}


@app.get("/health")
def health():
    return {
        "ok": True,
        "status": "healthy",
        "model": MODEL_NAME,
        "modelLoaded": model is not None,
        "maxAudioFileSizeMb": MAX_AUDIO_FILE_SIZE_MB,
        "authEnabled": bool(SERVICE_TOKEN),
    }


async def require_service_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(auth_scheme),
):
    if not SERVICE_TOKEN:
        return

    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="No autorizado")

    if not secrets.compare_digest(credentials.credentials, SERVICE_TOKEN):
        raise HTTPException(status_code=401, detail="No autorizado")


def enforce_rate_limit(request: Request):
    if RATE_LIMIT_PER_MINUTE <= 0:
        return

    key = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window_start = now - 60
    recent = [ts for ts in rate_buckets[key] if ts >= window_start]

    if len(recent) >= RATE_LIMIT_PER_MINUTE:
        rate_buckets[key] = recent
        raise HTTPException(status_code=429, detail="Demasiadas peticiones")

    recent.append(now)
    rate_buckets[key] = recent


def safe_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in ALLOWED_EXTENSIONS else ".audio"


TIME_RE = re.compile(r"time=(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")


def probe_audio_duration(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode == 0:
        raw = (result.stdout or "").strip()
        if raw and raw.upper() != "N/A":
            try:
                return float(raw)
            except ValueError:
                pass

    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-stats",
            "-i",
            path,
            "-map",
            "0:a:0",
            "-c",
            "copy",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    matches = TIME_RE.findall(result.stderr or "")
    if matches:
        h, m, s = matches[-1]
        return int(h) * 3600 + int(m) * 60 + float(s)

    raise ValueError("could not determine audio duration")


def format_paragraphs_from_segments(
    seg_list,
    max_paragraph_sec=22.0,
    pause_sec=0.7,
    min_words=8,
):
    paragraphs = []
    buf = []
    start_t = None
    last_end = None

    def buf_word_count():
        return len(" ".join(buf).split())

    def flush(force=False):
        nonlocal buf, start_t
        text = " ".join(" ".join(buf).split()).strip()
        if not text:
            buf = []
            start_t = None
            return

        if not force and buf_word_count() < min_words:
            return

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
                if start_t is None:
                    start_t = s

        buf.append(t)

        if t.endswith(".") or t.endswith("?") or t.endswith("!"):
            flush()

        if start_t is not None and (e - start_t) >= max_paragraph_sec:
            flush()

        last_end = e

    flush(force=True)
    return "\n\n".join(paragraphs).strip()


@app.post("/transcribe/file", dependencies=[Depends(require_service_token)])
async def transcribe_file(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form("es"),
    context: str = Form(""),
):
    enforce_rate_limit(request)

    if not file:
        raise HTTPException(status_code=400, detail="Falta archivo")

    if len(language) > 30 or len(context) > 300:
        raise HTTPException(status_code=400, detail="Parametros invalidos")

    suffix = Path(file.filename or "").suffix.lower()
    if file.content_type:
        if file.content_type not in ALLOWED_MIME:
            raise HTTPException(status_code=415, detail="Tipo de audio no permitido")
    elif suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Tipo de audio no permitido")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=safe_suffix(file.filename)) as tmp:
            tmp_path = tmp.name
            total = 0
            chunk_size = 1024 * 1024

            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Archivo demasiado grande (max {MAX_AUDIO_FILE_SIZE_MB}MB)",
                    )
                tmp.write(chunk)

        try:
            probed_duration = probe_audio_duration(tmp_path)
        except (ValueError, subprocess.TimeoutExpired):
            raise HTTPException(status_code=400, detail="Audio no valido")

        if probed_duration > MAX_AUDIO_DURATION_SEC:
            raise HTTPException(
                status_code=413,
                detail=f"Audio demasiado largo (max {MAX_AUDIO_DURATION_SEC // 60} min)",
            )

        whisper_model = get_model()
        segments_iter, info = whisper_model.transcribe(
            tmp_path,
            language=language if language else None,
            vad_filter=True,
            beam_size=5,
            best_of=5,
        )

        seg_list: list = []

        def consume_segments():
            try:
                for seg in segments_iter:
                    seg_list.append(seg)
            except GeneratorExit:
                return

        consume_task = asyncio.create_task(asyncio.to_thread(consume_segments))

        while True:
            try:
                await asyncio.wait_for(asyncio.shield(consume_task), timeout=0.5)
                break
            except asyncio.TimeoutError:
                pass

            if await request.is_disconnected():
                try:
                    segments_iter.close()
                except Exception:
                    pass
                logger.info("client disconnected, aborting transcription")
                try:
                    await asyncio.wait_for(consume_task, timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning("segment consumer still running after cancel signal")
                raise HTTPException(status_code=499, detail="Cliente desconectado")

        await consume_task
        raw_text = " ".join((seg.text or "").strip() for seg in seg_list).strip()
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

        paragraph_text = format_paragraphs_from_segments(
            seg_list,
            max_paragraph_sec=22.0,
            pause_sec=0.7,
            min_words=8,
        )

        duration = float(getattr(info, "duration", 0) or 0)

        return {
            "ok": True,
            "text": paragraph_text if paragraph_text else (raw_text if raw_text else "(sin texto)"),
            "rawText": raw_text,
            "durationSec": round(duration),
            "language": language,
            "type": "file",
            "segments": segments_out,
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("transcription failed")
        raise HTTPException(status_code=500, detail="Error procesando audio")
    finally:
        try:
            if tmp_path:
                os.remove(tmp_path)
        except Exception:
            logger.warning("temporary file cleanup failed")
