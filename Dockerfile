FROM python:3.11-slim

WORKDIR /app

ENV HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG WHISPER_MODEL=small
ARG WHISPER_COMPUTE_TYPE=int8
ENV WHISPER_MODEL=${WHISPER_MODEL} \
    WHISPER_COMPUTE_TYPE=${WHISPER_COMPUTE_TYPE}
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='${WHISPER_COMPUTE_TYPE}')"

ENV HF_HUB_OFFLINE=1

COPY main.py .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
