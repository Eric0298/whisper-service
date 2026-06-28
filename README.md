# whisper-service

Servicio FastAPI interno para transcribir audio con `faster-whisper`.

## Desarrollo

```bash
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Variables

Copia `.env.example` a `.env` y ajusta valores reales fuera de Git.

- `WHISPER_SERVICE_TOKEN`: token Bearer que debe enviar JustWriteIt.
- `MAX_AUDIO_FILE_SIZE_MB`: limite maximo aceptado por el servicio.
- `ALLOWED_ORIGINS`: origenes CORS permitidos si se expone a navegador.
- `RATE_LIMIT_PER_MINUTE`: limite simple por IP.
- `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`: configuracion de modelo.

## Seguridad

El endpoint costoso `/transcribe/file` exige `Authorization: Bearer <token>` cuando
`WHISPER_SERVICE_TOKEN` esta definido. Los errores internos se registran en logs y la
respuesta publica se mantiene generica.
