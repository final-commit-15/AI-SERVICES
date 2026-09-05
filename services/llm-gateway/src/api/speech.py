import structlog
import base64
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Depends, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..router.router import ModelRouter
from ..dependencies import get_router
from ..config.settings import settings

logger = structlog.get_logger()

router = APIRouter()


class TranscribeRequest(BaseModel):
    model: Optional[str] = None
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: str = "json"
    temperature: float = 0.0


class SpeechRequest(BaseModel):
    model: Optional[str] = None
    voice: Optional[str] = None
    input: str
    response_format: str = "mp3"
    speed: float = 1.0


@router.post("/speech/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    http_request: Request = None,
    router: ModelRouter = Depends(get_router),
):
    """Transcribe audio to text (Speech-to-Text)."""
    try:
        # Check file size
        content = await file.read()
        max_size = 25 * 1024 * 1024  # 25MB
        if len(content) > max_size:
            raise HTTPException(status_code=413, detail="File too large (max 25MB)")

        # Route to speech-capable provider
        provider = router.route(task_type="speech", model=model)

        if not hasattr(provider, 'transcribe_audio'):
            raise HTTPException(status_code=501, detail="Provider does not support speech transcription")

        text = await provider.transcribe_audio(
            file=content,
            model=model or settings.speech_stt_model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
        )

        return {"text": text}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("transcribe_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/speech/synthesize")
async def synthesize_speech(
    request: SpeechRequest,
    http_request: Request,
    router: ModelRouter = Depends(get_router),
):
    """Synthesize speech from text (Text-to-Speech)."""
    try:
        provider = router.route(task_type="speech", model=request.model)

        if not hasattr(provider, 'synthesize_speech'):
            raise HTTPException(status_code=501, detail="Provider does not support speech synthesis")

        audio_data = await provider.synthesize_speech(
            text=request.input,
            model=request.model or settings.speech_tts_model,
            voice=request.voice or settings.speech_tts_voice,
            response_format=request.response_format,
            speed=request.speed,
        )

        return StreamingResponse(
            iter([audio_data]),
            media_type=f"audio/{request.response_format}",
            headers={"Content-Disposition": f"attachment; filename=speech.{request.response_format}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("synthesize_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/speech/transcribe/base64")
async def transcribe_base64(
    request: TranscribeRequest,
    audio_base64: str,
    http_request: Request = None,
    router: ModelRouter = Depends(get_router),
):
    """Transcribe base64-encoded audio."""
    try:
        audio_data = base64.b64decode(audio_base64)

        provider = router.route(task_type="speech", model=request.model)

        if not hasattr(provider, 'transcribe_audio'):
            raise HTTPException(status_code=501, detail="Provider does not support speech transcription")

        text = await provider.transcribe_audio(
            file=audio_data,
            model=request.model or settings.speech_stt_model,
            language=request.language,
            prompt=request.prompt,
            response_format=request.response_format,
            temperature=request.temperature,
        )

        return {"text": text}
    except Exception as e:
        logger.error("transcribe_base64_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))