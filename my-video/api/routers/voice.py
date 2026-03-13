from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from ..dependencies import get_engine_manager
from ..errors import APIError, ErrorCode, success_response
from ..schemas import SpeakerItem, VoiceSynthesizeRequest
from ..storage import get_storage

router = APIRouter()


@router.get("/speakers")
async def list_speakers():
    em = get_engine_manager()
    if em.voice_engine is None:
        raise APIError(ErrorCode.ENGINE_ERROR, "VoiceEngine not loaded", 503)
    speakers = em.voice_engine.list_speakers()
    items = [SpeakerItem(name=s, vid=i + 1).model_dump() for i, s in enumerate(speakers)]
    return success_response(items)


@router.post("/synthesize")
async def synthesize(req: VoiceSynthesizeRequest):
    em = get_engine_manager()
    if em.voice_engine is None:
        raise APIError(ErrorCode.ENGINE_ERROR, "VoiceEngine not loaded", 503)

    with em.voice_lock:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            em.voice_engine.synthesize_to_file(
                text=req.text, speaker=req.speaker, output_path=tmp_path, speed=req.speed
            )
        except Exception as e:
            raise APIError(ErrorCode.ENGINE_ERROR, str(e))

    storage = get_storage()
    file_id = storage.save(tmp_path, "voice", f"{req.speaker}.wav")
    Path(tmp_path).unlink(missing_ok=True)
    return success_response({
        "file_id": file_id,
        "download_url": storage.get_url(file_id),
    })


@router.post("/synthesize/stream")
async def synthesize_stream(req: VoiceSynthesizeRequest):
    em = get_engine_manager()
    if em.voice_engine is None:
        raise APIError(ErrorCode.ENGINE_ERROR, "VoiceEngine not loaded", 503)

    with em.voice_lock:
        try:
            wav_bytes = em.voice_engine.synthesize_to_wav_bytes(
                text=req.text, speaker=req.speaker, speed=req.speed
            )
        except Exception as e:
            raise APIError(ErrorCode.ENGINE_ERROR, str(e))

    import io
    return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav")


@router.post("/zero-shot")
async def zero_shot(
    text: str = Form(...),
    prompt_text: str = Form(...),
    speed: float = Form(1.0),
    prompt_wav: UploadFile = File(...),
):
    em = get_engine_manager()
    if em.voice_engine is None:
        raise APIError(ErrorCode.ENGINE_ERROR, "VoiceEngine not loaded", 503)

    suffix = Path(prompt_wav.filename or "prompt.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await prompt_wav.read())
        prompt_path = tmp.name

    try:
        with em.voice_lock:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_tmp:
                    out_path = out_tmp.name
                em.voice_engine.synthesize_zero_shot_to_file(
                    text=text,
                    prompt_text=prompt_text,
                    prompt_audio_path=prompt_path,
                    output_path=out_path,
                    speed=speed,
                )
            except Exception as e:
                raise APIError(ErrorCode.ENGINE_ERROR, str(e))
    finally:
        Path(prompt_path).unlink(missing_ok=True)

    storage = get_storage()
    file_id = storage.save(out_path, "voice", "zero_shot.wav")
    Path(out_path).unlink(missing_ok=True)
    return success_response({
        "file_id": file_id,
        "download_url": storage.get_url(file_id),
    })


@router.post("/voices/export")
async def export_voice(
    voice_name: str = Form(...),
    prompt_text: str = Form(...),
    prompt_wav: UploadFile = File(...),
):
    em = get_engine_manager()
    if em.voice_engine is None:
        raise APIError(ErrorCode.ENGINE_ERROR, "VoiceEngine not loaded", 503)

    suffix = Path(prompt_wav.filename or "prompt.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await prompt_wav.read())
        prompt_path = tmp.name

    try:
        with em.voice_lock:
            try:
                saved = em.voice_engine.export_custom_voice(
                    voice_name=voice_name,
                    prompt_text=prompt_text,
                    prompt_audio_path=prompt_path,
                )
            except Exception as e:
                raise APIError(ErrorCode.ENGINE_ERROR, str(e))
    finally:
        Path(prompt_path).unlink(missing_ok=True)

    return success_response({
        "voice_name": voice_name,
        "path": str(saved),
    })
