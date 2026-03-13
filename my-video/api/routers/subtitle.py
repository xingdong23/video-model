from __future__ import annotations

import logging

from fastapi import APIRouter

from ..dependencies import get_engine_manager
from ..errors import APIError, ErrorCode, success_response
from ..schemas import SubtitleBurnRequest, SubtitleCorrectRequest, SubtitleGenerateSrtRequest
from ..storage import get_storage

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/generate-srt")
async def generate_srt(req: SubtitleGenerateSrtRequest):
    em = get_engine_manager()
    if em.subtitle_engine is None:
        raise APIError(ErrorCode.ENGINE_ERROR, "SubtitleEngine not loaded", 503)

    storage = get_storage()
    try:
        audio_path = storage.get_path(req.audio_file_id)
    except FileNotFoundError:
        raise APIError(ErrorCode.FILE_NOT_FOUND, f"Audio file not found: {req.audio_file_id}", 404)

    logger.info("generate-srt: input=%s language=%s", req.audio_file_id, req.language)
    with em.subtitle_lock:
        try:
            result = em.subtitle_engine.generate_srt(
                input_path=str(audio_path),
                language=req.language,
                max_chars=req.max_chars,
                beam_size=req.beam_size,
                best_of=req.best_of,
                vad_filter=req.vad_filter,
                vad_min_silence_ms=req.vad_min_silence_ms,
                speech_pad_ms=req.speech_pad_ms,
                apply_correction=req.apply_correction,
                correction_api_key=req.correction_api_key,
                correction_api_base=req.correction_api_base,
                correction_model_name=req.correction_model_name,
                correction_timeout=req.correction_timeout,
            )
        except Exception as e:
            logger.exception("generate-srt failed for %s", req.audio_file_id)
            raise APIError(ErrorCode.ENGINE_ERROR, str(e))

    file_id = storage.save(result.srt_path, "subtitle", f"{audio_path.stem}.srt")
    logger.info("generate-srt: done, file_id=%s entries=%d", file_id, result.entries_count)
    return success_response({
        "file_id": file_id,
        "download_url": storage.get_url(file_id),
        "entries_count": result.entries_count,
        "detected_language": result.detected_language,
        "correction_applied": result.correction_applied,
    })


@router.post("/correct")
async def correct_subtitles(req: SubtitleCorrectRequest):
    em = get_engine_manager()
    if em.subtitle_engine is None:
        raise APIError(ErrorCode.ENGINE_ERROR, "SubtitleEngine not loaded", 503)

    storage = get_storage()
    try:
        srt_path = storage.get_path(req.srt_file_id)
    except FileNotFoundError:
        raise APIError(ErrorCode.FILE_NOT_FOUND, f"SRT file not found: {req.srt_file_id}", 404)

    logger.info("correct: input=%s", req.srt_file_id)
    with em.subtitle_lock:
        try:
            result = em.subtitle_engine.correct_subtitles(
                subtitle_path=str(srt_path),
                api_key=req.api_key,
                api_base=req.api_base,
                model_name=req.model_name,
                request_timeout=req.request_timeout,
            )
        except Exception as e:
            logger.exception("correct failed for %s", req.srt_file_id)
            raise APIError(ErrorCode.ENGINE_ERROR, str(e))

    file_id = storage.save(result.srt_path, "subtitle", f"corrected_{srt_path.stem}.srt")
    logger.info("correct: done, file_id=%s", file_id)
    return success_response({
        "file_id": file_id,
        "download_url": storage.get_url(file_id),
        "entries_count": result.entries_count,
    })


@router.post("/burn")
async def burn_subtitles(req: SubtitleBurnRequest):
    em = get_engine_manager()
    if em.subtitle_engine is None:
        raise APIError(ErrorCode.ENGINE_ERROR, "SubtitleEngine not loaded", 503)

    storage = get_storage()
    try:
        video_path = storage.get_path(req.video_file_id)
    except FileNotFoundError:
        raise APIError(ErrorCode.FILE_NOT_FOUND, f"Video file not found: {req.video_file_id}", 404)
    try:
        srt_path = storage.get_path(req.srt_file_id)
    except FileNotFoundError:
        raise APIError(ErrorCode.FILE_NOT_FOUND, f"SRT file not found: {req.srt_file_id}", 404)

    logger.info("burn: video=%s srt=%s", req.video_file_id, req.srt_file_id)
    with em.subtitle_lock:
        try:
            result = em.subtitle_engine.burn_subtitles(
                video_path=str(video_path),
                subtitle_path=str(srt_path),
                font_path=req.font_path,
                font_name=req.font_name,
                font_index=req.font_index,
                font_size=req.font_size,
                font_color=req.font_color,
                outline_color=req.outline_color,
                outline=req.outline,
                wrap_style=req.wrap_style,
                bottom_margin=req.bottom_margin,
            )
        except Exception as e:
            logger.exception("burn failed for video=%s srt=%s", req.video_file_id, req.srt_file_id)
            raise APIError(ErrorCode.ENGINE_ERROR, str(e))

    file_id = storage.save(result.output_path, "subtitle", f"burned_{video_path.stem}.mp4")
    logger.info("burn: done, file_id=%s", file_id)
    return success_response({
        "file_id": file_id,
        "download_url": storage.get_url(file_id),
    })
