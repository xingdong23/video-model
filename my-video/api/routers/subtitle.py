from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from ..dependencies import get_engine_manager
from ..errors import APIError, ErrorCode, success_response
from ..schemas import SubtitleBurnRequest, SubtitleCorrectRequest, SubtitleGenerateSrtRequest
from ..storage import get_storage
from ..task_manager import Task, get_task_manager, _make_progress_callback

logger = logging.getLogger(__name__)

router = APIRouter()


def _execute_generate_srt(task: Task, params: dict) -> dict:
    em = get_engine_manager()
    storage = get_storage()

    cb = _make_progress_callback(task)
    cb("transcribing", 10, "语音识别中...")

    result = em.subtitle_engine.generate_srt(
        input_path=params["audio_path"],
        language=params["language"],
        max_chars=params["max_chars"],
        beam_size=params["beam_size"],
        best_of=params["best_of"],
        vad_filter=params["vad_filter"],
        vad_min_silence_ms=params["vad_min_silence_ms"],
        speech_pad_ms=params["speech_pad_ms"],
        apply_correction=params["apply_correction"],
        correction_api_key=params["correction_api_key"],
        correction_api_base=params["correction_api_base"],
        correction_model_name=params["correction_model_name"],
        correction_timeout=params["correction_timeout"],
    )

    cb("saving", 90, "保存字幕文件...")
    file_id = storage.save(result.srt_path, "subtitle", f"{params['audio_stem']}.srt")
    return {
        "file_id": file_id,
        "download_url": storage.get_url(file_id),
        "entries_count": result.entries_count,
        "detected_language": result.detected_language,
        "correction_applied": result.correction_applied,
    }


@router.post("/generate-srt", status_code=202)
async def generate_srt(req: SubtitleGenerateSrtRequest, request: Request):
    em = get_engine_manager()
    if em.subtitle_engine is None:
        raise APIError(ErrorCode.ENGINE_ERROR, "SubtitleEngine not loaded", 503)

    storage = get_storage()
    try:
        audio_path = storage.get_path(req.audio_file_id)
    except FileNotFoundError:
        raise APIError(ErrorCode.FILE_NOT_FOUND, f"Audio file not found: {req.audio_file_id}", 404)

    logger.info("generate-srt: input=%s language=%s", req.audio_file_id, req.language)
    task_id = get_task_manager().submit(
        task_type="subtitle/generate-srt",
        params={
            "audio_path": str(audio_path),
            "audio_stem": audio_path.stem,
            "language": req.language,
            "max_chars": req.max_chars,
            "beam_size": req.beam_size,
            "best_of": req.best_of,
            "vad_filter": req.vad_filter,
            "vad_min_silence_ms": req.vad_min_silence_ms,
            "speech_pad_ms": req.speech_pad_ms,
            "apply_correction": req.apply_correction,
            "correction_api_key": req.correction_api_key,
            "correction_api_base": req.correction_api_base,
            "correction_model_name": req.correction_model_name,
            "correction_timeout": req.correction_timeout,
        },
        executor_fn=_execute_generate_srt,
        request_id=request.headers.get("X-Request-ID"),
        callback_url=req.callback_url,
        gpu=True,
    )
    return success_response({"task_id": task_id})


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


def _execute_burn(task: Task, params: dict) -> dict:
    em = get_engine_manager()
    storage = get_storage()

    cb = _make_progress_callback(task)
    cb("burning", 10, "字幕烧录中...")

    result = em.subtitle_engine.burn_subtitles(
        video_path=params["video_path"],
        subtitle_path=params["srt_path"],
        font_path=params["font_path"],
        font_name=params["font_name"],
        font_index=params["font_index"],
        font_size=params["font_size"],
        font_color=params["font_color"],
        outline_color=params["outline_color"],
        outline=params["outline"],
        wrap_style=params["wrap_style"],
        bottom_margin=params["bottom_margin"],
    )

    cb("saving", 90, "保存视频...")
    file_id = storage.save(result.output_path, "subtitle", f"burned_{params['video_stem']}.mp4")
    return {
        "file_id": file_id,
        "download_url": storage.get_url(file_id),
    }


@router.post("/burn", status_code=202)
async def burn_subtitles(req: SubtitleBurnRequest, request: Request):
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
    task_id = get_task_manager().submit(
        task_type="subtitle/burn",
        params={
            "video_path": str(video_path),
            "video_stem": video_path.stem,
            "srt_path": str(srt_path),
            "font_path": req.font_path,
            "font_name": req.font_name,
            "font_index": req.font_index,
            "font_size": req.font_size,
            "font_color": req.font_color,
            "outline_color": req.outline_color,
            "outline": req.outline,
            "wrap_style": req.wrap_style,
            "bottom_margin": req.bottom_margin,
        },
        executor_fn=_execute_burn,
        request_id=request.headers.get("X-Request-ID"),
        callback_url=req.callback_url,
        gpu=False,
    )
    return success_response({"task_id": task_id})
