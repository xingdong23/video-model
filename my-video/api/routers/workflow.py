from __future__ import annotations

import logging

from fastapi import APIRouter

from ..dependencies import get_engine_manager
from ..errors import APIError, ErrorCode, success_response
from ..schemas import WorkflowRunRequest
from ..storage import get_storage

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/run")
async def workflow_run(req: WorkflowRunRequest):
    em = get_engine_manager()
    storage = get_storage()

    audio_path = None
    if req.audio_file_id:
        try:
            audio_path = str(storage.get_path(req.audio_file_id))
        except FileNotFoundError:
            raise APIError(ErrorCode.FILE_NOT_FOUND, f"Audio file not found: {req.audio_file_id}", 404)

    video_path = None
    if req.video_file_id:
        try:
            video_path = str(storage.get_path(req.video_file_id))
        except FileNotFoundError:
            raise APIError(ErrorCode.FILE_NOT_FOUND, f"Video file not found: {req.video_file_id}", 404)

    prompt_audio_path = None
    if req.prompt_audio_file_id:
        try:
            prompt_audio_path = str(storage.get_path(req.prompt_audio_file_id))
        except FileNotFoundError:
            raise APIError(ErrorCode.FILE_NOT_FOUND, f"Prompt audio file not found: {req.prompt_audio_file_id}", 404)

    wf = em.workflow_engine

    # Workflow uses voice + digital_human + subtitle engines internally.
    # Acquire all GPU locks to prevent conflicts with individual endpoint calls.
    logger.info("Workflow: acquiring engine locks ...")
    with em.voice_lock, em.digital_human_lock, em.subtitle_lock:
        logger.info("Workflow: locks acquired, starting run ...")
        try:
            result = wf.run(
                audio=audio_path,
                text=req.text,
                speaker=req.speaker,
                prompt_text=req.prompt_text,
                prompt_audio=prompt_audio_path,
                speed=req.speed,
                face=req.face,
                video=video_path,
                with_subtitles=req.with_subtitles,
                subtitle_correct=req.subtitle_correct,
                subtitle_language=req.subtitle_language,
                subtitle_max_chars=req.subtitle_max_chars,
                subtitle_beam_size=req.subtitle_beam_size,
                subtitle_best_of=req.subtitle_best_of,
                subtitle_vad_filter=req.subtitle_vad_filter,
                subtitle_vad_min_silence_ms=req.subtitle_vad_min_silence_ms,
                subtitle_speech_pad_ms=req.subtitle_speech_pad_ms,
                subtitle_api_key=req.subtitle_api_key,
                subtitle_api_base=req.subtitle_api_base,
                subtitle_llm_model=req.subtitle_llm_model,
                subtitle_request_timeout=req.subtitle_request_timeout,
                subtitle_font_path=req.subtitle_font_path,
                subtitle_font_name=req.subtitle_font_name,
                subtitle_font_index=req.subtitle_font_index,
                subtitle_font_size=req.subtitle_font_size,
                subtitle_font_color=req.subtitle_font_color,
                subtitle_outline_color=req.subtitle_outline_color,
                subtitle_outline=req.subtitle_outline,
                subtitle_wrap_style=req.subtitle_wrap_style,
                subtitle_bottom_margin=req.subtitle_bottom_margin,
                bgm_path=req.bgm_path,
                bgm_name=req.bgm_name,
                bgm_random=req.bgm_random,
                bgm_volume=req.bgm_volume,
                bgm_original_volume=req.bgm_original_volume,
                bgm_fade_out=req.bgm_fade_out,
                bgm_loop=req.bgm_loop,
                batch_size=req.batch_size,
                sync_offset=req.sync_offset,
                scale_h=req.scale_h,
                scale_w=req.scale_w,
                compress_inference=req.compress_inference,
                beautify_teeth=req.beautify_teeth,
                runtime=req.runtime,
            )
        except Exception as e:
            logger.exception("Workflow run failed")
            raise APIError(ErrorCode.ENGINE_ERROR, str(e))

    logger.info("Workflow: run complete, saving outputs ...")

    data = {
        "audio_generated": result.audio_generated,
        "subtitle_generated": result.subtitle_generated,
        "subtitle_burned": result.subtitle_burned,
        "bgm_applied": result.bgm_applied,
        "video_runtime": result.video_runtime,
        "video_runtime_description": result.video_runtime_description,
        "video_elapsed_seconds": result.video_elapsed_seconds,
    }

    fid = storage.save(result.final_video_path, "workflow", "workflow_final.mp4")
    data["final_video"] = {"file_id": fid, "download_url": storage.get_url(fid)}

    if result.audio_path and result.audio_path.exists():
        fid = storage.save(result.audio_path, "workflow", "workflow_audio.wav")
        data["audio"] = {"file_id": fid, "download_url": storage.get_url(fid)}

    if result.raw_video_path != result.final_video_path and result.raw_video_path.exists():
        fid = storage.save(result.raw_video_path, "workflow", "workflow_raw.mp4")
        data["raw_video"] = {"file_id": fid, "download_url": storage.get_url(fid)}

    if result.subtitle_path and result.subtitle_path.exists():
        fid = storage.save(result.subtitle_path, "workflow", "workflow.srt")
        data["subtitle"] = {"file_id": fid, "download_url": storage.get_url(fid)}

    return success_response(data)
