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

    # Resolve file IDs to paths
    audio_file_id = req.audio.file_id
    audio_path = None
    if audio_file_id:
        try:
            audio_path = str(storage.get_path(audio_file_id))
        except FileNotFoundError:
            raise APIError(ErrorCode.FILE_NOT_FOUND, f"Audio file not found: {audio_file_id}", 404)

    video_file_id = req.digital_human.video_file_id
    video_path = None
    if video_file_id:
        try:
            video_path = str(storage.get_path(video_file_id))
        except FileNotFoundError:
            raise APIError(ErrorCode.FILE_NOT_FOUND, f"Video file not found: {video_file_id}", 404)

    prompt_audio_path = None
    if req.audio.prompt_audio_file_id:
        try:
            prompt_audio_path = str(storage.get_path(req.audio.prompt_audio_file_id))
        except FileNotFoundError:
            raise APIError(ErrorCode.FILE_NOT_FOUND, f"Prompt audio file not found: {req.audio.prompt_audio_file_id}", 404)

    wf = em.workflow_engine

    logger.info("Workflow: starting run ...")
    try:
        result = wf.run(
            audio_config=req.audio,
            digital_human_config=req.digital_human,
            subtitle_config=req.subtitle,
            subtitle_style_config=req.subtitle_style,
            bgm_config=req.bgm,
            # Pass resolved paths for file_id fields
            audio=audio_path,
            video=video_path,
            prompt_audio=prompt_audio_path,
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

    fid = storage.save(result.final_video_path, "pipeline", "workflow_final.mp4")
    data["final_video"] = {"file_id": fid, "download_url": storage.get_url(fid)}

    if result.audio_path and result.audio_path.exists():
        fid = storage.save(result.audio_path, "pipeline", "workflow_audio.wav")
        data["audio"] = {"file_id": fid, "download_url": storage.get_url(fid)}

    if result.raw_video_path != result.final_video_path and result.raw_video_path.exists():
        fid = storage.save(result.raw_video_path, "pipeline", "workflow_raw.mp4")
        data["raw_video"] = {"file_id": fid, "download_url": storage.get_url(fid)}

    if result.subtitle_path and result.subtitle_path.exists():
        fid = storage.save(result.subtitle_path, "pipeline", "workflow.srt")
        data["subtitle"] = {"file_id": fid, "download_url": storage.get_url(fid)}

    return success_response(data)
