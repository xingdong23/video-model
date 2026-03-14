from __future__ import annotations

from fastapi import APIRouter, Request

from ..dependencies import get_engine_manager
from ..errors import APIError, ErrorCode, success_response
from ..schemas import DigitalHumanGenerateRequest
from ..storage import get_storage
from ..task_manager import Task, get_task_manager, _make_progress_callback

router = APIRouter()


@router.get("/faces")
async def list_faces():
    em = get_engine_manager()
    if em.digital_human_engine is None:
        raise APIError(ErrorCode.ENGINE_ERROR, "DigitalHumanEngine not loaded", 503)
    faces = em.digital_human_engine.list_face_videos()
    return success_response(faces)


def _execute_digital_human(task: Task, params: dict) -> dict:
    em = get_engine_manager()
    storage = get_storage()

    progress_cb = _make_progress_callback(task)
    result = em.digital_human_engine.generate(
        audio=params["audio_path"],
        face=params["face"],
        video=params.get("video_path"),
        batch_size=params["batch_size"],
        sync_offset=params["sync_offset"],
        scale_h=params["scale_h"],
        scale_w=params["scale_w"],
        compress_inference=params["compress_inference"],
        beautify_teeth=params["beautify_teeth"],
        runtime=params.get("runtime"),
        progress_callback=progress_cb,
    )

    file_id = storage.save(result.output_path, "avatar", "avatar.mp4")
    return {
        "file_id": file_id,
        "download_url": storage.get_url(file_id),
        "elapsed_seconds": result.elapsed_seconds,
        "runtime": result.runtime,
        "runtime_description": result.runtime_description,
    }


@router.post("/generate", status_code=202)
async def generate(req: DigitalHumanGenerateRequest, request: Request):
    em = get_engine_manager()
    if em.digital_human_engine is None:
        raise APIError(ErrorCode.ENGINE_ERROR, "DigitalHumanEngine not loaded", 503)

    storage = get_storage()
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

    task_id = get_task_manager().submit(
        task_type="avatar",
        params={
            "audio_path": audio_path,
            "face": req.face,
            "video_path": video_path,
            "batch_size": req.batch_size,
            "sync_offset": req.sync_offset,
            "scale_h": req.scale_h,
            "scale_w": req.scale_w,
            "compress_inference": req.compress_inference,
            "beautify_teeth": req.beautify_teeth,
            "runtime": req.runtime,
        },
        executor_fn=_execute_digital_human,
        request_id=request.headers.get("X-Request-ID"),
        callback_url=req.callback_url,
        gpu=True,
    )
    return success_response({"task_id": task_id})
