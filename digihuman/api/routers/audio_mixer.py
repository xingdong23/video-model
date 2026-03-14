from __future__ import annotations

from fastapi import APIRouter, Request

from ..dependencies import get_engine_manager
from ..errors import APIError, ErrorCode, success_response
from ..schemas import BgmMixRequest, BgmTrackItem
from ..storage import get_storage
from ..task_manager import Task, get_task_manager, _make_progress_callback

router = APIRouter()


@router.get("/tracks")
async def list_tracks():
    em = get_engine_manager()
    if em.bgm_engine is None:
        raise APIError(ErrorCode.ENGINE_ERROR, "BgmEngine not loaded", 503)
    tracks = em.bgm_engine.list_library_tracks()
    items = [BgmTrackItem(name=t.name, relative_path=str(t.relative_path)).model_dump() for t in tracks]
    return success_response(items)


def _execute_mix(task: Task, params: dict) -> dict:
    em = get_engine_manager()
    storage = get_storage()

    cb = _make_progress_callback(task)
    cb("mixing", 10, "混合背景音乐中...")

    result = em.bgm_engine.mix(
        video_path=params["video_path"],
        bgm_path=params["bgm_path"],
        bgm_name=params["bgm_name"],
        random_choice=params["random_choice"],
        bgm_volume=params["bgm_volume"],
        original_volume=params["original_volume"],
        loop_bgm=params["loop_bgm"],
        fade_out_seconds=params["fade_out_seconds"],
    )

    cb("saving", 90, "保存视频...")
    file_id = storage.save(result.output_path, "audio_mixer", f"mixed_{result.track_name}.mp4")
    return {
        "file_id": file_id,
        "download_url": storage.get_url(file_id),
        "track_name": result.track_name,
        "had_original_audio": result.had_original_audio,
    }


@router.post("/mix", status_code=202)
async def mix_bgm(req: BgmMixRequest, request: Request):
    em = get_engine_manager()
    if em.bgm_engine is None:
        raise APIError(ErrorCode.ENGINE_ERROR, "BgmEngine not loaded", 503)

    storage = get_storage()
    try:
        video_path = str(storage.get_path(req.video_file_id))
    except FileNotFoundError:
        raise APIError(ErrorCode.FILE_NOT_FOUND, f"Video file not found: {req.video_file_id}", 404)

    task_id = get_task_manager().submit(
        task_type="audio_mixer/mix",
        params={
            "video_path": video_path,
            "bgm_path": req.bgm_path,
            "bgm_name": req.bgm_name,
            "random_choice": req.random_choice,
            "bgm_volume": req.bgm_volume,
            "original_volume": req.original_volume,
            "loop_bgm": req.loop_bgm,
            "fade_out_seconds": req.fade_out_seconds,
        },
        executor_fn=_execute_mix,
        request_id=request.headers.get("X-Request-ID"),
        callback_url=req.callback_url,
        gpu=False,
    )
    return success_response({"task_id": task_id})
