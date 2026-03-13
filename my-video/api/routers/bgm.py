from __future__ import annotations

from fastapi import APIRouter

from ..dependencies import get_engine_manager
from ..errors import APIError, ErrorCode, success_response
from ..schemas import BgmMixRequest, BgmTrackItem
from ..storage import get_storage

router = APIRouter()


@router.get("/tracks")
async def list_tracks():
    em = get_engine_manager()
    if em.bgm_engine is None:
        raise APIError(ErrorCode.ENGINE_ERROR, "BgmEngine not loaded", 503)
    tracks = em.bgm_engine.list_library_tracks()
    items = [BgmTrackItem(name=t.name, relative_path=str(t.relative_path)).model_dump() for t in tracks]
    return success_response(items)


@router.post("/mix")
async def mix_bgm(req: BgmMixRequest):
    em = get_engine_manager()
    if em.bgm_engine is None:
        raise APIError(ErrorCode.ENGINE_ERROR, "BgmEngine not loaded", 503)

    storage = get_storage()
    try:
        video_path = storage.get_path(req.video_file_id)
    except FileNotFoundError:
        raise APIError(ErrorCode.FILE_NOT_FOUND, f"Video file not found: {req.video_file_id}", 404)

    try:
        result = em.bgm_engine.mix(
            video_path=str(video_path),
            bgm_path=req.bgm_path,
            bgm_name=req.bgm_name,
            random_choice=req.random_choice,
            bgm_volume=req.bgm_volume,
            original_volume=req.original_volume,
            loop_bgm=req.loop_bgm,
            fade_out_seconds=req.fade_out_seconds,
        )
    except Exception as e:
        raise APIError(ErrorCode.ENGINE_ERROR, str(e))

    file_id = storage.save(result.output_path, "bgm", f"mixed_{result.track_name}.mp4")
    return success_response({
        "file_id": file_id,
        "download_url": storage.get_url(file_id),
        "track_name": result.track_name,
        "had_original_audio": result.had_original_audio,
    })
