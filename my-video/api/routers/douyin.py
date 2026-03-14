from __future__ import annotations

import logging
import shutil
import sys
import tempfile
from pathlib import Path

from fastapi import APIRouter, Request

from ..config import get_settings
from ..dependencies import get_engine_manager
from ..errors import APIError, ErrorCode, success_response
from ..schemas import DouyinDownloadRequest, DouyinExistingRequest, DouyinTranscribeRequest
from ..project_store import get_project_store
from ..storage import get_storage
from ..task_manager import Task, get_task_manager, _make_progress_callback

router = APIRouter()
logger = logging.getLogger(__name__)


def _find_cached_video(modal_id: str) -> dict | None:
    """Check if a video with this douyin modal_id is already in storage."""
    if not modal_id or modal_id == "unknown":
        return None
    storage = get_storage()
    return storage.find_by_tag("douyin", "douyin_id", modal_id)


def _get_douyin_module():
    settings = get_settings()
    root = str(settings.my_video_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from douyin import douyin_transcript
    return douyin_transcript


def _transcribe_with_subtitle_engine(video_path: Path) -> str:
    """Use the shared subtitle engine's whisper model instead of loading a separate one."""
    em = get_engine_manager()
    if em.subtitle_engine is None:
        return ""

    import opencc
    result = em.subtitle_engine.model.transcribe(
        str(video_path),
        beam_size=5,
        language="zh",
    )
    segments, _ = result
    converter = opencc.OpenCC("t2s")
    lines = []
    for segment in segments:
        text = (segment.text or "").strip()
        text = converter.convert(text)
        if text:
            lines.append(text)
    return "\n\n".join(lines).strip()


def _execute_transcribe(task: Task, params: dict) -> dict:
    dt = _get_douyin_module()

    cb = _make_progress_callback(task)
    cb("downloading", 10, "解析抖音链接...")

    share_link = params["share_link"]

    # Resolve short link to full page URL first
    session = dt._create_session()
    try:
        modal_id, page_url = dt.resolve_douyin_page_url(share_link, session=session)
    except Exception:
        page_url = share_link
        modal_id = "unknown"
    finally:
        session.close()

    # Check cache: skip download if we already have this video
    cached = _find_cached_video(modal_id)
    if cached:
        logger.info("Using cached video for douyin_id=%s (file_id=%s)", modal_id, cached["file_id"])
        cb("transcribing", 30, "使用缓存视频，语音识别中...")

        storage = get_storage()
        video_path = storage.get_path(cached["file_id"])

        em = get_engine_manager()
        if em.subtitle_engine is not None:
            transcript = _transcribe_with_subtitle_engine(video_path)
        else:
            transcript = dt.transcribe_video(video_path, save_text_file=False)

        if not transcript:
            raise RuntimeError("语音识别完成，但未提取到有效文本")

        return {
            "transcript": transcript,
            "files": [{
                "file_id": cached["file_id"],
                "filename": cached["original_name"],
                "download_url": storage.get_url(cached["file_id"]),
            }],
        }

    cb("downloading", 20, "下载抖音视频中...")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            video_path, title = dt.download_video_ytdlp(
                page_url, download_dir=tmpdir,
            )
            logger.info("yt-dlp download succeeded: %s", title)
        except Exception as ytdlp_err:
            logger.warning("yt-dlp failed (%s), falling back to manual scraping", ytdlp_err)
            session = dt._create_session()
            try:
                play_url, title = dt.resolve_play_url(page_url, session=session)
                video_path = dt.download_video(
                    play_url, f"{title}_{modal_id}", download_dir=tmpdir, session=session,
                )
            finally:
                session.close()

        cb("transcribing", 40, "语音识别中...")
        em = get_engine_manager()
        if em.subtitle_engine is not None:
            transcript = _transcribe_with_subtitle_engine(video_path)
        else:
            transcript = dt.transcribe_video(video_path, save_text_file=True)

        if not transcript:
            raise RuntimeError("语音识别完成，但未提取到有效文本")

        cb("saving", 80, "保存文件...")
        storage = get_storage()
        saved = []
        for p in Path(tmpdir).iterdir():
            if p.is_file():
                tags = {"douyin_id": modal_id} if modal_id != "unknown" else None
                file_id = storage.save(p, "douyin", p.name, tags=tags)
                saved.append({
                    "file_id": file_id,
                    "filename": p.name,
                    "download_url": storage.get_url(file_id),
                })

    return {"transcript": transcript, "files": saved}


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@router.get("/videos")
async def list_videos():
    storage = get_storage()
    store = get_project_store()
    items = storage.list_category("douyin")
    videos = []
    for m in items:
        ext = Path(m.get("original_name", "")).suffix.lower()
        if ext not in VIDEO_EXTENSIONS:
            continue
        file_id = m["file_id"]
        # Find projects that used this video
        related = store.find_by_video(file_id)
        projects = [
            {
                "project_id": p["project_id"],
                "title": p.get("title", ""),
                "steps_done": len(p.get("steps", {})),
                "updated_at": p.get("updated_at", 0),
            }
            for p in related
        ]
        videos.append({
            "file_id": file_id,
            "filename": m["original_name"],
            "size_mb": round(m.get("size_bytes", 0) / 1048576, 1),
            "created_at": m.get("created_at"),
            "download_url": storage.get_url(file_id),
            "tags": m.get("tags", {}),
            "projects": projects,
        })
    return success_response({"videos": videos})


def _execute_transcribe_existing(task: Task, params: dict) -> dict:
    file_id = params["file_id"]
    storage = get_storage()
    video_path = storage.get_path(file_id)
    info = storage.get_info(file_id)

    cb = _make_progress_callback(task)
    cb("transcribing", 30, "语音识别中...")

    em = get_engine_manager()
    if em.subtitle_engine is not None:
        transcript = _transcribe_with_subtitle_engine(video_path)
    else:
        dt = _get_douyin_module()
        transcript = dt.transcribe_video(video_path, save_text_file=False)

    if not transcript:
        raise RuntimeError("语音识别完成，但未提取到有效文本")

    cb("done", 100, "转写完成")
    return {
        "transcript": transcript,
        "files": [{
            "file_id": file_id,
            "filename": info["original_name"],
            "download_url": storage.get_url(file_id),
        }],
    }


@router.post("/transcribe-existing", status_code=202)
async def transcribe_existing(req: DouyinExistingRequest, request: Request):
    storage = get_storage()
    try:
        storage.get_path(req.file_id)
    except FileNotFoundError:
        raise APIError(ErrorCode.VALIDATION_ERROR, f"文件不存在: {req.file_id}", 400)

    task_id = get_task_manager().submit(
        task_type="douyin/transcribe-existing",
        params={"file_id": req.file_id},
        executor_fn=_execute_transcribe_existing,
        request_id=request.headers.get("X-Request-ID"),
        gpu=False,
    )
    return success_response({"task_id": task_id})


@router.post("/transcribe", status_code=202)
async def transcribe(req: DouyinTranscribeRequest, request: Request):
    dt = _get_douyin_module()
    try:
        share_link = dt.extract_douyin_share_link(req.share_link)
    except Exception as e:
        raise APIError(ErrorCode.VALIDATION_ERROR, f"Invalid share link: {e}", 400)

    task_id = get_task_manager().submit(
        task_type="douyin/transcribe",
        params={"share_link": share_link},
        executor_fn=_execute_transcribe,
        request_id=request.headers.get("X-Request-ID"),
        callback_url=req.callback_url,
        gpu=False,
    )
    return success_response({"task_id": task_id})


def _execute_download(task: Task, params: dict) -> dict:
    dt = _get_douyin_module()

    cb = _make_progress_callback(task)
    cb("downloading", 10, "解析抖音链接...")

    share_link = params["share_link"]

    session = dt._create_session()
    try:
        modal_id, page_url = dt.resolve_douyin_page_url(share_link, session=session)
    except Exception:
        page_url = share_link
        modal_id = "unknown"
    finally:
        session.close()

    # Check cache
    cached = _find_cached_video(modal_id)
    if cached:
        logger.info("Using cached video for douyin_id=%s (file_id=%s)", modal_id, cached["file_id"])
        cb("done", 100, "使用缓存视频")
        storage = get_storage()
        return {
            "file_id": cached["file_id"],
            "download_url": storage.get_url(cached["file_id"]),
        }

    cb("downloading", 20, "下载抖音视频中...")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            video_path, title = dt.download_video_ytdlp(
                page_url, download_dir=tmpdir,
            )
            logger.info("yt-dlp download succeeded: %s", title)
        except Exception as ytdlp_err:
            logger.warning("yt-dlp failed (%s), falling back to manual scraping", ytdlp_err)
            session = dt._create_session()
            try:
                play_url, title = dt.resolve_play_url(page_url, session=session)
                video_path = dt.download_video(
                    play_url, title, download_dir=tmpdir, session=session,
                )
            finally:
                session.close()

        cb("saving", 80, "保存文件...")
        storage = get_storage()
        tags = {"douyin_id": modal_id} if modal_id != "unknown" else None
        file_id = storage.save(video_path, "douyin", f"{title}.mp4", tags=tags)

    return {
        "file_id": file_id,
        "download_url": storage.get_url(file_id),
    }


@router.post("/download", status_code=202)
async def download(req: DouyinDownloadRequest, request: Request):
    dt = _get_douyin_module()
    try:
        share_link = dt.extract_douyin_share_link(req.share_link)
    except Exception as e:
        raise APIError(ErrorCode.VALIDATION_ERROR, f"Invalid share link: {e}", 400)

    task_id = get_task_manager().submit(
        task_type="douyin/download",
        params={"share_link": share_link},
        executor_fn=_execute_download,
        request_id=request.headers.get("X-Request-ID"),
        callback_url=req.callback_url,
        gpu=False,
    )
    return success_response({"task_id": task_id})
