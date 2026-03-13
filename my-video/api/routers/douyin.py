from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

from fastapi import APIRouter, Request

from ..config import get_settings
from ..dependencies import get_engine_manager
from ..errors import APIError, ErrorCode, success_response
from ..schemas import DouyinDownloadRequest, DouyinTranscribeRequest
from ..storage import get_storage
from ..task_manager import Task, get_task_manager, _make_progress_callback

router = APIRouter()
logger = logging.getLogger(__name__)


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
    cb("downloading", 10, "下载抖音视频中...")

    with tempfile.TemporaryDirectory() as tmpdir:
        session = dt._create_session()
        try:
            modal_id, page_url = dt.resolve_douyin_page_url(params["share_link"], session=session)
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
                file_id = storage.save(p, "douyin", p.name)
                saved.append({
                    "file_id": file_id,
                    "filename": p.name,
                    "download_url": storage.get_url(file_id),
                })

    return {"transcript": transcript, "files": saved}


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
        gpu=False,
    )
    return success_response({"task_id": task_id})


def _execute_download(task: Task, params: dict) -> dict:
    dt = _get_douyin_module()

    cb = _make_progress_callback(task)
    cb("downloading", 10, "下载抖音视频中...")

    session = dt._create_session()
    try:
        _modal_id, page_url = dt.resolve_douyin_page_url(params["share_link"], session=session)
        play_url, title = dt.resolve_play_url(page_url, session=session)

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = dt.download_video(
                play_url, title, download_dir=tmpdir, session=session,
            )
            cb("saving", 80, "保存文件...")
            storage = get_storage()
            file_id = storage.save(video_path, "douyin", f"{title}.mp4")
    finally:
        session.close()

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
        gpu=False,
    )
    return success_response({"task_id": task_id})
