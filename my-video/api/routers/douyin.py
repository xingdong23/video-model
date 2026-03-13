from __future__ import annotations

import contextlib
import logging
import sys
import tempfile
from pathlib import Path

from fastapi import APIRouter

from ..config import get_settings
from ..dependencies import get_engine_manager
from ..errors import APIError, ErrorCode, success_response
from ..schemas import DouyinDownloadRequest, DouyinTranscribeRequest
from ..storage import get_storage

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


@router.post("/transcribe")
async def transcribe(req: DouyinTranscribeRequest):
    dt = _get_douyin_module()
    try:
        share_link = dt.extract_douyin_share_link(req.share_link)
    except Exception as e:
        raise APIError(ErrorCode.VALIDATION_ERROR, f"Invalid share link: {e}", 400)

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Download video first
            session = dt._create_session()
            try:
                modal_id, page_url = dt.resolve_douyin_page_url(share_link, session=session)
                play_url, title = dt.resolve_play_url(page_url, session=session)
                video_path = dt.download_video(
                    play_url, f"{title}_{modal_id}", download_dir=tmpdir, session=session,
                )
            finally:
                session.close()

            # Try using the shared subtitle engine's whisper model
            em = get_engine_manager()
            if em.subtitle_engine is not None:
                transcript = _transcribe_with_subtitle_engine(video_path)
            else:
                transcript = dt.transcribe_video(video_path, save_text_file=True)

            if not transcript:
                raise RuntimeError("语音识别完成，但未提取到有效文本")
        except APIError:
            raise
        except Exception as e:
            raise APIError(ErrorCode.ENGINE_ERROR, str(e))

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

    return success_response({"transcript": transcript, "files": saved})


@router.post("/download")
async def download(req: DouyinDownloadRequest):
    dt = _get_douyin_module()
    try:
        share_link = dt.extract_douyin_share_link(req.share_link)
    except Exception as e:
        raise APIError(ErrorCode.VALIDATION_ERROR, f"Invalid share link: {e}", 400)

    session = dt._create_session()
    try:
        _modal_id, page_url = dt.resolve_douyin_page_url(share_link, session=session)
        play_url, title = dt.resolve_play_url(page_url, session=session)

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = dt.download_video(
                play_url, title, download_dir=tmpdir, session=session,
            )
            storage = get_storage()
            file_id = storage.save(video_path, "douyin", f"{title}.mp4")
    except APIError:
        raise
    except Exception as e:
        raise APIError(ErrorCode.ENGINE_ERROR, f"Failed to download video: {e}")
    finally:
        session.close()

    return success_response({
        "file_id": file_id,
        "download_url": storage.get_url(file_id),
    })
