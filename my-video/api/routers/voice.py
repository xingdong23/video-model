from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import StreamingResponse

from ..dependencies import get_engine_manager
from ..errors import APIError, ErrorCode, success_response
from ..schemas import SpeakerItem, VoiceSynthesizeRequest
from ..storage import get_storage
from ..task_manager import Task, get_task_manager, _make_progress_callback

router = APIRouter()
logger = logging.getLogger(__name__)

def _resolve_ffmpeg() -> str:
    import shutil
    which = shutil.which("ffmpeg")
    if which:
        return which
    # Fallback: check conda env
    conda_ffmpeg = Path(sys.executable).parent / "ffmpeg"
    if conda_ffmpeg.exists():
        return str(conda_ffmpeg)
    return "ffmpeg"


FFMPEG_BIN = _resolve_ffmpeg()

_SPLIT_CHARS = "。？！\n"
_MAX_CHUNK_CHARS = 500


def _split_text(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Split long text into chunks ≤ *max_chars* at sentence boundaries.

    Split points: 。？！ and newline.  If a single sentence exceeds
    *max_chars* it is kept as-is (the TTS engine handles further splitting
    internally).
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for char in text:
        current += char
        if char in _SPLIT_CHARS and len(current.strip()) > 0:
            chunks.append(current.strip())
            current = ""
    if current.strip():
        chunks.append(current.strip())

    # Merge small adjacent chunks so each is as close to max_chars as possible
    merged: list[str] = []
    buf = ""
    for chunk in chunks:
        if buf and len(buf) + len(chunk) > max_chars:
            merged.append(buf)
            buf = chunk
        else:
            buf = buf + chunk if buf else chunk
    if buf:
        merged.append(buf)
    return merged


def _concat_wav_files(wav_paths: list[str | Path], output_path: str | Path) -> Path:
    """Concatenate multiple WAV files using ffmpeg concat demuxer."""
    import tempfile as _tmpmod

    output_path = Path(output_path)
    if len(wav_paths) == 1:
        shutil.copy2(str(wav_paths[0]), str(output_path))
        return output_path

    with _tmpmod.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="concat_"
    ) as flist:
        for p in wav_paths:
            flist.write("file '%s'\n" % str(Path(p).resolve()))
        list_path = flist.name

    try:
        cmd = [
            FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0",
            "-i", list_path, "-c", "copy", str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError("ffmpeg concat failed: %s" % result.stderr[:300])
    finally:
        Path(list_path).unlink(missing_ok=True)
    return output_path


def _wav_to_mp3(wav_path: str | Path, mp3_path: str | Path) -> Path:
    """Convert WAV to MP3 (128kbps) using ffmpeg. Returns mp3_path."""
    mp3_path = Path(mp3_path)
    cmd = [
        FFMPEG_BIN, "-y", "-i", str(wav_path),
        "-codec:a", "libmp3lame", "-b:a", "128k",
        "-ar", "24000", "-ac", "1",
        str(mp3_path),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        logger.warning("ffmpeg WAV->MP3 failed: %s", result.stderr[:200])
        raise RuntimeError(f"ffmpeg failed: {result.stderr[:200]}")
    return mp3_path


@router.get("/speakers")
async def list_speakers():
    em = get_engine_manager()
    if em.voice_engine is None:
        raise APIError(ErrorCode.ENGINE_ERROR, "VoiceEngine not loaded", 503)
    speakers = em.voice_engine.list_speakers()
    items = [SpeakerItem(name=s, vid=i + 1).model_dump() for i, s in enumerate(speakers)]
    return success_response(items)


def _execute_synthesize(task: Task, params: dict) -> dict:
    em = get_engine_manager()
    storage = get_storage()
    engine = em.voice_engine

    cb = _make_progress_callback(task)
    cb("synthesizing", 5, "语音合成中...")

    text = params["text"]
    chunks = _split_text(text)
    total_chunks = len(chunks)

    chunk_wavs: list[str] = []
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        for ci, chunk in enumerate(chunks):
            cb("synthesizing", int(5 + ci / total_chunks * 75),
               f"语音合成中... ({ci + 1}/{total_chunks} 块)")

            if total_chunks == 1:
                chunk_out = tmp_path
            else:
                chunk_out = tmp_path.replace(".wav", f"_chunk{ci}.wav")
                chunk_wavs.append(chunk_out)

            engine.synthesize_to_file(
                text=chunk, speaker=params["speaker"],
                output_path=chunk_out, speed=params["speed"],
            )

            # Release GPU memory between chunks
            if total_chunks > 1:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

        # Concatenate chunks if needed
        if total_chunks > 1:
            cb("synthesizing", 80, "拼接音频分段...")
            _concat_wav_files(chunk_wavs, tmp_path)

        cb("encoding", 85, "转码MP3中...")
        mp3_path = tmp_path.replace(".wav", ".mp3")
        try:
            _wav_to_mp3(tmp_path, mp3_path)
            save_path, save_name = mp3_path, f"{params['speaker']}.mp3"
        except Exception:
            logger.warning("MP3 encoding failed, saving as WAV", exc_info=True)
            save_path, save_name = tmp_path, f"{params['speaker']}.wav"

        cb("saving", 90, "保存文件...")
        file_id = storage.save(save_path, "voice", save_name)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        Path(tmp_path.replace(".wav", ".mp3")).unlink(missing_ok=True)
        for cw in chunk_wavs:
            Path(cw).unlink(missing_ok=True)

    return {
        "file_id": file_id,
        "download_url": storage.get_url(file_id),
    }


@router.post("/synthesize", status_code=202)
async def synthesize(req: VoiceSynthesizeRequest, request: Request):
    em = get_engine_manager()
    if em.voice_engine is None:
        raise APIError(ErrorCode.ENGINE_ERROR, "VoiceEngine not loaded", 503)

    task_id = get_task_manager().submit(
        task_type="voice/synthesize",
        params={"text": req.text, "speaker": req.speaker, "speed": req.speed},
        executor_fn=_execute_synthesize,
        request_id=request.headers.get("X-Request-ID"),
        callback_url=req.callback_url,
        gpu=True,
    )
    return success_response({"task_id": task_id})


@router.post("/synthesize/stream")
async def synthesize_stream(req: VoiceSynthesizeRequest):
    em = get_engine_manager()
    if em.voice_engine is None:
        raise APIError(ErrorCode.ENGINE_ERROR, "VoiceEngine not loaded", 503)

    try:
        wav_bytes = em.voice_engine.synthesize_to_wav_bytes(
            text=req.text, speaker=req.speaker, speed=req.speed
        )
    except Exception as e:
        raise APIError(ErrorCode.ENGINE_ERROR, str(e))

    import io
    return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav")


def _execute_zero_shot(task: Task, params: dict) -> dict:
    em = get_engine_manager()
    storage = get_storage()
    engine = em.voice_engine

    cb = _make_progress_callback(task)
    cb("synthesizing", 5, "零样本语音合成中...")

    text = params["text"]
    chunks = _split_text(text)
    total_chunks = len(chunks)

    chunk_wavs: list[str] = []
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_tmp:
        out_path = out_tmp.name
    try:
        for ci, chunk in enumerate(chunks):
            cb("synthesizing", int(5 + ci / total_chunks * 75),
               f"零样本合成中... ({ci + 1}/{total_chunks} 块)")

            if total_chunks == 1:
                chunk_out = out_path
            else:
                chunk_out = out_path.replace(".wav", f"_chunk{ci}.wav")
                chunk_wavs.append(chunk_out)

            engine.synthesize_zero_shot_to_file(
                text=chunk,
                prompt_text=params["prompt_text"],
                prompt_audio_path=params["prompt_path"],
                output_path=chunk_out,
                speed=params["speed"],
            )

            # Release GPU memory between chunks
            if total_chunks > 1:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

        # Concatenate chunks if needed
        if total_chunks > 1:
            cb("synthesizing", 80, "拼接音频分段...")
            _concat_wav_files(chunk_wavs, out_path)

        cb("encoding", 85, "转码MP3中...")
        mp3_path = out_path.replace(".wav", ".mp3")
        try:
            _wav_to_mp3(out_path, mp3_path)
            save_path, save_name = mp3_path, "zero_shot.mp3"
        except Exception:
            logger.warning("MP3 encoding failed, saving as WAV", exc_info=True)
            save_path, save_name = out_path, "zero_shot.wav"

        cb("saving", 90, "保存文件...")
        file_id = storage.save(save_path, "voice", save_name)
    finally:
        Path(params["prompt_path"]).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)
        Path(out_path.replace(".wav", ".mp3")).unlink(missing_ok=True)
        for cw in chunk_wavs:
            Path(cw).unlink(missing_ok=True)

    return {
        "file_id": file_id,
        "download_url": storage.get_url(file_id),
    }


@router.post("/zero-shot", status_code=202)
async def zero_shot(
    text: str = Form(...),
    prompt_text: str = Form(...),
    speed: float = Form(1.0),
    prompt_wav: UploadFile = File(...),
    callback_url: Optional[str] = Form(None),
    request: Request = None,
):
    em = get_engine_manager()
    if em.voice_engine is None:
        raise APIError(ErrorCode.ENGINE_ERROR, "VoiceEngine not loaded", 503)

    suffix = Path(prompt_wav.filename or "prompt.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await prompt_wav.read())
        prompt_path = tmp.name

    task_id = get_task_manager().submit(
        task_type="voice/zero-shot",
        params={
            "text": text,
            "prompt_text": prompt_text,
            "prompt_path": prompt_path,
            "speed": speed,
        },
        executor_fn=_execute_zero_shot,
        request_id=request.headers.get("X-Request-ID") if request else None,
        callback_url=callback_url,
        gpu=True,
    )
    return success_response({"task_id": task_id})


@router.post("/voices/export")
async def export_voice(
    voice_name: str = Form(...),
    prompt_text: str = Form(...),
    prompt_wav: UploadFile = File(...),
):
    em = get_engine_manager()
    if em.voice_engine is None:
        raise APIError(ErrorCode.ENGINE_ERROR, "VoiceEngine not loaded", 503)

    suffix = Path(prompt_wav.filename or "prompt.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await prompt_wav.read())
        prompt_path = tmp.name

    try:
        try:
            saved = em.voice_engine.export_custom_voice(
                voice_name=voice_name,
                prompt_text=prompt_text,
                prompt_audio_path=prompt_path,
            )
        except Exception as e:
            raise APIError(ErrorCode.ENGINE_ERROR, str(e))
    finally:
        Path(prompt_path).unlink(missing_ok=True)

    return success_response({
        "voice_name": voice_name,
        "path": str(saved),
    })
