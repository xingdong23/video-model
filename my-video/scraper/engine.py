from __future__ import annotations

import argparse
import contextlib
import html
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DOWNLOAD_DIR = PACKAGE_ROOT / "downloads"
DEFAULT_MODEL_DIR = PACKAGE_ROOT / "models"
SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
DEFAULT_COOKIE_ENV = "DOUYIN_COOKIE"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
DEFAULT_HTML_ACCEPT = (
    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,"
    "image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
)
TRAILING_URL_CHARS = "\"'”’）)]】>}，。！？；;,.!?"


class TranscriptError(RuntimeError):
    """Raised when transcript extraction fails."""


def _import_required(module_name: str, install_hint: str):
    try:
        return __import__(module_name, fromlist=["*"])
    except ModuleNotFoundError as exc:
        raise TranscriptError(install_hint) from exc


def _import_optional(module_name: str):
    try:
        return __import__(module_name, fromlist=["*"])
    except ModuleNotFoundError:
        return None


def _normalize_candidate_url(candidate: str) -> str:
    normalized = html.unescape(candidate or "")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized.rstrip(TRAILING_URL_CHARS)


def _resolve_cookie(cookie: str | None = None) -> str | None:
    if cookie:
        return cookie.strip() or None
    env_cookie = os.getenv(DEFAULT_COOKIE_ENV, "").strip()
    return env_cookie or None


def _build_headers(
    *,
    cookie: str | None = None,
    accept_html: bool = False,
    referer: str | None = None,
) -> dict[str, str]:
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    if accept_html:
        headers["Accept"] = DEFAULT_HTML_ACCEPT
    if referer:
        headers["Referer"] = referer
    if cookie:
        headers["Cookie"] = cookie
    return headers


def _create_session(cookie: str | None = None):
    requests = _import_required("requests", "缺少依赖 requests，请先安装 requirements.txt")
    session = requests.Session()
    session.headers.update(_build_headers(cookie=_resolve_cookie(cookie)))
    return session


def extract_douyin_share_link(text: str) -> str:
    cleaned_text = html.unescape(text or "")
    patterns = (
        r"https?://v\.douyin\.com/\S+",
        r"https?://(?:www\.)?douyin\.com/video/\d+[^\s\"'<>]*",
        r"https?://(?:www\.)?iesdouyin\.com/share/video/\d+[^\s\"'<>]*",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned_text)
        if match:
            return _normalize_candidate_url(match.group(0))
    raise TranscriptError("未找到有效的抖音视频链接")


def _extract_modal_id_from_url(url: str) -> str | None:
    patterns = (
        r"/video/(\d+)",
        r"/share/video/(\d+)",
        r"modal_id=(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _extract_modal_id_from_html(page_html: str) -> str | None:
    patterns = (
        r'"awemeId":"(\d+)"',
        r'"aweme_id":"(\d+)"',
        r'"itemId":"(\d+)"',
        r'"group_id":"(\d+)"',
        r"/video/(\d+)",
        r"modal_id=(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, page_html)
        if match:
            return match.group(1)
    return None


def resolve_douyin_page_url(
    link_text: str,
    *,
    session,
    cookie: str | None = None,
) -> tuple[str, str]:
    share_link = extract_douyin_share_link(link_text)
    modal_id = _extract_modal_id_from_url(share_link)
    if modal_id:
        return modal_id, f"https://www.douyin.com/video/{modal_id}"

    response = session.get(
        share_link,
        headers=_build_headers(cookie=_resolve_cookie(cookie), accept_html=True),
        allow_redirects=True,
        timeout=30,
    )
    response.raise_for_status()

    modal_id = _extract_modal_id_from_url(response.url) or _extract_modal_id_from_html(response.text)
    if not modal_id:
        raise TranscriptError(f"抖音链接解析失败，无法识别视频 ID: {response.url}")

    return modal_id, f"https://www.douyin.com/video/{modal_id}"


def _find_key(node, target_key: str):
    if isinstance(node, dict):
        if target_key in node:
            return node[target_key]
        for value in node.values():
            found = _find_key(value, target_key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_key(value, target_key)
            if found is not None:
                return found
    return None


def _resolve_play_url_from_render_data(render_data: dict) -> tuple[str, str]:
    video_detail = _find_key(render_data, "videoDetail")
    if not isinstance(video_detail, dict):
        raise TranscriptError("未从页面中解析到 videoDetail")

    bit_rate_list = (
        video_detail.get("video", {}).get("bitRateList")
        if isinstance(video_detail.get("video"), dict)
        else None
    )
    if not bit_rate_list:
        raise TranscriptError("未从页面中解析到视频播放地址")

    play_addr = bit_rate_list[0].get("playAddr") or []
    if not play_addr:
        raise TranscriptError("未从页面中解析到视频播放地址")

    play_url = play_addr[0].get("src") if isinstance(play_addr[0], dict) else None
    if not play_url:
        raise TranscriptError("未从页面中解析到视频播放地址")

    title = video_detail.get("desc") or "douyin_video"
    if not play_url.startswith("https://"):
        play_url = "https://" + play_url.lstrip("/")

    return play_url, title


def _extract_render_data(page_html: str) -> dict:
    match = re.search(
        r'<script id="RENDER_DATA" type="application/json">(.*?)</script>',
        page_html,
        flags=re.DOTALL,
    )
    if not match:
        raise TranscriptError("页面中未找到 RENDER_DATA")
    return json.loads(unquote(match.group(1)))


def _decode_play_url(raw_value: str) -> str:
    decoded = html.unescape(raw_value or "").strip().strip('"')
    decoded = decoded.replace("\\u002F", "/").replace("\\/", "/")
    if decoded.startswith("//"):
        return "https:" + decoded
    if decoded.startswith("http://"):
        return decoded.replace("http://", "https://", 1)
    return decoded


def _extract_title_from_html(page_html: str) -> str:
    match = re.search(r"<title>(.*?)</title>", page_html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return "douyin_video"
    title = html.unescape(match.group(1)).strip()
    title = re.sub(r"\s*-\s*抖音\s*$", "", title)
    return title or "douyin_video"


def _resolve_play_url_from_html_regex(page_html: str) -> tuple[str, str]:
    patterns = (
        r'"playAddr"\s*:\s*\[\s*\{\s*"src"\s*:\s*"([^"]+)"',
        r'"src"\s*:\s*"(https?:\\/\\/[^"]+video[^"]+)"',
        r'"src"\s*:\s*"(//[^"]+video[^"]+)"',
    )
    for pattern in patterns:
        match = re.search(pattern, page_html)
        if not match:
            continue
        play_url = _decode_play_url(match.group(1))
        if play_url:
            return play_url, _extract_title_from_html(page_html)
    raise TranscriptError("未从页面中解析到视频播放地址")


def _get_douyin_cookies() -> str:
    """Generate fresh Douyin cookies (ttwid + s_v_web_id) needed for API access."""
    import uuid
    requests = _import_required("requests", "缺少依赖 requests")

    ttwid = ""
    try:
        resp = requests.post(
            "https://ttwid.bytedance.com/ttwid/union/register/",
            json={
                "region": "cn", "aid": 6383, "needFid": False,
                "service": "www.douyin.com",
                "migrate_info": {"ticket": "", "source": "node"},
                "cbUrlProtocol": "https", "union": True,
            },
            timeout=10,
        )
        for c in resp.cookies:
            if c.name == "ttwid":
                ttwid = c.value
                break
    except Exception:
        pass

    svid = f"verify_{uuid.uuid4().hex[:32]}"
    mstoken = uuid.uuid4().hex + uuid.uuid4().hex[:16]
    parts = [f"s_v_web_id={svid}", f"msToken={mstoken}"]
    if ttwid:
        parts.insert(0, f"ttwid={ttwid}")
    return "; ".join(parts)


def _resolve_play_url_f2_api(video_id: str, cookie: str | None = None) -> tuple[str, str]:
    """Use f2 library to call Douyin API with proper ABogus signature."""
    import asyncio
    try:
        from f2.apps.douyin.crawler import DouyinCrawler
        from f2.apps.douyin.model import PostDetail
    except ImportError:
        raise TranscriptError("f2 库未安装")

    cookie_str = cookie or _get_douyin_cookies()

    kwargs = {
        "cookie": cookie_str,
        "headers": {
            "User-Agent": DEFAULT_USER_AGENT,
            "Referer": "https://www.douyin.com/",
        },
        "proxies": {"http://": None, "https://": None},
    }

    async def _fetch():
        async with DouyinCrawler(kwargs) as crawler:
            params = PostDetail(aweme_id=video_id)
            return await crawler.fetch_post_detail(params)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = pool.submit(lambda: asyncio.run(_fetch())).result(timeout=30)
    else:
        result = asyncio.run(_fetch())

    if not result or not isinstance(result, dict):
        raise TranscriptError("抖音 API 返回为空")

    detail = result.get("aweme_detail")
    if not detail:
        filter_info = result.get("filter_detail", {})
        reason = filter_info.get("filter_reason", "unknown")
        raise TranscriptError(f"视频不可用 (reason: {reason})")

    title = detail.get("desc", "")[:80] or "douyin_video"

    video = detail.get("video", {})
    play_addr = video.get("play_addr", {})
    url_list = play_addr.get("url_list", [])
    if not url_list:
        # Try bit_rate fallback
        bit_rate = video.get("bit_rate", [])
        if bit_rate:
            play_addr = bit_rate[0].get("play_addr", {})
            url_list = play_addr.get("url_list", [])

    if not url_list:
        raise TranscriptError("API 返回中未找到播放地址")

    play_url = url_list[0]
    if not play_url.startswith("https://"):
        play_url = "https://" + play_url.lstrip("/")

    return play_url, title


def _resolve_play_url_ytdlp(page_url: str, cookie: str | None = None) -> tuple[str, str]:
    """Use yt-dlp to extract video play URL (more robust against anti-bot)."""
    yt_dlp = _import_optional("yt_dlp")
    if yt_dlp is None:
        raise TranscriptError("yt-dlp 未安装")

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "format": "best",
    }
    if cookie:
        ydl_opts["http_headers"] = {"Cookie": cookie}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(page_url, download=False)

    if not info:
        raise TranscriptError("yt-dlp 未能解析视频信息")

    play_url = info.get("url")
    if not play_url:
        # Try formats list
        formats = info.get("formats") or []
        if formats:
            play_url = formats[-1].get("url")

    if not play_url:
        raise TranscriptError("yt-dlp 未能获取视频播放地址")

    title = info.get("title") or info.get("description", "")[:80] or "douyin_video"
    return play_url, title


def resolve_play_url(
    page_url: str,
    *,
    session,
    cookie: str | None = None,
) -> tuple[str, str]:
    # Extract video ID from page URL
    modal_id = _extract_modal_id_from_url(page_url)

    # Method 1: f2 API with ABogus signature (most reliable)
    if modal_id:
        try:
            return _resolve_play_url_f2_api(modal_id, cookie=_resolve_cookie(cookie))
        except Exception:
            pass

    # Method 2: yt-dlp
    try:
        return _resolve_play_url_ytdlp(page_url, cookie=_resolve_cookie(cookie))
    except Exception:
        pass

    # Method 3: direct HTML scraping (original fallback)
    response = session.get(
        page_url,
        headers=_build_headers(cookie=_resolve_cookie(cookie), accept_html=True),
        timeout=30,
    )
    response.raise_for_status()

    try:
        render_data = _extract_render_data(response.text)
        return _resolve_play_url_from_render_data(render_data)
    except Exception:
        return _resolve_play_url_from_html_regex(response.text)


def sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename).strip()
    sanitized = sanitized[:80]
    return sanitized or "douyin_video"


def download_video_ytdlp(
    page_url: str,
    download_dir: str | Path = DEFAULT_DOWNLOAD_DIR,
    *,
    cookie: str | None = None,
) -> tuple[Path, str]:
    """Download video using f2 API + requests. Returns (video_path, title)."""
    modal_id = _extract_modal_id_from_url(page_url)
    if not modal_id:
        raise TranscriptError(f"无法从 URL 提取视频 ID: {page_url}")

    # Get play URL via f2 API
    play_url, title = _resolve_play_url_f2_api(modal_id, cookie=_resolve_cookie(cookie))

    # Download using requests
    output_dir = Path(download_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    session = _create_session(cookie)
    try:
        video_path = download_video(
            play_url, sanitize_filename(title),
            download_dir=output_dir, session=session, cookie=cookie,
        )
    finally:
        session.close()

    return video_path, title


def download_video(
    play_url: str,
    title: str,
    download_dir: str | Path = DEFAULT_DOWNLOAD_DIR,
    *,
    session,
    cookie: str | None = None,
) -> Path:
    tqdm_module = _import_optional("tqdm")

    output_dir = Path(download_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sanitize_filename(title)}.mp4"

    response = session.get(
        play_url,
        headers=_build_headers(
            cookie=_resolve_cookie(cookie),
            referer="https://www.douyin.com/",
        ),
        stream=True,
        timeout=60,
    )
    response.raise_for_status()

    total_size = int(response.headers.get("Content-Length", 0))
    progress = None
    if tqdm_module is not None:
        progress = tqdm_module.tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc="Downloading",
        )

    try:
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if not chunk:
                    continue
                handle.write(chunk)
                if progress is not None:
                    progress.update(len(chunk))
    finally:
        if progress is not None:
            progress.close()

    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise TranscriptError(f"视频下载失败: {output_path}")

    return output_path


def _select_whisper_source(model_dir: Path) -> tuple[str, str, str, bool]:
    torch = _import_optional("torch")
    medium_path = model_dir / "whisper-medium"
    large_path = model_dir / "whisper-large-v3"

    def resolve(local_path: Path, fallback_name: str, device: str, compute_type: str):
        if local_path.exists():
            return str(local_path), device, compute_type, True
        return fallback_name, device, compute_type, False

    if torch is None or not torch.cuda.is_available():
        return resolve(medium_path, "medium", "cpu", "int8")

    try:
        device = torch.device("cuda")
        total_vram = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        if total_vram < 9.0:
            return resolve(medium_path, "medium", "cuda", "float16")
        return resolve(large_path, "large-v3", "cuda", "float16")
    except Exception:
        return resolve(medium_path, "medium", "cpu", "int8")


def _build_whisper_model(model_dir: Path):
    faster_whisper = _import_required(
        "faster_whisper",
        "缺少依赖 faster-whisper，请先安装 requirements.txt",
    )

    source, device, compute_type, local_files_only = _select_whisper_source(model_dir)
    try:
        return faster_whisper.WhisperModel(
            source,
            device=device,
            compute_type=compute_type,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        if source == "medium" and device == "cpu":
            raise TranscriptError(f"Whisper 模型加载失败: {exc}") from exc

        try:
            return faster_whisper.WhisperModel(
                "medium",
                device="cpu",
                compute_type="int8",
                local_files_only=False,
            )
        except Exception as fallback_exc:
            raise TranscriptError(
                f"Whisper 模型加载失败: {exc}; 回退到 CPU 也失败: {fallback_exc}"
            ) from fallback_exc


def _load_video_file_clip():
    moviepy_module = _import_optional("moviepy")
    if moviepy_module is not None:
        video_file_clip = getattr(moviepy_module, "VideoFileClip", None)
        if video_file_clip is not None:
            return video_file_clip

    moviepy_editor = _import_optional("moviepy.editor")
    if moviepy_editor is not None:
        video_file_clip = getattr(moviepy_editor, "VideoFileClip", None)
        if video_file_clip is not None:
            return video_file_clip

    raise TranscriptError("缺少依赖 moviepy，请先安装 requirements.txt")


def transcribe_video(
    video_path: str | Path,
    *,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    save_text_file: bool = True,
) -> str:
    opencc_module = _import_optional("opencc")
    torch = _import_optional("torch")
    video_file_clip = _load_video_file_clip()

    resolved_video_path = Path(video_path).expanduser().resolve()
    if not resolved_video_path.exists():
        raise TranscriptError(f"视频文件不存在: {resolved_video_path}")
    if resolved_video_path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        raise TranscriptError(
            f"不支持的视频类型: {resolved_video_path.suffix}，仅支持 {sorted(SUPPORTED_VIDEO_SUFFIXES)}"
        )

    model_root = Path(model_dir).expanduser().resolve()
    whisper_model = _build_whisper_model(model_root)
    video_clip = None
    audio_clip = None
    audio_path = resolved_video_path.with_name(f"temp_{resolved_video_path.stem}.wav")

    try:
        video_clip = video_file_clip(str(resolved_video_path))
        audio_clip = video_clip.audio
        if audio_clip is None:
            raise TranscriptError("视频不包含可转写音轨")
        audio_clip.write_audiofile(str(audio_path), logger=None)

        segments, _ = whisper_model.transcribe(
            str(audio_path),
            beam_size=5,
            language="zh",
        )

        converter = opencc_module.OpenCC("t2s") if opencc_module is not None else None
        transcript_lines = []
        for segment in segments:
            segment_text = (segment.text or "").strip()
            if converter is not None:
                segment_text = converter.convert(segment_text)
            if segment_text:
                transcript_lines.append(segment_text)

        transcript = "\n\n".join(transcript_lines).strip()
        if not transcript:
            raise TranscriptError("语音识别完成，但未提取到有效文本")

        if save_text_file:
            resolved_video_path.with_suffix(".txt").write_text(transcript, encoding="utf-8")

        return transcript
    finally:
        if audio_clip is not None:
            audio_clip.close()
        if video_clip is not None:
            video_clip.close()
        if audio_path.exists():
            audio_path.unlink()

        del whisper_model
        if torch is not None and torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
            with contextlib.suppress(Exception):
                torch.cuda.synchronize()


def cleanup_artifacts(video_path: str | Path) -> None:
    resolved_video_path = Path(video_path).expanduser().resolve()
    artifacts = (
        resolved_video_path,
        resolved_video_path.with_suffix(".txt"),
        resolved_video_path.with_name(f"temp_{resolved_video_path.stem}.wav"),
    )
    for artifact in artifacts:
        if artifact.exists():
            artifact.unlink()


def transcribe_douyin_link(
    link_text: str,
    *,
    download_dir: str | Path = DEFAULT_DOWNLOAD_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    keep_artifacts: bool = False,
    save_text_file: bool = False,
    cookie: str | None = None,
) -> str:
    session = _create_session(cookie=cookie)
    try:
        modal_id, page_url = resolve_douyin_page_url(
            link_text,
            session=session,
            cookie=cookie,
        )
        play_url, title = resolve_play_url(
            page_url,
            session=session,
            cookie=cookie,
        )
        video_path = download_video(
            play_url,
            f"{title}_{modal_id}",
            download_dir=download_dir,
            session=session,
            cookie=cookie,
        )

        try:
            return transcribe_video(
                video_path,
                model_dir=model_dir,
                save_text_file=save_text_file or keep_artifacts,
            )
        finally:
            if not keep_artifacts:
                cleanup_artifacts(video_path)
    finally:
        session.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="输入抖音视频链接，输出视频文案")
    parser.add_argument("link", help="抖音短链/直链，支持直接粘贴整段分享文案")
    parser.add_argument(
        "--text-output",
        help="将提取到的文案写入指定文件",
    )
    parser.add_argument(
        "--download-dir",
        default=str(DEFAULT_DOWNLOAD_DIR),
        help=f"视频下载目录，默认 {DEFAULT_DOWNLOAD_DIR}",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help=(
            "Whisper 模型目录。若目录下不存在 whisper-medium / whisper-large-v3，"
            "则会回退为 faster-whisper 默认模型名下载"
        ),
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="保留下载的视频和生成的 txt 文件",
    )
    parser.add_argument(
        "--cookie",
        help=f"可选。抖音请求 Cookie；未提供时会读取环境变量 {DEFAULT_COOKIE_ENV}",
    )
    args = parser.parse_args(argv)

    try:
        transcript = transcribe_douyin_link(
            args.link,
            download_dir=args.download_dir,
            model_dir=args.model_dir,
            keep_artifacts=args.keep_artifacts,
            save_text_file=bool(args.text_output),
            cookie=args.cookie,
        )
    except TranscriptError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    if args.text_output:
        output_path = Path(args.text_output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(transcript, encoding="utf-8")
    else:
        print(transcript)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
