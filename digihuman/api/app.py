from __future__ import annotations

import logging
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .errors import APIError, ErrorCode, api_error_handler, error_response, generic_error_handler

logger = logging.getLogger("api")


def _start_cleanup_timer(interval_seconds: int, ttl_seconds: int):
    """Background thread that periodically cleans up expired storage files."""
    def _run():
        while True:
            time.sleep(interval_seconds)
            try:
                from .storage import get_storage
                from .project_store import get_project_store
                protected = get_project_store().collect_all_referenced_file_ids()
                removed = get_storage().cleanup_expired(ttl_seconds, protected_ids=protected)
                if removed:
                    logger.info("Cleanup: removed %d expired files (%d protected)", removed, len(protected))
            except Exception:
                logger.warning("Cleanup job failed", exc_info=True)

    t = threading.Thread(target=_run, daemon=True, name="storage-cleanup")
    t.start()
    return t


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # ── sys.path setup (once) ──
    root = str(settings.digihuman_root)
    if root not in sys.path:
        sys.path.insert(0, root)

    # ── Pre-load engines ──
    from .dependencies import get_engine_manager
    em = get_engine_manager()

    logger.info("Pre-loading engines ...")
    for init_fn, name in [
        (em.init_voice, "VoiceEngine"),
        (em.init_digital_human, "DigitalHumanEngine"),
        (em.init_subtitle, "SubtitleEngine"),
        (em.init_bgm, "BgmEngine"),
        (em.init_rewrite, "RewriteEngine"),
    ]:
        try:
            init_fn()
        except Exception:
            logger.warning("Failed to pre-load %s", name, exc_info=True)
    logger.info("Engine pre-loading complete")

    # ── Start TaskManager workers ──
    from .task_manager import get_task_manager
    get_task_manager().start()

    # ── Start storage cleanup job ──
    _start_cleanup_timer(
        interval_seconds=3600,
        ttl_seconds=settings.file_ttl_seconds,
    )
    logger.info("Storage cleanup job started (TTL=%ds)", settings.file_ttl_seconds)

    yield

    # ── Shutdown ──
    logger.info("Shutting down")
    from .task_manager import get_task_manager
    get_task_manager().shutdown()


def create_app():
    settings = get_settings()

    app = FastAPI(
        title="DigiHuman API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── Auth middleware ──
    api_key = settings.api_key
    if api_key:
        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            if request.url.path in ("/health", "/health/gpu", "/docs", "/openapi.json", "/redoc", "/") or request.url.path.startswith("/static"):
                return await call_next(request)
            provided = request.headers.get("X-API-Key") or request.query_params.get("api_key")
            if provided != api_key:
                return error_response(ErrorCode.AUTH_ERROR, "Invalid or missing API key", 401)
            return await call_next(request)
        logger.info("API key authentication enabled")

    # ── CORS ──
    origins = [o.strip() for o in settings.cors_origins.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request-ID + logging middleware ──
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        start = time.time()
        response = await call_next(request)
        elapsed = time.time() - start
        response.headers["X-Request-ID"] = request_id
        if elapsed > 1.0:
            logger.info("[%s] %s %s -> %d (%.1fs)", request_id, request.method, request.url.path, response.status_code, elapsed)
        return response

    # ── Error handlers ──
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(Exception, generic_error_handler)

    # ── Health ──
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/health/gpu")
    async def health_gpu():
        info = {"cuda_available": False, "device_count": 0, "devices": []}
        try:
            import torch
            info["cuda_available"] = torch.cuda.is_available()
            info["device_count"] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_mem / 1024**3, 2),
                })
        except Exception:
            pass
        return info

    # ── Routers ──
    from .routers import tts, avatar, subtitle, audio_mixer, copywriter, scraper, pipeline, files, tasks, projects
    app.include_router(tts.router, prefix="/api/v1/voice", tags=["voice"])
    app.include_router(avatar.router, prefix="/api/v1/digital-human", tags=["digital-human"])
    app.include_router(subtitle.router, prefix="/api/v1/subtitle", tags=["subtitle"])
    app.include_router(audio_mixer.router, prefix="/api/v1/bgm", tags=["bgm"])
    app.include_router(copywriter.router, prefix="/api/v1/rewrite", tags=["rewrite"])
    app.include_router(scraper.router, prefix="/api/v1/douyin", tags=["douyin"])
    app.include_router(pipeline.router, prefix="/api/v1/pipeline", tags=["pipeline"])
    app.include_router(files.router, prefix="/api/v1/files", tags=["files"])
    app.include_router(tasks.router, prefix="/api/v1/tasks", tags=["tasks"])
    app.include_router(projects.router, prefix="/api/v1/projects", tags=["projects"])

    # ── Static frontend ──
    from pathlib import Path
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        @app.get("/")
        async def index():
            return FileResponse(static_dir / "index.html")

        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app
