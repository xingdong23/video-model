from __future__ import annotations

import logging
import sys
import threading
import time

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
                removed = get_storage().cleanup_expired(ttl_seconds)
                if removed:
                    logger.info("Cleanup: removed %d expired files", removed)
            except Exception:
                logger.warning("Cleanup job failed", exc_info=True)

    t = threading.Thread(target=_run, daemon=True, name="storage-cleanup")
    t.start()
    return t


def create_app():
    settings = get_settings()

    app = FastAPI(
        title="my-video API",
        version="1.0.0",
    )

    # ── Auth middleware ──
    api_key = settings.api_key
    if api_key:
        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            # Health endpoints are public
            if request.url.path in ("/health", "/health/gpu", "/docs", "/openapi.json", "/redoc", "/") or request.url.path.startswith("/static"):
                return await call_next(request)
            # Check API key
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

    # ── Request logging middleware ──
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = time.time() - start
        if elapsed > 1.0:
            logger.info("%s %s -> %d (%.1fs)", request.method, request.url.path, response.status_code, elapsed)
        return response

    # ── Error handlers ──
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(Exception, generic_error_handler)

    # ── Startup ──
    @app.on_event("startup")
    async def startup():
        root = str(settings.my_video_root)
        if root not in sys.path:
            sys.path.insert(0, root)

        from .dependencies import get_engine_manager
        em = get_engine_manager()

        logger.info("Pre-loading engines ...")
        try:
            em.init_voice()
        except Exception:
            logger.warning("Failed to pre-load VoiceEngine", exc_info=True)
        try:
            em.init_digital_human()
        except Exception:
            logger.warning("Failed to pre-load DigitalHumanEngine", exc_info=True)
        try:
            em.init_subtitle()
        except Exception:
            logger.warning("Failed to pre-load SubtitleEngine", exc_info=True)
        try:
            em.init_bgm()
        except Exception:
            logger.warning("Failed to pre-load BgmEngine", exc_info=True)
        try:
            em.init_rewrite()
        except Exception:
            logger.warning("Failed to pre-load RewriteEngine", exc_info=True)
        logger.info("Engine pre-loading complete")

        # Start storage cleanup job
        _start_cleanup_timer(
            interval_seconds=3600,  # every hour
            ttl_seconds=settings.file_ttl_seconds,
        )
        logger.info("Storage cleanup job started (TTL=%ds)", settings.file_ttl_seconds)

    @app.on_event("shutdown")
    async def shutdown():
        logger.info("Shutting down")

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
    from .routers import voice, digital_human, subtitle, bgm, rewrite, douyin, workflow, files
    app.include_router(voice.router, prefix="/api/v1/voice", tags=["voice"])
    app.include_router(digital_human.router, prefix="/api/v1/digital-human", tags=["digital-human"])
    app.include_router(subtitle.router, prefix="/api/v1/subtitle", tags=["subtitle"])
    app.include_router(bgm.router, prefix="/api/v1/bgm", tags=["bgm"])
    app.include_router(rewrite.router, prefix="/api/v1/rewrite", tags=["rewrite"])
    app.include_router(douyin.router, prefix="/api/v1/douyin", tags=["douyin"])
    app.include_router(workflow.router, prefix="/api/v1/workflow", tags=["workflow"])
    app.include_router(files.router, prefix="/api/v1/files", tags=["files"])

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
