from __future__ import annotations
"""Uvicorn entry-point: python -m api.main"""

import logging
import uvicorn

from .app import create_app
from .config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
)

app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        workers=1,
        timeout_keep_alive=settings.timeout_keep_alive,
        timeout_graceful_shutdown=settings.timeout_graceful_shutdown,
        log_level="info",
    )
