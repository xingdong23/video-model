from __future__ import annotations

from typing import Any, Optional

from fastapi import Request
from fastapi.responses import JSONResponse


class ErrorCode:
    VALIDATION_ERROR = "VALIDATION_ERROR"
    ENGINE_ERROR = "ENGINE_ERROR"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_UPLOAD_ERROR = "FILE_UPLOAD_ERROR"
    ENGINE_BUSY = "ENGINE_BUSY"
    AUTH_ERROR = "AUTH_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


def success_response(data: Any = None) -> dict:
    return {"success": True, "data": data, "error": None}


def error_response(code: str, message: str, status_code: int = 500) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "data": None,
            "error": {"code": code, "message": message},
        },
    )


class APIError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 500):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    return error_response(exc.code, exc.message, exc.status_code)


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return error_response(
        ErrorCode.INTERNAL_ERROR,
        str(exc) or "Internal server error",
        500,
    )
