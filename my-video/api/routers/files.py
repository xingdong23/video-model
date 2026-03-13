from __future__ import annotations

import mimetypes
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse

from ..errors import APIError, ErrorCode, success_response
from ..storage import get_storage

router = APIRouter()


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    category: str = Form("upload"),
):
    suffix = Path(file.filename or "file").suffix or ""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        storage = get_storage()
        file_id = storage.save(tmp_path, category, file.filename or "file")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return success_response({
        "file_id": file_id,
        "filename": file.filename,
        "download_url": storage.get_url(file_id),
    })


@router.get("/{file_id}")
async def download_file(file_id: str):
    try:
        storage = get_storage()
        path = storage.get_path(file_id)
        info = storage.get_info(file_id)
    except FileNotFoundError:
        raise APIError(ErrorCode.FILE_NOT_FOUND, f"File not found: {file_id}", 404)
    media_type = mimetypes.guess_type(info.get("original_name", ""))[0] or "application/octet-stream"
    return FileResponse(
        path=str(path),
        media_type=media_type,
        filename=info.get("original_name", path.name),
    )


@router.get("/{file_id}/info")
async def file_info(file_id: str):
    try:
        storage = get_storage()
        info = storage.get_info(file_id)
    except FileNotFoundError:
        raise APIError(ErrorCode.FILE_NOT_FOUND, f"File not found: {file_id}", 404)
    info["download_url"] = storage.get_url(file_id)
    return success_response(info)


@router.delete("/{file_id}")
async def delete_file(file_id: str):
    try:
        storage = get_storage()
        storage.delete(file_id)
    except FileNotFoundError:
        raise APIError(ErrorCode.FILE_NOT_FOUND, f"File not found: {file_id}", 404)
    return success_response({"deleted": file_id})
