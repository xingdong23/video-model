from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel

from ..errors import APIError, ErrorCode, success_response
from ..project_store import get_project_store
from ..storage import get_storage

router = APIRouter()
logger = logging.getLogger(__name__)


class CreateProjectRequest(BaseModel):
    title: str = ""


class UpdateStepRequest(BaseModel):
    data: dict


# ── Endpoints ──

@router.post("")
async def create_project(req: CreateProjectRequest):
    store = get_project_store()
    project = store.create(title=req.title)
    return success_response(project)


@router.get("")
async def list_projects():
    store = get_project_store()
    return success_response(store.list_all())


@router.get("/{project_id}")
async def get_project(project_id: str):
    store = get_project_store()
    project = store.get(project_id)
    if project is None:
        raise APIError(ErrorCode.NOT_FOUND, f"Project not found: {project_id}", 404)
    return success_response(project)


@router.put("/{project_id}/steps/{step}")
async def update_step(project_id: str, step: str, req: UpdateStepRequest):
    store = get_project_store()
    try:
        project = store.update_step(project_id, step, req.data)
    except ValueError as e:
        raise APIError(ErrorCode.VALIDATION_ERROR, str(e), 400)
    if project is None:
        raise APIError(ErrorCode.NOT_FOUND, f"Project not found: {project_id}", 404)
    return success_response(project)


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    store = get_project_store()
    # Collect file IDs before deleting the project JSON
    project_file_ids = store.collect_file_ids(project_id)

    if not store.delete(project_id):
        raise APIError(ErrorCode.NOT_FOUND, f"Project not found: {project_id}", 404)

    # Remove files that are no longer referenced by any remaining project
    if project_file_ids:
        still_referenced = store.collect_all_referenced_file_ids()
        orphan_ids = project_file_ids - still_referenced
        storage = get_storage()
        for fid in orphan_ids:
            try:
                storage.delete(fid)
            except FileNotFoundError:
                pass
            except Exception:
                logger.warning("Failed to delete orphan file: %s", fid, exc_info=True)
        if orphan_ids:
            logger.info("Deleted %d orphan files for project %s", len(orphan_ids), project_id)

    return success_response({"deleted": project_id})
