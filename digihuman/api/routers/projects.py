from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from ..errors import APIError, ErrorCode, success_response
from ..project_store import get_project_store

router = APIRouter()


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
    project = store.update_step(project_id, step, req.data)
    if project is None:
        raise APIError(ErrorCode.NOT_FOUND, f"Project not found: {project_id}", 404)
    return success_response(project)


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    store = get_project_store()
    if not store.delete(project_id):
        raise APIError(ErrorCode.NOT_FOUND, f"Project not found: {project_id}", 404)
    return success_response({"deleted": project_id})
