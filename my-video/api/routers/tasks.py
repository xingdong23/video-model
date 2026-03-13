from __future__ import annotations

from fastapi import APIRouter

from ..errors import APIError, ErrorCode, success_response
from ..task_manager import get_task_manager

router = APIRouter()


@router.get("/{task_id}")
async def get_task(task_id: str):
    task = get_task_manager().get(task_id)
    if task is None:
        raise APIError(ErrorCode.FILE_NOT_FOUND, f"Task not found: {task_id}", 404)
    return success_response(task.to_dict())


@router.delete("/{task_id}")
async def cancel_task(task_id: str):
    task = get_task_manager().get(task_id)
    if task is None:
        raise APIError(ErrorCode.FILE_NOT_FOUND, f"Task not found: {task_id}", 404)
    if not get_task_manager().cancel(task_id):
        raise APIError(
            ErrorCode.VALIDATION_ERROR,
            f"Task {task_id} cannot be cancelled (status: {task.status})",
            409,
        )
    return success_response({"task_id": task_id, "status": "cancelled"})
