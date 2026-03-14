from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

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


@router.get("/{task_id}/stream")
async def stream_task(task_id: str):
    """SSE endpoint: streams task progress until terminal state.

    Usage:
        const es = new EventSource("/api/v1/tasks/{task_id}/stream");
        es.addEventListener("progress", (e) => { ... });
        es.addEventListener("completed", (e) => { es.close(); });
        es.addEventListener("failed", (e) => { es.close(); });
    """
    task = get_task_manager().get(task_id)
    if task is None:
        raise APIError(ErrorCode.FILE_NOT_FOUND, f"Task not found: {task_id}", 404)

    async def event_generator():
        import json
        last_progress = -1
        last_status = ""

        while True:
            t = get_task_manager().get(task_id)
            if t is None:
                yield _sse_event("error", {"message": "Task not found"})
                return

            # Only emit when something changed
            if t.progress != last_progress or t.status != last_status:
                last_progress = t.progress
                last_status = t.status
                data = t.to_dict()

                if t.status in ("completed", "failed", "cancelled"):
                    yield _sse_event(t.status, data)
                    return
                else:
                    yield _sse_event("progress", data)

            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


def _sse_event(event_type: str, data: dict) -> str:
    import json
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n"
