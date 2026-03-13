from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("api.task_manager")


@dataclass
class Task:
    task_id: str
    task_type: str
    status: str = "queued"  # queued | processing | completed | failed | cancelled
    step: str = ""
    progress: int = 0
    message: str = ""
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    request_id: Optional[str] = None
    _cancelled: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "step": self.step,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "request_id": self.request_id,
        }


# Sentinel value to signal worker threads to shut down
_SHUTDOWN = object()


class TaskManager:
    """In-memory task manager with GPU (serial) and CPU (concurrent) worker queues."""

    def __init__(self, cpu_workers: int = 2):
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()
        self._gpu_queue: queue.Queue = queue.Queue()
        self._cpu_queue: queue.Queue = queue.Queue()
        self._gpu_worker: Optional[threading.Thread] = None
        self._cpu_workers: list[threading.Thread] = []
        self._cpu_worker_count = cpu_workers
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._gpu_worker = threading.Thread(
            target=self._worker_loop, args=(self._gpu_queue, "gpu"), daemon=True, name="task-gpu-worker"
        )
        self._gpu_worker.start()

        for i in range(self._cpu_worker_count):
            t = threading.Thread(
                target=self._worker_loop, args=(self._cpu_queue, f"cpu-{i}"), daemon=True, name=f"task-cpu-worker-{i}"
            )
            t.start()
            self._cpu_workers.append(t)

        self._started = True
        logger.info("TaskManager started: 1 GPU worker, %d CPU workers", self._cpu_worker_count)

    def shutdown(self) -> None:
        self._gpu_queue.put(_SHUTDOWN)
        for _ in self._cpu_workers:
            self._cpu_queue.put(_SHUTDOWN)

    def submit(
        self,
        task_type: str,
        params: dict[str, Any],
        executor_fn: Callable[[Task, dict[str, Any]], dict],
        request_id: Optional[str] = None,
        gpu: bool = True,
    ) -> str:
        task_id = uuid.uuid4().hex[:16]
        task = Task(
            task_id=task_id,
            task_type=task_type,
            request_id=request_id,
        )
        with self._lock:
            self._tasks[task_id] = task

        target_queue = self._gpu_queue if gpu else self._cpu_queue
        target_queue.put((task, params, executor_fn))
        logger.info("Task %s (%s) submitted to %s queue", task_id, task_type, "GPU" if gpu else "CPU")
        return task_id

    def get(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self._tasks.get(task_id)

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            return False
        if task.status != "queued":
            return False
        task._cancelled.set()
        task.status = "cancelled"
        task.updated_at = time.time()
        return True

    def _worker_loop(self, q: queue.Queue, worker_name: str) -> None:
        logger.info("Worker %s started", worker_name)
        while True:
            item = q.get()
            if item is _SHUTDOWN:
                logger.info("Worker %s shutting down", worker_name)
                break

            task, params, executor_fn = item

            if task._cancelled.is_set():
                logger.info("Worker %s: task %s was cancelled, skipping", worker_name, task.task_id)
                continue

            task.status = "processing"
            task.updated_at = time.time()
            logger.info("Worker %s: processing task %s (%s)", worker_name, task.task_id, task.task_type)

            try:
                result = executor_fn(task, params)
                if task._cancelled.is_set():
                    task.status = "cancelled"
                else:
                    task.status = "completed"
                    task.progress = 100
                    task.result = result
            except Exception as e:
                logger.exception("Worker %s: task %s failed", worker_name, task.task_id)
                task.status = "failed"
                task.error = str(e)
            finally:
                task.updated_at = time.time()


def _make_progress_callback(task: Task):
    """Create a progress callback function bound to a specific task."""
    def callback(step: str, progress: int, message: str) -> None:
        task.step = step
        task.progress = progress
        task.message = message
        task.updated_at = time.time()
    return callback


_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager
