from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ProjectStore:
    """Simple JSON-file-based project storage.

    Each project is stored as ``{project_id}.json`` under the *root* directory.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ── helpers ──

    def _path(self, project_id: str) -> Path:
        return self.root / f"{project_id}.json"

    def _read(self, project_id: str) -> dict | None:
        p = self._path(project_id)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def _write(self, data: dict) -> None:
        p = self._path(data["project_id"])
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── public API ──

    def create(self, title: str = "") -> dict:
        project_id = uuid.uuid4().hex[:12]
        now = time.time()
        data = {
            "project_id": project_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "steps": {},
        }
        with self._lock:
            self._write(data)
        logger.info("Project created: %s", project_id)
        return data

    def get(self, project_id: str) -> dict | None:
        with self._lock:
            return self._read(project_id)

    def update_step(self, project_id: str, step: str, step_data: dict) -> dict | None:
        with self._lock:
            data = self._read(project_id)
            if data is None:
                return None
            data["steps"][step] = step_data
            data["updated_at"] = time.time()
            # Auto-update title from first meaningful content
            if not data["title"] and step == "douyin" and step_data.get("transcript"):
                data["title"] = step_data["transcript"][:60]
            if not data["title"] and step == "script" and step_data.get("text"):
                data["title"] = step_data["text"][:60]
            self._write(data)
        return data

    def update_title(self, project_id: str, title: str) -> dict | None:
        with self._lock:
            data = self._read(project_id)
            if data is None:
                return None
            data["title"] = title
            data["updated_at"] = time.time()
            self._write(data)
        return data

    def list_all(self) -> list[dict]:
        """Return all projects as summary dicts, sorted by updated_at desc."""
        projects = []
        with self._lock:
            for p in self.root.glob("*.json"):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    projects.append({
                        "project_id": data["project_id"],
                        "title": data.get("title", ""),
                        "created_at": data.get("created_at", 0),
                        "updated_at": data.get("updated_at", 0),
                        "steps_done": len(data.get("steps", {})),
                    })
                except Exception:
                    continue
        projects.sort(key=lambda x: x["updated_at"], reverse=True)
        return projects

    def find_by_video(self, video_file_id: str) -> list[dict]:
        """Return projects whose douyin step references the given video file_id."""
        results = []
        with self._lock:
            for p in self.root.glob("*.json"):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    douyin = data.get("steps", {}).get("douyin", {})
                    if douyin.get("video_file_id") == video_file_id:
                        results.append(data)
                except Exception:
                    continue
        results.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
        return results

    def delete(self, project_id: str) -> bool:
        with self._lock:
            p = self._path(project_id)
            if p.exists():
                p.unlink()
                return True
        return False


_store: Optional[ProjectStore] = None


def get_project_store() -> ProjectStore:
    global _store
    if _store is None:
        from .config import get_settings
        settings = get_settings()
        _store = ProjectStore(Path(settings.storage_root) / "projects")
    return _store
