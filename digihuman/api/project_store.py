from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path

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
            data = self._read(project_id)
            if data is None:
                return None
            # Migrate legacy dict steps to list format on read
            migrated = False
            for step_name, step_val in data.get("steps", {}).items():
                if isinstance(step_val, dict):
                    data["steps"][step_name] = self._normalize_step_value(step_val)
                    migrated = True
            if migrated:
                self._write(data)
            return data

    VALID_STEPS = {"douyin", "script", "voice", "digital_human", "subtitle", "bgm"}

    @staticmethod
    def _normalize_step_value(val) -> list:
        """Convert legacy dict format to list format for backward compatibility."""
        if isinstance(val, list):
            return val
        if isinstance(val, dict):
            # Wrap old dict format with default meta fields
            record = {
                "record_id": val.get("record_id", uuid.uuid4().hex[:8]),
                "created_at": val.get("created_at", 0),
                "parent_record_id": val.get("parent_record_id"),
                **{k: v for k, v in val.items() if k not in ("record_id", "created_at", "parent_record_id")},
            }
            return [record]
        return []

    _META_KEYS = frozenset({"record_id", "created_at", "parent_record_id"})

    def update_step(
        self, project_id: str, step: str, step_data: dict,
        parent_record_id: str | None = None,
    ) -> dict | None:
        if step not in self.VALID_STEPS:
            raise ValueError(f"Invalid step name: {step!r}, must be one of {self.VALID_STEPS}")
        with self._lock:
            data = self._read(project_id)
            if data is None:
                return None
            clean_data = {k: v for k, v in step_data.items() if k not in self._META_KEYS}
            record = {
                "record_id": uuid.uuid4().hex[:8],
                "created_at": time.time(),
                "parent_record_id": parent_record_id,
                **clean_data,
            }
            # Ensure steps[step] is a list, then append
            existing = data["steps"].get(step)
            records = self._normalize_step_value(existing) if existing else []
            records.append(record)
            data["steps"][step] = records
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
                    steps = data.get("steps", {})
                    # Count steps that have at least one record
                    steps_done = sum(
                        1 for v in steps.values()
                        if (isinstance(v, list) and v) or (isinstance(v, dict) and v)
                    )
                    projects.append({
                        "project_id": data["project_id"],
                        "title": data.get("title", ""),
                        "created_at": data.get("created_at", 0),
                        "updated_at": data.get("updated_at", 0),
                        "steps_done": steps_done,
                    })
                except Exception:
                    logger.debug("Failed to read project file: %s", p, exc_info=True)
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
                    douyin = data.get("steps", {}).get("douyin")
                    if douyin is None:
                        continue
                    # Handle both list (new) and dict (legacy) formats
                    records = self._normalize_step_value(douyin)
                    if any(r.get("video_file_id") == video_file_id for r in records):
                        results.append(data)
                except Exception:
                    logger.debug("Failed to read project file: %s", p, exc_info=True)
                    continue
        results.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
        return results

    # Keys in step data that hold file_id references
    _FILE_ID_KEYS = ("video_file_id", "file_id", "srt_file_id", "burned_file_id", "upload_video_file_id")

    @classmethod
    def _extract_file_ids_from_record(cls, record: dict) -> set[str]:
        """Extract file_id values from a single step record."""
        ids: set[str] = set()
        for key in cls._FILE_ID_KEYS:
            fid = record.get(key)
            if fid:
                ids.add(fid)
        for f in record.get("files", []):
            if isinstance(f, dict) and f.get("file_id"):
                ids.add(f["file_id"])
        return ids

    def collect_all_referenced_file_ids(self) -> set[str]:
        """Return every file_id referenced by any project step."""
        file_ids: set[str] = set()
        with self._lock:
            for p in self.root.glob("*.json"):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    for step_val in data.get("steps", {}).values():
                        records = self._normalize_step_value(step_val)
                        for record in records:
                            file_ids.update(self._extract_file_ids_from_record(record))
                except Exception:
                    logger.debug("Failed to read project file: %s", p, exc_info=True)
                    continue
        return file_ids

    def collect_file_ids(self, project_id: str) -> set[str]:
        """Return all file_ids referenced by a single project."""
        file_ids: set[str] = set()
        with self._lock:
            data = self._read(project_id)
            if data is None:
                return file_ids
            for step_val in data.get("steps", {}).values():
                records = self._normalize_step_value(step_val)
                for record in records:
                    file_ids.update(self._extract_file_ids_from_record(record))
        return file_ids

    def delete(self, project_id: str) -> bool:
        with self._lock:
            p = self._path(project_id)
            if p.exists():
                p.unlink()
                return True
        return False


_store: ProjectStore | None = None
_store_lock = threading.Lock()


def get_project_store() -> ProjectStore:
    global _store
    if _store is not None:
        return _store
    with _store_lock:
        if _store is None:
            from .config import get_settings
            settings = get_settings()
            _store = ProjectStore(Path(settings.storage_root) / "projects")
    return _store
