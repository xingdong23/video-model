from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    @abstractmethod
    def save(self, source_path: str | Path, category: str, filename: str) -> str:
        """Copy file into storage, return file_id."""

    @abstractmethod
    def get_path(self, file_id: str) -> Path:
        """Return the local filesystem path for a file_id."""

    @abstractmethod
    def get_url(self, file_id: str) -> str:
        """Return a download URL for a file_id."""

    @abstractmethod
    def get_info(self, file_id: str) -> dict:
        """Return metadata for a file_id."""

    @abstractmethod
    def delete(self, file_id: str) -> None:
        """Delete a file by file_id."""

    @abstractmethod
    def cleanup_expired(self, max_age_seconds: int) -> int:
        """Delete files older than max_age_seconds, return count deleted."""


class LocalStorage(StorageBackend):
    """Local filesystem storage with in-memory index for O(1) lookups."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, Path] = {}
        self._lock = threading.Lock()
        self._rebuild_index()

    def _rebuild_index(self):
        """Scan disk once at startup to build the in-memory file_id -> manifest_path index."""
        count = 0
        for manifest_path in self.root.rglob("*.json"):
            file_id = manifest_path.stem
            self._index[file_id] = manifest_path
            count += 1
        if count:
            logger.info("Storage index rebuilt: %d files", count)

    def save(self, source_path: str | Path, category: str, filename: str) -> str:
        source = Path(source_path)
        file_id = uuid.uuid4().hex
        suffix = source.suffix or Path(filename).suffix
        stored_name = f"{file_id}{suffix}"

        category_dir = self.root / category
        category_dir.mkdir(parents=True, exist_ok=True)

        dest = category_dir / stored_name
        try:
            shutil.move(str(source), str(dest))
        except (OSError, shutil.Error):
            # Cross-device move falls back to copy
            shutil.copy2(str(source), str(dest))

        manifest = {
            "file_id": file_id,
            "original_name": filename,
            "category": category,
            "stored_name": stored_name,
            "created_at": time.time(),
            "size_bytes": dest.stat().st_size,
        }
        manifest_path = category_dir / f"{file_id}.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False))

        with self._lock:
            self._index[file_id] = manifest_path

        return file_id

    def get_path(self, file_id: str) -> Path:
        manifest = self._load_manifest(file_id)
        path = self.root / manifest["category"] / manifest["stored_name"]
        if not path.exists():
            raise FileNotFoundError(f"File data missing: {file_id}")
        return path

    def get_url(self, file_id: str) -> str:
        self._load_manifest(file_id)
        return f"/api/v1/files/{file_id}"

    def get_info(self, file_id: str) -> dict:
        return self._load_manifest(file_id)

    def delete(self, file_id: str) -> None:
        manifest = self._load_manifest(file_id)
        category_dir = self.root / manifest["category"]
        data_path = category_dir / manifest["stored_name"]
        manifest_path = category_dir / f"{file_id}.json"
        data_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        with self._lock:
            self._index.pop(file_id, None)

    def cleanup_expired(self, max_age_seconds: int) -> int:
        now = time.time()
        count = 0
        with self._lock:
            expired_ids = []
            for file_id, manifest_path in self._index.items():
                try:
                    manifest = json.loads(manifest_path.read_text())
                    if now - manifest.get("created_at", now) > max_age_seconds:
                        data_path = manifest_path.parent / manifest["stored_name"]
                        data_path.unlink(missing_ok=True)
                        manifest_path.unlink(missing_ok=True)
                        expired_ids.append(file_id)
                        count += 1
                except Exception:
                    continue
            for fid in expired_ids:
                self._index.pop(fid, None)
        if count:
            logger.info("Storage cleanup: removed %d expired files", count)
        return count

    def _load_manifest(self, file_id: str) -> dict:
        with self._lock:
            manifest_path = self._index.get(file_id)
        if manifest_path is None or not manifest_path.exists():
            raise FileNotFoundError(f"File not found: {file_id}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))


_storage: Optional[StorageBackend] = None


def get_storage() -> StorageBackend:
    global _storage
    if _storage is None:
        from .config import get_settings
        settings = get_settings()
        _storage = LocalStorage(settings.storage_root)
    return _storage
