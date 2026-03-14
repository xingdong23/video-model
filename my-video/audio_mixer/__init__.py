from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "BgmEngine",
    "BgmLibraryTrack",
    "BgmMixResult",
    "VideoAudioInfo",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    engine_module = import_module(".engine", __name__)
    return getattr(engine_module, name)
