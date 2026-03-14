from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "VoiceEngine",
    "export_custom_voice",
    "synthesize_to_file",
    "synthesize_zero_shot_to_file",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    engine_module = import_module(".engine", __name__)
    return getattr(engine_module, name)
