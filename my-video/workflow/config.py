from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkflowPaths:
    module_dir: Path
    output_dir: Path


def get_paths() -> WorkflowPaths:
    module_dir = Path(__file__).resolve().parent
    return WorkflowPaths(
        module_dir=module_dir,
        output_dir=module_dir / "output",
    )
