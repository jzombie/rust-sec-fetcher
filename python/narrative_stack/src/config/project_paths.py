from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from utils.os import get_project_root_path

PYTHON_ROOT = get_project_root_path()

@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """
    Common project paths for the Python and underlying Rust project environments.
    """

    python_root: Path = PYTHON_ROOT
    python_data: Path = PYTHON_ROOT / "data"
    rust_data: Path = (PYTHON_ROOT.parent.parent / "data").resolve()

project_paths = ProjectPaths()

__all__ = ["project_paths"]
