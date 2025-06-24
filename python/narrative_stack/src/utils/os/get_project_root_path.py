import logging
import os
from pathlib import Path
from typing import Union
from .to_path import to_path


def get_project_root_path(relative_path: Union[str, Path] = "") -> Path:
    """
    Return the Path to the directory that contains `pyproject.toml`.
    Optionally, append a relative sub-path under that root.

    :param relative_path: Optional relative path from the project root.
    :return: Absolute Path object.
    :raises FileNotFoundError: If `pyproject.toml` is not found.
    """
    logging.info("Original cwd: %s", os.getcwd())

    current = Path.cwd()
    project_root: Path | None = None
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").is_file():
            project_root = parent
            break

    if project_root is None:
        raise FileNotFoundError(
            "Could not find pyproject.toml in current or parent directories."
        )

    rel_path = to_path(relative_path)  # ensure Path type

    # Treat "", ".", or Path(".") as "no extra path"
    if str(rel_path) in ("", "."):
        target_path = project_root
    else:
        target_path = project_root / rel_path

    resolved_path = target_path.resolve()
    logging.info("Project root path: %s", resolved_path)
    return resolved_path
