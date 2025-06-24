from pathlib import Path
from typing import Union


def to_path(x: Union[str, Path], *, as_str: bool = False) -> Union[str, Path]:
    """Convert input to a Path or str, depending on `as_str`."""
    path = x if isinstance(x, Path) else Path(x)
    return str(path) if as_str else path
