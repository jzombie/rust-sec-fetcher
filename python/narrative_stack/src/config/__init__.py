from .config import init_config
from .project_paths import project_paths

# Auto-init config
init_config()

__all__ = ["init_config", "project_paths"]
