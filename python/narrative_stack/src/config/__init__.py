from .config import init_config
from .db_config import DBConfig, db_config
from .project_paths import project_paths

# Auto-init config
init_config()

__all__ = ["init_config", "DBConfig", "db_config", "project_paths"]
