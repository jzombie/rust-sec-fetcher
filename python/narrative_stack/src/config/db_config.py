import os
from .config import init_config
from pydantic import BaseModel

init_config()


class DBConfig(BaseModel):
    host: str
    port: int
    user: str
    password: str
    database: str

db_config = DBConfig(
    host=os.getenv("MYSQL_HOST"),
    port=os.getenv("MYSQL_PORT"),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DATABASE"),
)

__all__ = ["DBConfig", "db_config"]
