import os
from .config import init_config
from pydantic import BaseModel

# Auto-init config to acquire environment vars
init_config()

class SimdRDriveServerConfig(BaseModel):
    host: str
    port: str

simd_r_drive_server_config = SimdRDriveServerConfig (
    host=os.getenv("SIMD_R_DRIVE_SERVER_HOST"),
    port=os.getenv("SIMD_R_DRIVE_SERVER_PORT"),
)

__all__ = ["SimdRDriveServerConfig", "simd_r_drive_server_config"]
