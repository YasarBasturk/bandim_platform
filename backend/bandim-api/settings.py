import os
import json
import base64

VERSION = "v0.1.0"

REDIS_TTL: int = int(os.getenv("REDIS_TTL", default=8600))
REDIS_HOST: str = os.getenv("REDIS_HOST", default="localhost")
REDIS_PORT: int = int(os.getenv("REDIS_PORT", default=6379))
REDIS_DB: int = int(os.getenv("REDIS_DB", default=1))

SQLITE_DB: str = str(os.getenv("SQLITE_DB", default="./bandim.db"))

_API_KEYS = os.getenv("API_KEYS", default=None)
if _API_KEYS is None:
    API_KEYS = ["a6bf09ec-09a5-4e2d-a142-43e77e87b3d0"]
else:
    API_KEYS: str = json.loads(base64.b64decode(os.getenv("API_KEYS")))

JWT_SECRET: str = os.getenv("JWT_SECRET", default="secret_key")
JWT_EXPIRATION_TIME: int = int(os.getenv("JWT_EXPIRATION_TIME", default=12))
