import logging
from typing import Union, Any
from redis import asyncio as aioredis

# from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import create_engine
from settings import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    SQLITE_DB,
)


class RedisCache:

    def __init__(self, url: str) -> None:
        self.url = url
        self.redis_cache = None

    async def init_cache(self) -> None:
        pool = aioredis.ConnectionPool.from_url(self.url)
        self.redis_cache = aioredis.Redis.from_pool(pool)

    async def keys(self, pattern: str) -> Union[None, Any]:
        if self.redis_cache is not None:
            return await self.redis_cache.keys(pattern)
        else:
            logging.debug(
                "A Redis connection pool has not yet been initialized. "
                + "Used keys can thus not be retrieved from Redis. "
                + "Please call method .init_cache()",
            )
            return None

    async def set(self, key: str, value: Any, expire: int = 0) -> Union[None, bool]:
        if self.redis_cache is not None:
            return await self.redis_cache.set(
                key=key,
                value=value,
                expire=expire,
            )
        else:
            logging.debug(
                "A Redis connection pool has not yet been initialized. "
                + "A key-value pair can thus not be set in Redis. "
                + "Please call method .init_cache()",
            )
            return None

    async def get(self, key: str) -> Union[None, Any]:
        if self.redis_cache is not None:
            return await self.redis_cache.get(key=key)
        else:
            logging.debug(
                "A Redis connection pool has not yet been initialized. "
                + "A value can thus not retrieved from Redis. "
                + "Please call method .init_cache()",
            )
            return None

    async def close(self) -> None:
        if self.redis_cache is not None:
            await self.redis_cache.close()
            # await self.redis_cache.wait_closed()
        else:
            logging.debug(
                "A Redis connection pool has not yet been initialized. "
                + "The connection to Redis can thus not be closed. "
                + "Please call method .init_cache()",
            )
            return None


# SQLALCHEMY_DATABASE_URL = "sqlite+aiosqlite:///./bandim.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{SQLITE_DB}"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
# engine = create_async_engine(SQLALCHEMY_DATABASE_URL)

# class SqliteDB():

#     def __init__(self):
#         """Initialization."""
#         # self.sqlite_file_name = f"{DATABASE_VOLUME}/bandim.db"
#         self.sqlite_url = f"sqlite+aiosqlite:///{self.sqlite_file_name}"

#         self.engine = create_async_engine(self.sqlite_url, echo=True)


#     async def init_db_and_tables(self):
#         """Initialize the database and the tables."""
#         async with self.engine.begin() as session:
#             await session.run_sync(SQLModel.metadata.create_all)

REDIS_DATABASE_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}?encoding=utf-8"
redis_cache = RedisCache(url=REDIS_DATABASE_URL)
