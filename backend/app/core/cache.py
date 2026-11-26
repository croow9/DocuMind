import logging
from abc import ABC, abstractmethod

import redis.asyncio as redis
from redis.exceptions import RedisError

from app.core.config import settings

logger = logging.getLogger(__name__)

# Redis должен отвечать мгновенно: если не отвечает, лучше обойтись без кэша,
# чем заставлять пользователя ждать
REDIS_TIMEOUT_SECONDS = 2.0


class Cache(ABC):
    """Хранилище «ключ -> строка» с временем жизни.

    Кэш только ускоряет: его недоступность не должна ломать ответ, поэтому
    реализации не бросают исключений, а на сбое ведут себя как пустой кэш.
    """

    @abstractmethod
    async def get(self, key: str) -> str | None:
        raise NotImplementedError

    @abstractmethod
    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """Освободить соединения."""
        raise NotImplementedError


class RedisCache(Cache):
    def __init__(self, client: redis.Redis) -> None:
        self._client = client

    async def get(self, key: str) -> str | None:
        try:
            value = await self._client.get(key)
        except (RedisError, OSError):
            logger.warning(
                "Redis не ответил на чтение, работаю без кэша", exc_info=True
            )
            return None
        # клиент создан с decode_responses=True, но тип ответа у него общий
        return value.decode() if isinstance(value, bytes) else value

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        try:
            await self._client.set(key, value, ex=ttl_seconds)
        except (RedisError, OSError):
            logger.warning("Redis не принял запись, работаю без кэша", exc_info=True)

    async def close(self) -> None:
        await self._client.aclose()


class InMemoryCache(Cache):
    """Кэш в памяти процесса: для тестов. Время жизни запоминает, но не применяет."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self.values[key] = value
        self.ttls[key] = ttl_seconds

    async def close(self) -> None:
        """Освобождать нечего."""


def build_cache() -> Cache:
    """Кэш в Redis из настроек; сам клиент подключается лениво, при первом обращении."""
    client = redis.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=REDIS_TIMEOUT_SECONDS,
        socket_timeout=REDIS_TIMEOUT_SECONDS,
    )
    return RedisCache(client)
