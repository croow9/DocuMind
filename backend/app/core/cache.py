import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import redis.asyncio as redis
from redis.exceptions import RedisError

from app.core.config import settings

logger = logging.getLogger(__name__)

# Redis должен отвечать мгновенно: если не отвечает, лучше обойтись без кэша,
# чем заставлять пользователя ждать
REDIS_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class WindowCount:
    """Сколько раз ключ увеличили в текущем окне и через сколько оно закончится."""

    count: int
    resets_in_seconds: int


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
    async def incr(self, key: str, window_seconds: int) -> WindowCount | None:
        """Увеличить счётчик ключа на 1 и вернуть его значение вместе с остатком окна.

        Окно начинается с первого увеличения и длится window_seconds, потом ключ
        исчезает сам и счёт идёт заново. Последующие увеличения окно не продлевают.
        None значит «кэш не ответил, счёта нет»: ограничитель в этом случае
        никого не блокирует (лучше без защиты, чем без сервиса).
        """
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

    async def incr(self, key: str, window_seconds: int) -> WindowCount | None:
        try:
            # Одна транзакция: все три команды выполняются подряд и целиком. Если
            # ставить срок отдельным запросом, сбой между ними оставил бы ключ без
            # срока жизни, и пользователь был бы заблокирован навсегда
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.incr(key)
                # NX: срок ставится только новому ключу. Без него каждый запрос
                # продлевал бы окно, и активного пользователя не отпускало бы
                pipe.expire(key, window_seconds, nx=True)
                pipe.ttl(key)
                count, _, ttl = await pipe.execute()
        except (RedisError, OSError):
            logger.warning(
                "Redis не ответил на счётчик, ограничение не применяется",
                exc_info=True,
            )
            return None
        # TTL бывает отрицательным (-1 — срока нет, -2 — ключа нет): берём окно целиком
        if ttl < 0:
            ttl = window_seconds
        return WindowCount(int(count), max(ttl, 1))

    async def close(self) -> None:
        await self._client.aclose()


class InMemoryCache(Cache):
    """Кэш в памяти процесса: для тестов. Время жизни запоминает, но не применяет.

    Конец окна счётчика тест изображает сам: убирает ключ из counters.
    """

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.counters: dict[str, int] = {}

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self.values[key] = value
        self.ttls[key] = ttl_seconds

    async def incr(self, key: str, window_seconds: int) -> WindowCount | None:
        self.counters[key] = self.counters.get(key, 0) + 1
        self.ttls[key] = window_seconds
        return WindowCount(self.counters[key], window_seconds)

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
