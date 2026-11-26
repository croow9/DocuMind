from unittest.mock import AsyncMock

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.core.cache import InMemoryCache, RedisCache, build_cache
from app.core.config import settings


def make_cache(**client_methods: object) -> tuple[RedisCache, AsyncMock]:
    client = AsyncMock()
    for name, value in client_methods.items():
        setattr(client, name, value)
    return RedisCache(client), client


async def test_value_is_written_with_its_lifetime_and_read_back() -> None:
    cache, client = make_cache(get=AsyncMock(return_value="значение"))

    await cache.set("ключ", "значение", 120)

    client.set.assert_awaited_once_with("ключ", "значение", ex=120)
    assert await cache.get("ключ") == "значение"


async def test_missing_key_is_none() -> None:
    cache, _ = make_cache(get=AsyncMock(return_value=None))

    assert await cache.get("нет") is None


async def test_redis_failure_on_read_is_a_miss_not_an_error() -> None:
    cache, _ = make_cache(get=AsyncMock(side_effect=RedisConnectionError("нет связи")))

    assert await cache.get("ключ") is None


async def test_redis_failure_on_write_is_swallowed() -> None:
    cache, _ = make_cache(set=AsyncMock(side_effect=RedisConnectionError("нет связи")))

    await cache.set("ключ", "значение", 60)  # не бросает


async def test_os_level_network_error_is_also_swallowed() -> None:
    cache, _ = make_cache(
        get=AsyncMock(side_effect=OSError("сеть")),
        set=AsyncMock(side_effect=OSError("сеть")),
    )

    assert await cache.get("ключ") is None
    await cache.set("ключ", "значение", 60)


async def test_programming_errors_are_not_hidden() -> None:
    cache, _ = make_cache(get=AsyncMock(side_effect=KeyError("баг")))

    with pytest.raises(KeyError):
        await cache.get("ключ")


async def test_close_releases_the_client() -> None:
    cache, client = make_cache()

    await cache.close()

    client.aclose.assert_awaited_once()


async def test_in_memory_cache_remembers_values_and_lifetimes() -> None:
    cache = InMemoryCache()

    await cache.set("a", "1", 30)

    assert await cache.get("a") == "1"
    assert await cache.get("b") is None
    assert cache.ttls == {"a": 30}
    await cache.close()  # по умолчанию ничего не делает


async def test_built_cache_is_lazy_and_does_not_need_redis_at_start() -> None:
    cache = build_cache()
    try:
        assert isinstance(cache, RedisCache)
    finally:
        await cache.close()


def test_default_lifetime_of_web_cache_is_a_day() -> None:
    assert settings.web_cache_ttl_seconds == 86_400


async def test_bytes_from_a_client_without_decoding_are_returned_as_text() -> None:
    cache, _ = make_cache(get=AsyncMock(return_value="значение".encode()))

    assert await cache.get("ключ") == "значение"
