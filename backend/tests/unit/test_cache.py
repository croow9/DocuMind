from unittest.mock import AsyncMock, MagicMock

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.core.cache import InMemoryCache, RedisCache, WindowCount, build_cache
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


# --- счётчик окна


def make_counter(results: object = None, error: Exception | None = None):  # type: ignore[no-untyped-def]
    """RedisCache с подставным конвейером: что в него положили и что он вернул."""
    pipe = MagicMock()
    pipe.execute = AsyncMock(return_value=results, side_effect=error)
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=pipe)
    context.__aexit__ = AsyncMock(return_value=False)
    client = MagicMock()
    client.pipeline = MagicMock(return_value=context)
    return RedisCache(client), client, pipe


async def test_counter_runs_incr_expire_and_ttl_in_one_transaction() -> None:
    cache, client, pipe = make_counter([3, True, 41])

    state = await cache.incr("rl:chat:u1", 60)

    assert state == WindowCount(count=3, resets_in_seconds=41)
    # одной транзакцией: без неё сбой между командами оставил бы ключ без срока
    client.pipeline.assert_called_once_with(transaction=True)
    pipe.incr.assert_called_once_with("rl:chat:u1")
    # nx=True: срок только новому ключу, иначе каждый запрос продлевал бы окно
    pipe.expire.assert_called_once_with("rl:chat:u1", 60, nx=True)
    pipe.ttl.assert_called_once_with("rl:chat:u1")
    pipe.execute.assert_awaited_once()


@pytest.mark.parametrize(
    ("ttl", "expected"),
    [(41, 41), (1, 1), (0, 1), (-1, 60), (-2, 60)],
)
async def test_odd_ttl_from_redis_never_gives_a_zero_or_negative_wait(
    ttl: int, expected: int
) -> None:
    cache, _, _ = make_counter([1, True, ttl])

    state = await cache.incr("k", 60)

    assert state is not None and state.resets_in_seconds == expected


@pytest.mark.parametrize("error", [RedisConnectionError("нет связи"), OSError("сеть")])
async def test_counter_failure_means_no_count_not_an_error(error: Exception) -> None:
    cache, _, _ = make_counter(error=error)

    assert await cache.incr("k", 60) is None


async def test_programming_errors_in_the_counter_are_not_hidden() -> None:
    cache, _, _ = make_counter(error=KeyError("баг"))

    with pytest.raises(KeyError):
        await cache.incr("k", 60)


async def test_in_memory_counter_counts_per_key_and_remembers_the_window() -> None:
    cache = InMemoryCache()

    assert await cache.incr("a", 60) == WindowCount(1, 60)
    assert await cache.incr("a", 60) == WindowCount(2, 60)
    assert await cache.incr("b", 3600) == WindowCount(1, 3600)
    assert cache.counters == {"a": 2, "b": 1}
    assert cache.ttls == {"a": 60, "b": 3600}


async def test_in_memory_counter_starts_over_when_the_window_is_over() -> None:
    cache = InMemoryCache()
    await cache.incr("a", 60)
    await cache.incr("a", 60)

    del cache.counters["a"]  # так тест изображает конец окна

    assert await cache.incr("a", 60) == WindowCount(1, 60)
