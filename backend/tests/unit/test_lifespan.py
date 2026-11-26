from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.testclient import TestClient

from app.api.deps import get_cache
from app.core.cache import Cache, InMemoryCache
from app.main import app


class TrackingCache(InMemoryCache):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def start_without_outside_services(monkeypatch: pytest.MonkeyPatch) -> TrackingCache:
    """Подменяет Kafka, Redis и базу: lifespan проходит без настоящих сервисов."""
    cache = TrackingCache()
    monkeypatch.setattr("app.main.build_cache", lambda: cache)
    monkeypatch.setattr(
        "app.main.KafkaProducerManager",
        lambda **_: SimpleNamespace(start=AsyncMock(), stop=AsyncMock()),
    )
    monkeypatch.setattr("app.main.engine", SimpleNamespace(dispose=AsyncMock()))
    return cache


def test_the_cache_is_shared_between_dependencies_and_web_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = start_without_outside_services(monkeypatch)

    with TestClient(app) as client:
        assert client.app.state.cache is cache  # type: ignore[attr-defined]
        # один и тот же экземпляр: иначе у поиска и у ограничителя были бы разные кэши
        assert client.app.state.web_search.cache is cache  # type: ignore[attr-defined]


def test_the_cache_is_closed_when_the_app_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = start_without_outside_services(monkeypatch)

    with TestClient(app):
        assert cache.closed is False

    assert cache.closed is True


def test_dependency_hands_out_the_cache_from_app_state() -> None:
    cache: Cache = InMemoryCache()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(cache=cache)))

    assert get_cache(request) is cache  # type: ignore[arg-type]
