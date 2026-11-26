from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, status
from starlette.testclient import TestClient

from app.api.deps import get_cache
from app.api.rate_limit import (
    HOUR,
    LOGIN_PER_ACCOUNT,
    LOGIN_PER_IP,
    LOGIN_WINDOW,
    REGISTER_PER_HOUR,
    client_ip,
)
from app.core.cache import Cache, InMemoryCache
from app.core.config import settings
from app.db.session import get_db
from app.main import app

PASSWORD = "Correct-Pass-1"
REGISTER_BODY = {"email": "new@example.com", "password": PASSWORD}


class FakeAuthService:
    """Подставной сервис: пароль верный только «Correct-Pass-1»."""

    def __init__(self, db: object) -> None:
        pass

    async def authenticate_user(self, email: str, password: str) -> str:
        if password != PASSWORD:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED, "Неверный email или пароль"
            )
        return "token"

    async def register_new_user(self, user_in: object) -> SimpleNamespace:
        return SimpleNamespace(
            id="3f0c2a52-6a38-4c0e-9a3a-0d5d3f3f9b11",
            email="new@example.com",
            created_at="2026-01-01T00:00:00Z",
        )


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemoryCache]:
    monkeypatch.setattr("app.api.v1.auth.AuthService", FakeAuthService)
    monkeypatch.setattr(LOGIN_PER_ACCOUNT, "limit", 2)
    monkeypatch.setattr(LOGIN_PER_IP, "limit", 4)
    monkeypatch.setattr(REGISTER_PER_HOUR, "limit", 2)
    shared = InMemoryCache()
    app.dependency_overrides[get_cache] = lambda: shared
    app.dependency_overrides[get_db] = lambda: AsyncMock()
    yield shared
    app.dependency_overrides.clear()


@pytest.fixture
def client(cache: InMemoryCache) -> TestClient:
    return TestClient(app, client=("203.0.113.5", 50000))


def login(
    client: TestClient, email: str = "me@example.com", password: str = "wrong-pass"
) -> int:
    return client.post(
        "/api/v1/auth/login", data={"username": email, "password": password}
    ).status_code


# --- подбор пароля к одному аккаунту


def test_wrong_passwords_are_401_until_the_limit_then_429(client: TestClient) -> None:
    codes = [login(client) for _ in range(4)]

    assert codes == [401, 401, 429, 429]


def test_even_the_correct_password_is_refused_after_the_limit(
    client: TestClient,
) -> None:
    # иначе перебор шёл бы дальше: подходящий пароль в потоке неверных нашёлся бы
    for _ in range(2):
        login(client)

    assert login(client, password=PASSWORD) == 429


def test_successful_logins_count_too_but_within_the_limit_they_pass(
    client: TestClient,
) -> None:
    assert [login(client, password=PASSWORD) for _ in range(2)] == [200, 200]


def test_the_429_names_the_reason_and_says_how_long_to_wait(client: TestClient) -> None:
    for _ in range(2):
        login(client)

    response = client.post(
        "/api/v1/auth/login", data={"username": "me@example.com", "password": "x"}
    )

    assert response.status_code == 429
    assert response.headers["retry-after"] == str(LOGIN_WINDOW)
    assert response.json()["detail"] == (
        "Слишком много запросов (попытки входа). Повторите через 15 мин"
    )


def test_a_neighbour_account_is_not_locked_by_attempts_against_another(
    client: TestClient,
) -> None:
    for _ in range(3):
        login(client, "victim@example.com")

    assert login(client, "other@example.com") == 401  # не 429


def test_the_same_account_in_another_spelling_is_the_same_bucket(
    client: TestClient,
) -> None:
    # иначе «A@x.com» и «a@x.com» давали бы вдвое больше попыток
    login(client, "Me@Example.com")
    login(client, "  me@example.com ")

    assert login(client, "ME@EXAMPLE.COM") == 429


# --- подбор по многим аккаунтам с одного адреса


def test_many_accounts_from_one_address_run_into_the_address_limit(
    client: TestClient,
) -> None:
    # на аккаунт по 2 попытки, на адрес 4: пятая попытка с адреса отклоняется,
    # хотя у этого аккаунта попыток ещё не было
    codes = [
        login(client, email)
        for email in ("a@x.com", "a@x.com", "b@x.com", "b@x.com", "c@x.com")
    ]

    assert codes == [401, 401, 401, 401, 429]


def test_an_attempt_refused_by_the_account_does_not_spend_the_address_budget(
    client: TestClient, cache: InMemoryCache
) -> None:
    for _ in range(5):  # 2 принятых и 3 отклонённых по аккаунту
        login(client, "a@x.com")

    assert cache.counters[f"rl:login-ip:{'203.0.113.5'}"] == 2


def test_addresses_are_counted_separately(
    cache: InMemoryCache, client: TestClient
) -> None:
    for _ in range(3):
        login(client)
    elsewhere = TestClient(app, client=("198.51.100.9", 40000))

    assert login(client) == 429
    assert login(elsewhere) == 401


# --- что попадает в Redis


def test_counters_do_not_store_the_email_in_clear(
    client: TestClient, cache: InMemoryCache
) -> None:
    login(client, "secret.person@example.com")

    keys = list(cache.counters)
    assert not any("secret" in key or "@" in key for key in keys)
    assert sorted(key.split(":")[1] for key in keys) == ["login-account", "login-ip"]
    account_key = next(key for key in keys if key.startswith("rl:login-account:"))
    assert account_key.startswith("rl:login-account:203.0.113.5:")


# --- когда считать нечего


def test_an_invalid_form_is_422_and_is_not_counted(
    client: TestClient, cache: InMemoryCache
) -> None:
    response = client.post("/api/v1/auth/login", data={"username": "me@example.com"})

    assert response.status_code == 422
    assert cache.counters == {}


def test_when_the_counter_is_unavailable_login_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DeadCache(InMemoryCache):
        async def incr(self, key: str, window_seconds: int):  # type: ignore[no-untyped-def]
            return None

    monkeypatch.setattr("app.api.v1.auth.AuthService", FakeAuthService)
    monkeypatch.setattr(LOGIN_PER_ACCOUNT, "limit", 1)
    app.dependency_overrides[get_cache] = lambda: DeadCache()
    app.dependency_overrides[get_db] = lambda: AsyncMock()
    try:
        client = TestClient(app)
        codes = [login(client) for _ in range(3)]
    finally:
        app.dependency_overrides.clear()

    assert codes == [401, 401, 401]


# --- регистрация


def test_registrations_from_one_address_are_limited(client: TestClient) -> None:
    codes = [
        client.post("/api/v1/auth/register", json=REGISTER_BODY).status_code
        for _ in range(3)
    ]

    assert codes == [201, 201, 429]


def test_the_registration_429_says_to_wait_up_to_an_hour(client: TestClient) -> None:
    for _ in range(2):
        client.post("/api/v1/auth/register", json=REGISTER_BODY)

    response = client.post("/api/v1/auth/register", json=REGISTER_BODY)

    assert response.headers["retry-after"] == str(HOUR)
    assert response.json()["detail"] == (
        "Слишком много запросов (регистрации). Повторите через 1 ч"
    )


def test_registrations_from_another_address_have_their_own_budget(
    client: TestClient,
) -> None:
    for _ in range(3):
        client.post("/api/v1/auth/register", json=REGISTER_BODY)
    elsewhere = TestClient(app, client=("198.51.100.9", 40000))

    assert (
        client.post("/api/v1/auth/register", json=REGISTER_BODY).status_code,
        elsewhere.post("/api/v1/auth/register", json=REGISTER_BODY).status_code,
    ) == (429, 201)


def test_registration_and_login_have_separate_budgets(client: TestClient) -> None:
    for _ in range(2):
        client.post("/api/v1/auth/register", json=REGISTER_BODY)

    assert login(client) == 401  # вход не заблокирован регистрациями


def test_a_registration_with_a_bad_body_is_422_but_still_counted(
    client: TestClient, cache: InMemoryCache
) -> None:
    # лимит стоит на маршруте и срабатывает раньше разбора тела: битые запросы тоже
    # упираются в лимит, спамить ими нельзя (на входе иначе: там форма сама зависимость)
    codes = [
        client.post("/api/v1/auth/register", json={"email": "not-an-email"}).status_code
        for _ in range(3)
    ]

    assert codes == [422, 422, 429]


# --- адрес клиента


def fake_request(host: str | None, **headers: str) -> SimpleNamespace:
    client = SimpleNamespace(host=host) if host else None
    return SimpleNamespace(client=client, headers=headers)


def test_client_ip_is_the_address_the_connection_came_from() -> None:
    assert client_ip(fake_request("203.0.113.5")) == "203.0.113.5"  # type: ignore[arg-type]


def test_a_forwarded_header_from_the_client_cannot_change_the_address() -> None:
    # заголовок пишет сам клиент: поверив ему, лимит обходился бы одной строкой
    request = fake_request("203.0.113.5", **{"x-forwarded-for": "10.9.9.9"})

    assert client_ip(request) == "203.0.113.5"  # type: ignore[arg-type]


def test_without_a_known_address_all_such_requests_share_one_bucket() -> None:
    assert client_ip(fake_request(None)) == "unknown"  # type: ignore[arg-type]


# --- настоящие значения


def test_the_instances_use_the_agreed_windows_and_settings() -> None:
    assert LOGIN_WINDOW == 15 * 60
    assert HOUR == 60 * 60
    assert LOGIN_PER_ACCOUNT.window_seconds == 900
    assert LOGIN_PER_IP.window_seconds == 900
    assert REGISTER_PER_HOUR.window_seconds == 3600
    assert LOGIN_PER_ACCOUNT.limit == settings.rate_limit_login_per_account
    assert LOGIN_PER_IP.limit == settings.rate_limit_login_per_ip
    assert REGISTER_PER_HOUR.limit == settings.rate_limit_register_per_hour
    assert LOGIN_PER_ACCOUNT.scope != LOGIN_PER_IP.scope


def test_cache_is_the_abstract_type_the_limiter_works_with() -> None:
    assert issubclass(InMemoryCache, Cache)
