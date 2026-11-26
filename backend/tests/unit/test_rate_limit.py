from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.rate_limit import (
    CHAT_PER_MINUTE,
    DAY,
    MINUTE,
    UPLOAD_PER_MINUTE,
    WEB_PER_DAY,
    WEB_PER_MINUTE,
    RateLimit,
    _human_wait,
    limit_message_rate,
)
from app.core.cache import InMemoryCache, WindowCount
from app.core.config import settings
from app.schemas.chat import MessageCreate


def make_user() -> SimpleNamespace:
    return SimpleNamespace(id=uuid4())


async def hit(limit: RateLimit, user: SimpleNamespace, cache: InMemoryCache) -> None:
    await limit(user, cache)  # type: ignore[arg-type]


# --- RateLimit: один вид ограничения


async def test_requests_up_to_the_limit_pass_and_the_next_one_is_429() -> None:
    limit = RateLimit("chat", 3, MINUTE, "вопросы")
    user, cache = make_user(), InMemoryCache()

    for _ in range(3):  # ровно «лимит» запросов проходит
        await hit(limit, user, cache)
    with pytest.raises(HTTPException) as exc_info:
        await hit(limit, user, cache)

    assert exc_info.value.status_code == 429


async def test_the_answer_says_what_is_limited_and_how_long_to_wait() -> None:
    limit = RateLimit("chat", 1, MINUTE, "вопросы по своим документам")
    user, cache = make_user(), InMemoryCache()
    await hit(limit, user, cache)

    with pytest.raises(HTTPException) as exc_info:
        await hit(limit, user, cache)

    assert exc_info.value.headers == {"Retry-After": "60"}
    assert "вопросы по своим документам" in exc_info.value.detail
    assert "1 мин" in exc_info.value.detail


async def test_retry_after_is_what_remains_of_the_window_not_the_whole_window() -> None:
    cache = AsyncMock()
    cache.incr.return_value = WindowCount(count=6, resets_in_seconds=17)
    limit = RateLimit("chat", 5, MINUTE, "вопросы")

    with pytest.raises(HTTPException) as exc_info:
        await limit(make_user(), cache)

    assert exc_info.value.headers == {"Retry-After": "17"}
    assert "17 с" in exc_info.value.detail


async def test_rejected_requests_are_counted_too() -> None:
    limit = RateLimit("chat", 1, MINUTE, "вопросы")
    user, cache = make_user(), InMemoryCache()

    for _ in range(5):
        try:
            await hit(limit, user, cache)
        except HTTPException:
            pass

    assert cache.counters == {f"rl:chat:{user.id}": 5}


async def test_counter_key_holds_the_kind_of_limit_and_the_user() -> None:
    user, cache = make_user(), InMemoryCache()

    await hit(RateLimit("upload", 5, MINUTE, "загрузка"), user, cache)

    assert list(cache.counters) == [f"rl:upload:{user.id}"]
    assert cache.ttls[f"rl:upload:{user.id}"] == MINUTE


async def test_users_do_not_share_a_counter() -> None:
    limit = RateLimit("chat", 1, MINUTE, "вопросы")
    cache = InMemoryCache()
    first, second = make_user(), make_user()
    await hit(limit, first, cache)

    await hit(limit, second, cache)  # у второго свой счётчик
    with pytest.raises(HTTPException):
        await hit(limit, first, cache)


async def test_kinds_of_limits_do_not_share_a_counter() -> None:
    chat = RateLimit("chat", 1, MINUTE, "вопросы")
    upload = RateLimit("upload", 1, MINUTE, "загрузка")
    user, cache = make_user(), InMemoryCache()
    await hit(chat, user, cache)

    await hit(upload, user, cache)  # загрузка не тратит квоту вопросов


async def test_a_new_window_starts_clean() -> None:
    limit = RateLimit("chat", 1, MINUTE, "вопросы")
    user, cache = make_user(), InMemoryCache()
    await hit(limit, user, cache)
    with pytest.raises(HTTPException):
        await hit(limit, user, cache)

    cache.counters.clear()  # окно кончилось

    await hit(limit, user, cache)


async def test_when_the_counter_is_unavailable_nobody_is_blocked() -> None:
    cache = AsyncMock()
    cache.incr.return_value = None
    limit = RateLimit("chat", 1, MINUTE, "вопросы")

    for _ in range(5):
        await limit(make_user(), cache)  # не бросает


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (1, "1 с"),
        (59, "59 с"),
        (60, "1 мин"),
        (61, "2 мин"),
        (3599, "60 мин"),
        (3600, "1 ч"),
        (3601, "2 ч"),
        (DAY, "24 ч"),
    ],
)
def test_waiting_time_is_shown_in_the_unit_a_person_understands(
    seconds: int, expected: str
) -> None:
    assert _human_wait(seconds) == expected


# --- настоящие пределы приложения берутся из настроек


def test_application_limits_come_from_settings_with_the_right_windows() -> None:
    assert (CHAT_PER_MINUTE.limit, CHAT_PER_MINUTE.window_seconds) == (
        settings.rate_limit_chat_per_minute,
        MINUTE,
    )
    assert (WEB_PER_MINUTE.limit, WEB_PER_MINUTE.window_seconds) == (
        settings.rate_limit_web_per_minute,
        MINUTE,
    )
    assert (WEB_PER_DAY.limit, WEB_PER_DAY.window_seconds) == (
        settings.rate_limit_web_per_day,
        DAY,
    )
    assert (UPLOAD_PER_MINUTE.limit, UPLOAD_PER_MINUTE.window_seconds) == (
        settings.rate_limit_upload_per_minute,
        MINUTE,
    )


def test_application_limits_have_distinct_counters() -> None:
    scopes = [
        limit.scope
        for limit in (CHAT_PER_MINUTE, WEB_PER_MINUTE, WEB_PER_DAY, UPLOAD_PER_MINUTE)
    ]

    assert len(set(scopes)) == 4


# --- вопрос к чату: свои документы и интернет считаются отдельно


def question(web: bool) -> MessageCreate:
    return MessageCreate(question="как?", web_search=web)


async def ask(
    payload: MessageCreate, user: SimpleNamespace, cache: InMemoryCache
) -> None:
    await limit_message_rate(payload, user, cache)  # type: ignore[arg-type]


async def test_a_question_over_own_documents_uses_only_the_chat_counter() -> None:
    user, cache = make_user(), InMemoryCache()

    await ask(question(web=False), user, cache)

    assert cache.counters == {f"rl:{CHAT_PER_MINUTE.scope}:{user.id}": 1}


async def test_a_web_question_uses_both_web_counters_and_not_the_chat_one() -> None:
    user, cache = make_user(), InMemoryCache()

    await ask(question(web=True), user, cache)

    assert cache.counters == {
        f"rl:{WEB_PER_MINUTE.scope}:{user.id}": 1,
        f"rl:{WEB_PER_DAY.scope}:{user.id}": 1,
    }


async def test_web_questions_do_not_spend_the_quota_of_ordinary_ones_and_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(CHAT_PER_MINUTE, "limit", 1)
    monkeypatch.setattr(WEB_PER_MINUTE, "limit", 100)
    monkeypatch.setattr(WEB_PER_DAY, "limit", 100)
    user, cache = make_user(), InMemoryCache()

    for _ in range(5):
        await ask(question(web=True), user, cache)
    await ask(question(web=False), user, cache)  # своя квота цела
    with pytest.raises(HTTPException):
        await ask(question(web=False), user, cache)
    await ask(question(web=True), user, cache)  # и интернет не пострадал


async def test_a_question_rejected_by_the_minute_limit_does_not_spend_the_daily_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(WEB_PER_MINUTE, "limit", 1)
    monkeypatch.setattr(WEB_PER_DAY, "limit", 100)
    user, cache = make_user(), InMemoryCache()
    await ask(question(web=True), user, cache)

    for _ in range(10):  # клиент в цикле стучится в закрытую дверь
        with pytest.raises(HTTPException):
            await ask(question(web=True), user, cache)

    assert cache.counters[f"rl:{WEB_PER_DAY.scope}:{user.id}"] == 1


async def test_the_daily_limit_stops_web_questions_even_when_the_minute_is_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(WEB_PER_MINUTE, "limit", 100)
    monkeypatch.setattr(WEB_PER_DAY, "limit", 2)
    user, cache = make_user(), InMemoryCache()
    await ask(question(web=True), user, cache)
    await ask(question(web=True), user, cache)

    with pytest.raises(HTTPException) as exc_info:
        await ask(question(web=True), user, cache)

    assert exc_info.value.status_code == 429
    assert exc_info.value.headers == {"Retry-After": str(DAY)}
