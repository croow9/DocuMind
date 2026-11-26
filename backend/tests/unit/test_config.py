import pytest
from pydantic import ValidationError

from app.core.config import Settings


def make_settings(**overrides: object) -> Settings:
    # _env_file=None: тест не должен зависеть от того, что лежит в локальном .env
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_rate_limit_defaults_are_the_agreed_policy() -> None:
    settings = make_settings()

    assert settings.rate_limit_chat_per_minute == 20
    assert settings.rate_limit_web_per_minute == 5
    assert settings.rate_limit_web_per_day == 30
    assert settings.rate_limit_upload_per_minute == 10
    assert settings.rate_limit_login_per_account == 10
    assert settings.rate_limit_login_per_ip == 30
    assert settings.rate_limit_register_per_hour == 10


def test_web_questions_are_limited_more_strictly_than_questions_over_own_documents() -> (
    None
):
    settings = make_settings()

    # веб-режим стоит поиска в Tavily и многих эмбеддингов: он должен быть строже
    assert settings.rate_limit_web_per_minute < settings.rate_limit_chat_per_minute


@pytest.mark.parametrize(
    "name",
    [
        "rate_limit_chat_per_minute",
        "rate_limit_web_per_minute",
        "rate_limit_web_per_day",
        "rate_limit_upload_per_minute",
        "rate_limit_login_per_account",
        "rate_limit_login_per_ip",
        "rate_limit_register_per_hour",
        "max_upload_bytes",
        "max_documents_per_user",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_a_limit_below_one_is_rejected_instead_of_silently_blocking_everyone(
    name: str, value: int
) -> None:
    with pytest.raises(ValidationError):
        make_settings(**{name: value})


def test_limits_can_be_overridden_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RATE_LIMIT_WEB_PER_DAY", "7")

    assert make_settings().rate_limit_web_per_day == 7


def test_the_address_wide_login_limit_is_looser_than_the_per_account_one() -> None:
    settings = make_settings()

    # за одним адресом бывает много честных людей (офис, общая сеть)
    assert settings.rate_limit_login_per_ip > settings.rate_limit_login_per_account


def test_upload_ceilings_can_be_overridden_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1000")
    monkeypatch.setenv("MAX_DOCUMENTS_PER_USER", "5")

    settings = make_settings()

    assert (settings.max_upload_bytes, settings.max_documents_per_user) == (1000, 5)
