import hashlib
from math import ceil
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm

from app.api.deps import CacheDep, UserDep
from app.core.cache import Cache
from app.core.config import settings
from app.schemas.chat import MessageCreate
from app.schemas.user import normalize_email

MINUTE = 60
HOUR = 60 * MINUTE
DAY = 24 * HOUR
LOGIN_WINDOW = 15 * MINUTE


def _human_wait(seconds: int) -> str:
    """«через 42 с», «через 5 мин», «через 3 ч»: сутки в секундах человеку ни к чему."""
    if seconds < MINUTE:
        return f"{seconds} с"
    if seconds < 60 * MINUTE:
        return f"{ceil(seconds / MINUTE)} мин"
    return f"{ceil(seconds / (60 * MINUTE))} ч"


class RateLimit:
    """Не больше `limit` запросов за окно для одного «кого-то»: пользователя или адреса.

    Окно фиксированное: оно начинается с первого запроса и длится window_seconds.
    Запрос сверх лимита получает 429 с заголовком Retry-After. Отклонённые запросы
    тоже считаются: иначе счётчик не показывал бы, что клиент долбит сервис.
    Если счётчик недоступен (Redis не ответил), запрос проходит.

    Как зависимость FastAPI работает для вошедшего пользователя; для остальных
    случаев (вход, регистрация) вызывается check() с нужным ключом.
    """

    def __init__(
        self, scope: str, limit: int, window_seconds: int, description: str
    ) -> None:
        self.scope = scope
        self.limit = limit
        self.window_seconds = window_seconds
        self.description = description

    async def check(self, cache: Cache, identity: str) -> None:
        # ключ по тому, кого считаем, и по виду ограничения: у каждого свой счётчик
        state = await cache.incr(f"rl:{self.scope}:{identity}", self.window_seconds)
        if state is None or state.count <= self.limit:
            return
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Слишком много запросов ({self.description}). "
                f"Повторите через {_human_wait(state.resets_in_seconds)}"
            ),
            headers={"Retry-After": str(state.resets_in_seconds)},
        )

    async def __call__(self, user: UserDep, cache: CacheDep) -> None:
        await self.check(cache, str(user.id))


CHAT_PER_MINUTE = RateLimit(
    "chat", settings.rate_limit_chat_per_minute, MINUTE, "вопросы по своим документам"
)
WEB_PER_MINUTE = RateLimit(
    "web-minute", settings.rate_limit_web_per_minute, MINUTE, "вопросы по интернету"
)
WEB_PER_DAY = RateLimit(
    "web-day", settings.rate_limit_web_per_day, DAY, "вопросы по интернету за сутки"
)
UPLOAD_PER_MINUTE = RateLimit(
    "upload", settings.rate_limit_upload_per_minute, MINUTE, "загрузка документов"
)


async def limit_message_rate(
    payload: MessageCreate, user: UserDep, cache: CacheDep
) -> None:
    """Ограничение вопроса к чату: веб-режим считается отдельно и строже.

    Сначала проверка по минуте, потом по суткам: отклонённый минутным лимитом
    запрос не должен тратить дневную квоту (клиент в цикле выжег бы её за секунды).
    """
    if payload.web_search:
        await WEB_PER_MINUTE(user, cache)
        await WEB_PER_DAY(user, cache)
    else:
        await CHAT_PER_MINUTE(user, cache)


# --- вход и регистрация: до входа пользователя нет, считаем по адресу клиента


LOGIN_PER_ACCOUNT = RateLimit(
    "login-account",
    settings.rate_limit_login_per_account,
    LOGIN_WINDOW,
    "попытки входа",
)
LOGIN_PER_IP = RateLimit(
    "login-ip", settings.rate_limit_login_per_ip, LOGIN_WINDOW, "попытки входа"
)
REGISTER_PER_HOUR = RateLimit(
    "register", settings.rate_limit_register_per_hour, HOUR, "регистрации"
)


def client_ip(request: Request) -> str:
    """Адрес, с которого пришёл запрос.

    За обратным прокси (nginx, балансировщик) это будет адрес самого прокси, и все
    клиенты окажутся «одним адресом». Тогда uvicorn надо запускать с настройкой
    доверия заголовкам прокси (--proxy-headers и --forwarded-allow-ips). Заголовку
    X-Forwarded-For из запроса напрямую верить нельзя: его пишет клиент, и,
    меняя его, можно было бы обходить лимит.
    """
    return request.client.host if request.client else "unknown"


async def limit_login(
    request: Request,
    cache: CacheDep,
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> None:
    """Ограничение подбора пароля: по связке «адрес + аккаунт» и по адресу в целом.

    Сначала узкая проверка, потом широкая: отклонённая узкой попытка не должна
    тратить общий запас адреса (за одним адресом бывает много честных людей).
    Аккаунт в ключе только в виде отпечатка: адреса почты в Redis не нужны.
    """
    ip = client_ip(request)
    account = hashlib.sha256(normalize_email(form.username).encode()).hexdigest()[:16]
    await LOGIN_PER_ACCOUNT.check(cache, f"{ip}:{account}")
    await LOGIN_PER_IP.check(cache, ip)


async def limit_register(request: Request, cache: CacheDep) -> None:
    await REGISTER_PER_HOUR.check(cache, client_ip(request))
