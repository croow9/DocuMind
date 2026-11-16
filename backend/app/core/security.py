from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import bcrypt
import jwt

from app.core.config import settings

# bcrypt умеет хэшировать не больше 72 байт пароля (не символов!)
BCRYPT_MAX_BYTES = 72


def hash_password(password: str) -> str:
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(),
    ).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    plain_bytes = plain.encode("utf-8")
    # такой пароль не мог быть задан при регистрации, а bcrypt на нём бросил бы
    # ValueError (500 в логине): просто считаем его неверным
    if len(plain_bytes) > BCRYPT_MAX_BYTES:
        return False
    return bcrypt.checkpw(plain_bytes, hashed.encode("utf-8"))


def create_access_token(user_id: UUID) -> str:
    now = datetime.now(UTC)

    payload: dict[str, Any] = {
        "sub": str(user_id),
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_expire_minutes),
    }

    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def decode_access_token(token: str) -> UUID:
    """Raise jwt.InvalidTokenError (incl. expired) on any invalid token."""
    payload: dict[str, Any] = jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=[settings.jwt_algorithm],
    )
    if payload.get("type") != "access":
        raise jwt.InvalidTokenError("Not an access token")
    return UUID(payload["sub"])
