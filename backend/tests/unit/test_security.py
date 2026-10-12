from datetime import UTC, datetime, timedelta
from uuid import uuid4

import jwt
import pytest

from app.core.config import settings
from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


#  Тесты на пароль, хэш
def test_verify_password_with_correct_password() -> None:
    password = "qwerty0123456"
    hash_pass = hash_password(password)
    assert verify_password(password, hash_pass)


def test_verify_password_with_wrong_password() -> None:
    true_password = "qwerty0123456"
    wrong_password = "falsefalse777"
    hash_true_password = hash_password(true_password)
    assert not verify_password(wrong_password, hash_true_password)


def test_same_password_gives_different_hashes() -> None:
    first_password = "asdfgcch1122"
    second_password = "asdfgcch1122"
    hash_first = hash_password(first_password)
    hash_second = hash_password(second_password)

    assert hash_first != hash_second
    assert verify_password(first_password, hash_first)
    assert verify_password(second_password, hash_second)


#  Тесты на token
def test_token_round_trip() -> None:
    user_id = uuid4()

    token = create_access_token(user_id)

    assert decode_access_token(token) == user_id


def test_expired_token_is_rejected() -> None:
    user_id = uuid4()

    expired_payload = {
        "sub": str(user_id),
        "type": "access",
        "exp": datetime.now(UTC) - timedelta(minutes=5),
    }
    expired_token = jwt.encode(
        expired_payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(jwt.ExpiredSignatureError):
        decode_access_token(expired_token)
