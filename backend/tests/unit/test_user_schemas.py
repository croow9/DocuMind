import pytest
from pydantic import ValidationError

from app.schemas.user import UserCreate, normalize_email


def test_password_of_72_bytes_is_accepted() -> None:
    user = UserCreate(email="user@example.com", password="a" * 72)

    assert len(user.password.encode("utf-8")) == 72


def test_cyrillic_password_over_72_bytes_is_rejected() -> None:
    with pytest.raises(ValidationError):
        UserCreate(email="user@example.com", password="п" * 40)


def test_email_is_lowercased_and_stripped() -> None:
    user = UserCreate(email=" Alex.Test@Example.COM ", password="12345678")

    assert user.email == "alex.test@example.com"


def test_normalize_email_matches_schema_rule() -> None:
    assert normalize_email(" A@X.com ") == "a@x.com"
