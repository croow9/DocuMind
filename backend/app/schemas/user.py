from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    EmailStr,
    Field,
    field_validator,
)


def normalize_email(email: str) -> str:
    return email.strip().lower()


LowerEmail = Annotated[
    EmailStr,
    BeforeValidator(lambda v: normalize_email(v) if isinstance(v, str) else v),
]


class UserCreate(BaseModel):
    email: LowerEmail
    password: str = Field(min_length=8, max_length=72)

    @field_validator("password")
    @classmethod
    def validate_password_bytes(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 72:
            raise ValueError(
                "Password must not exceed 72 bytes (multi-byte characters like Cyrillic take more space)"
            )
        return value


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: EmailStr
    created_at: datetime
