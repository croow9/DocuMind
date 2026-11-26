from datetime import datetime
from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.rag.web_profiles import (
    MAX_DOMAINS_PER_USER_PROFILE,
    clean_domains,
    normalize_domain,
)

MAX_TITLE_LENGTH = 100
MAX_DESCRIPTION_LENGTH = 500


def _check_domains(raw: list[str]) -> list[str]:
    """Домены из ввода пользователя: адреса сводим к доменам, повторы убираем.

    Пользователь вставляет то, что видит в адресной строке, поэтому «a.com» и
    «https://a.com/docs» для него одно и то же, и ругаться на повтор не за что.
    """
    unique = dict.fromkeys(normalize_domain(item) for item in raw)
    return list(clean_domains(unique, MAX_DOMAINS_PER_USER_PROFILE))


class UserWebProfileCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=MAX_TITLE_LENGTH)
    description: str = Field(default="", max_length=MAX_DESCRIPTION_LENGTH)
    # длину ограничивает и clean_domains, но проверка здесь отсекает огромный
    # список до разбора каждого адреса
    domains: list[str] = Field(
        min_length=1, max_length=MAX_DOMAINS_PER_USER_PROFILE * 4
    )

    @field_validator("domains")
    @classmethod
    def _domains(cls, domains: list[str]) -> list[str]:
        return _check_domains(domains)


class UserWebProfileUpdate(BaseModel):
    """Что изменить: непереданное поле остаётся как было, domains заменяет список целиком."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str | None = Field(default=None, min_length=1, max_length=MAX_TITLE_LENGTH)
    description: str | None = Field(default=None, max_length=MAX_DESCRIPTION_LENGTH)
    domains: list[str] | None = Field(
        default=None, min_length=1, max_length=MAX_DOMAINS_PER_USER_PROFILE * 4
    )

    @field_validator("domains")
    @classmethod
    def _domains(cls, domains: list[str] | None) -> list[str] | None:
        return None if domains is None else _check_domains(domains)

    @model_validator(mode="after")
    def _something_to_change(self) -> Self:
        if self.title is None and self.description is None and self.domains is None:
            raise ValueError("нечего менять: передайте title, description или domains")
        return self


class UserWebProfileRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    description: str
    # у личного списка домены показываем: их написал сам пользователь
    domains: list[str]
    created_at: datetime
