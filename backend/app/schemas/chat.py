from datetime import datetime
from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.message import MessageRole

# Каждый вопрос стоит денег (эмбеддинг + запрос к LLM), поэтому длина ограничена
MAX_QUESTION_LENGTH = 2000
MAX_DOCUMENT_IDS = 20

# Пока заголовок не задан, у диалога он такой; по первому вопросу его заменят
DEFAULT_CONVERSATION_TITLE = "Новый диалог"
MAX_TITLE_LENGTH = 255


class MessageCreate(BaseModel):
    """Вопрос пользователя. Список document_ids сужает поиск до этих документов."""

    # лишние поля отклоняем, а пробелы по краям срезаем до проверки длины
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    # None (поле не передано) значит «искать по всем документам»; пустой список
    # запрещён, чтобы «ничего не выбрано» не путалось с «выбрано всё»
    document_ids: list[UUID] | None = Field(
        default=None, min_length=1, max_length=MAX_DOCUMENT_IDS
    )
    # True: искать не в своих документах, а в официальной документации из
    # интернета. Платная операция, поэтому включается только явно
    web_search: bool = False
    # Набор сайтов для поиска в интернете (список: GET /web-profiles); без него
    # берётся профиль по умолчанию
    web_profile: str | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def _web_search_excludes_document_ids(self) -> Self:
        if self.web_search and self.document_ids is not None:
            raise ValueError("web_search нельзя сочетать с document_ids")
        return self

    @model_validator(mode="after")
    def _web_profile_needs_web_search(self) -> Self:
        if self.web_profile is not None and not self.web_search:
            raise ValueError("web_profile имеет смысл только вместе с web_search")
        return self


# --- данные событий потока (SSE)


class SourceRead(BaseModel):
    """Источник, который увидела модель: номер в промпте и откуда он взят."""

    number: int
    chunk_id: UUID | None
    document_id: UUID | None
    document_title: str
    heading_path: str | None
    content: str
    url: str | None = None


class TokenData(BaseModel):
    text: str


class DoneData(BaseModel):
    message_id: UUID
    # номера источников, на которые ответ действительно сослался
    citations: list[int]


class ErrorData(BaseModel):
    model_config = ConfigDict(frozen=True)

    detail: str


# --- профили сайтов для поиска в интернете


class WebProfileRead(BaseModel):
    name: str
    title: str
    description: str
    # сами домены не отдаём: клиенту нужно выбрать направление, а не читать список
    domains_count: int
    is_default: bool


# --- диалоги и их история


class ConversationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(
        default=DEFAULT_CONVERSATION_TITLE, min_length=1, max_length=MAX_TITLE_LENGTH
    )


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    created_at: datetime


class CitationRead(BaseModel):
    """Цитата: копия источника на момент ответа и (если жив) ссылка на него."""

    model_config = ConfigDict(from_attributes=True)

    number: int
    document_title: str
    heading_path: str | None
    content: str
    document_id: UUID | None
    chunk_id: UUID | None
    url: str | None


class MessageRead(BaseModel):
    id: UUID
    role: MessageRole
    content: str
    created_at: datetime
    citations: list[CitationRead]


class ConversationDetail(ConversationRead):
    messages: list[MessageRead]
