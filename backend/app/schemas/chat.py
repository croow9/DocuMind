from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# Каждый вопрос стоит денег (эмбеддинг + запрос к LLM), поэтому длина ограничена
MAX_QUESTION_LENGTH = 2000
MAX_DOCUMENT_IDS = 20


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


# --- данные событий потока (SSE)


class SourceRead(BaseModel):
    """Источник, который увидела модель: номер в промпте и откуда он взят."""

    number: int
    chunk_id: UUID
    document_id: UUID
    document_title: str
    heading_path: str | None
    content: str


class TokenData(BaseModel):
    text: str


class DoneData(BaseModel):
    message_id: UUID
    # номера источников, на которые ответ действительно сослался
    citations: list[int]


class ErrorData(BaseModel):
    model_config = ConfigDict(frozen=True)

    detail: str
