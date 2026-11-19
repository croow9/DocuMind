from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class IngestionRequested(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    event_id: UUID = Field(
        default_factory=uuid4,
    )
    document_id: UUID = Field(
        ...,
    )
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DeadLetter(BaseModel):
    """Сообщение, которое не удалось обработать: оригинал и причина отказа."""

    model_config = ConfigDict(frozen=True)

    source_topic: str
    # оригинал как текст: мусорные байты заменяются, чтобы письмо точно отправилось
    original_value: str
    error: str
    failed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
