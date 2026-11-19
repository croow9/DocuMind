from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from app.models.document import DocumentStatus, SourceType


class DocumentCreate(BaseModel):
    """Документ по ссылке. Файлы загружаются отдельным эндпоинтом /upload."""

    # лишние поля (например, старый source_type) отклоняем, а не молча игнорируем
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=512)
    source_url: HttpUrl


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    source_type: SourceType
    source_url: str | None
    original_filename: str | None
    status: DocumentStatus
    chunk_count: int
    error: str | None
    created_at: datetime
