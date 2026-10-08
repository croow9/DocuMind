from datetime import datetime
from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from app.models.document import DocumentStatus, SourceType


class DocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=512)
    source_type: SourceType
    source_url: HttpUrl | None = None

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if self.source_type is SourceType.URL and self.source_url is None:
            raise ValueError("source_url is required when source_type is 'url'")
        if self.source_type is SourceType.UPLOAD and self.source_url is not None:
            raise ValueError("source_url must not be set when source_type is 'upload'")
        return self


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
