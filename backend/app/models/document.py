from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class DocimentORM(Base):
    __tablename__ = "document"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    # owner_id: Mapped[] = mapped_column()
    title: Mapped[str] = mapped_column(String(512))
    # source_type: Mapped[] = mapped_column()
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    original_filename: Mapped[str | None] = mapped_column(String, nullable=True)
    # status: Mapped[] = mapped_column()
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, timezone=True, server_default=func.now()
    )
    # update_at: Mapped[] = mapped_column()
