from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, ForeignKey, Index, Integer, String, Text, Uuid
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base

# Язык полнотекстового поиска. Им считается колонка search_vector, и тем же языком
# обязаны разбираться вопросы пользователей: иначе формы слов не совпадут
FTS_CONFIG = "russian"


class ChunkORM(Base):
    __tablename__ = "chunks"
    # Без этого после каждого INSERT SQLAlchemy запрашивает у Postgres значение
    # вычисляемой search_vector обратно (RETURNING), а нам оно в Python не нужно
    __mapper_args__ = {"eager_defaults": False}
    # Индексы объявлены и в модели, и в миграции: иначе автогенерация Alembic
    # сочтёт их «лишними» и предложит удалить
    __table_args__ = (
        # HNSW: поиск ближайших векторов по косинусной близости
        Index(
            "ix_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # GIN: полнотекстовый поиск по search_vector
        Index("ix_chunks_search_vector", "search_vector", postgresql_using="gin"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    heading_path: Mapped[str | None] = mapped_column(String)
    token_count: Mapped[int | None] = mapped_column(Integer)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536))
    # Лексемы текста для полнотекстового поиска. Postgres сам пересчитывает колонку
    # при каждой записи content, поэтому в коде её никогда не заполняем. Язык
    # FTS_CONFIG (russian) сводит формы слова к основе («индексации» и
    # «индексация» совпадут). deferred: в Python колонка не нужна, по ней ищет
    # только база, и тянуть её с каждым чанком незачем
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(f"to_tsvector('{FTS_CONFIG}', content)", persisted=True),
        deferred=True,
    )
