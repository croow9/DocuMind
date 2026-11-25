from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, Integer, String, Text, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class CitationORM(Base):
    """Источник, на который ответ сослался как на [number].

    Название, путь по заголовкам и текст фрагмента хранятся копией: при
    переиндексации документа чанки удаляются и создаются заново с новыми id,
    а история диалога должна остаться читаемой. Ссылки document_id и chunk_id
    нужны лишь как «перейти к источнику, если он ещё жив», поэтому при удалении
    обнуляются, а не тянут за собой цитату.
    """

    __tablename__ = "citations"
    # номер источника уникален внутри ответа; индекс этого ограничения заодно
    # обслуживает выборку цитат по message_id
    __table_args__ = (
        UniqueConstraint("message_id", "number", name="uq_citations_message_number"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    message_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("messages.id", ondelete="CASCADE")
    )
    number: Mapped[int] = mapped_column(Integer)
    # копия данных источника на момент ответа
    document_title: Mapped[str] = mapped_column(String(512))
    heading_path: Mapped[str | None] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    # индексы нужны, чтобы SET NULL при удалении документа или чанков (а при
    # переиндексации удаляются все чанки документа) не сканировал всю таблицу
    document_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("documents.id", ondelete="SET NULL"), index=True
    )
    chunk_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("chunks.id", ondelete="SET NULL"), index=True
    )
