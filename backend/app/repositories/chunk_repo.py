from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import Select, cast, delete, func, select
from sqlalchemy.dialects.postgresql import REGCONFIG
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ChunkORM, DocumentORM
from app.models.chunk import FTS_CONFIG
from app.models.document import DocumentStatus
from app.schemas.search import RetrievedChunk


def _searchable_chunks(
    owner_id: UUID, document_ids: Sequence[UUID] | None
) -> Select[UUID, UUID, str, str | None, str]:
    """Основа обоих поисков: чанки, которые этот пользователь вправе увидеть.

    В chunks нет owner_id, владельца знает только документ, поэтому нужен JOIN.
    Фильтр по владельцу и статусу живёт ровно здесь, в одном месте: ни один
    поиск не обходит его, а ошибка в нём означала бы утечку чужих документов.
    """
    query = (
        select(
            ChunkORM.id.label("chunk_id"),
            ChunkORM.document_id,
            DocumentORM.title.label("document_title"),
            ChunkORM.heading_path,
            ChunkORM.content,
        )
        .join(DocumentORM, DocumentORM.id == ChunkORM.document_id)
        .where(
            DocumentORM.owner_id == owner_id,
            DocumentORM.status == DocumentStatus.READY,
        )
    )
    # None значит «все документы пользователя», а список (даже пустой) сужает
    # поиск. Чужой id в списке ничего не найдёт: владелец проверен выше
    if document_ids is not None:
        query = query.where(ChunkORM.document_id.in_(document_ids))
    return query


class ChunkRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_all(self, chunks: Sequence[ChunkORM]) -> None:
        self.session.add_all(chunks)
        await self.session.flush()

    async def delete_by_document_id(self, document_id: UUID) -> None:
        await self.session.execute(
            delete(ChunkORM).where(ChunkORM.document_id == document_id)
        )

    async def search_by_vector(
        self,
        query_vector: list[float],
        owner_id: UUID,
        document_ids: Sequence[UUID] | None,
        limit: int,
    ) -> list[RetrievedChunk]:
        """Поиск по смыслу: чанки, чьи векторы ближе всего к вектору вопроса."""
        query = (
            _searchable_chunks(owner_id, document_ids)
            # <=> в SQL: косинусное расстояние, 0 значит «то же направление»
            .order_by(ChunkORM.embedding.cosine_distance(query_vector))
            .limit(limit)
        )
        result = await self.session.execute(query)
        return [RetrievedChunk.model_validate(row) for row in result]

    async def search_by_text(
        self,
        question: str,
        owner_id: UUID,
        document_ids: Sequence[UUID] | None,
        limit: int,
    ) -> list[RetrievedChunk]:
        """Поиск по словам: чанки, в которых есть слова вопроса (по основам)."""
        # websearch_to_tsquery не падает на кавычках и скобках в вопросе,
        # в отличие от to_tsquery; слова он соединяет через И
        ts_query = func.websearch_to_tsquery(cast(FTS_CONFIG, REGCONFIG), question)
        query = (
            _searchable_chunks(owner_id, document_ids)
            # @@ «вектор текста удовлетворяет запросу»; идёт по GIN-индексу
            .where(ChunkORM.search_vector.bool_op("@@")(ts_query))
            # ts_rank_cd учитывает, насколько близко друг к другу стоят слова
            .order_by(func.ts_rank_cd(ChunkORM.search_vector, ts_query).desc())
            .limit(limit)
        )
        result = await self.session.execute(query)
        return [RetrievedChunk.model_validate(row) for row in result]
