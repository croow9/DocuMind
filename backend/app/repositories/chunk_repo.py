from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ChunkORM


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
