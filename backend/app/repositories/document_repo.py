from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DocumentORM


class DocumentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, document_id: UUID, owner_id: UUID) -> DocumentORM | None:
        result = await self.session.execute(
            select(DocumentORM).where(
                DocumentORM.id == document_id,
                DocumentORM.owner_id == owner_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_for_owner(
        self, owner_id: UUID, limit: int = 20, offset: int = 0
    ) -> Sequence[DocumentORM]:
        result = await self.session.execute(
            select(DocumentORM)
            .where(DocumentORM.owner_id == owner_id)
            .order_by(DocumentORM.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def create(self, document: DocumentORM) -> DocumentORM:
        self.session.add(document)
        await self.session.flush()
        return document

    async def delete(self, document: DocumentORM) -> None:
        await self.session.delete(document)
