from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ConversationORM


class ConversationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(
        self, conversation_id: UUID, owner_id: UUID
    ) -> ConversationORM | None:
        result = await self.session.execute(
            select(ConversationORM).where(
                ConversationORM.id == conversation_id,
                ConversationORM.owner_id == owner_id,
            )
        )
        return result.scalar_one_or_none()
