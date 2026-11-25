from collections.abc import Sequence
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

    async def list_for_owner(
        self, owner_id: UUID, limit: int = 20, offset: int = 0
    ) -> Sequence[ConversationORM]:
        result = await self.session.execute(
            select(ConversationORM)
            .where(ConversationORM.owner_id == owner_id)
            # id добавлен, чтобы порядок при равных датах был одинаковым от
            # запроса к запросу: иначе страницы могли бы пересекаться
            .order_by(ConversationORM.created_at.desc(), ConversationORM.id)
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def create(self, conversation: ConversationORM) -> ConversationORM:
        self.session.add(conversation)
        await self.session.flush()
        return conversation

    async def delete(self, conversation: ConversationORM) -> None:
        # сообщения и цитаты удалит сама база (ON DELETE CASCADE)
        await self.session.delete(conversation)
