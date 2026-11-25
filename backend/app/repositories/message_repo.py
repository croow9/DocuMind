from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import CitationORM, MessageORM
from app.models.message import MessageRole


class MessageRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self, conversation_id: UUID, role: MessageRole, content: str
    ) -> MessageORM:
        message = MessageORM(
            conversation_id=conversation_id, role=role, content=content
        )
        self.session.add(message)
        # flush выдаёт сообщению id, он нужен для цитат
        await self.session.flush()
        return message

    async def add_citations(self, citations: Sequence[CitationORM]) -> None:
        self.session.add_all(citations)
        await self.session.flush()

    async def last_messages(
        self, conversation_id: UUID, limit: int
    ) -> list[MessageORM]:
        """Последние limit сообщений диалога, от старых к новым."""
        result = await self.session.execute(
            select(MessageORM)
            .where(MessageORM.conversation_id == conversation_id)
            .order_by(MessageORM.created_at.desc())
            .limit(limit)
        )
        # выбирали от новых к старым, чтобы взять именно последние; в промпт
        # сообщения должны идти в порядке разговора
        return list(reversed(result.scalars().all()))

    async def list_for_conversation(self, conversation_id: UUID) -> list[MessageORM]:
        """Все сообщения диалога от старых к новым."""
        result = await self.session.execute(
            select(MessageORM)
            .where(MessageORM.conversation_id == conversation_id)
            .order_by(MessageORM.created_at)
        )
        return list(result.scalars().all())

    async def citations_for_messages(
        self, message_ids: Sequence[UUID]
    ) -> list[CitationORM]:
        """Цитаты сразу всех сообщений одним запросом (а не по запросу на каждое)."""
        if not message_ids:
            return []
        result = await self.session.execute(
            select(CitationORM)
            .where(CitationORM.message_id.in_(message_ids))
            .order_by(CitationORM.message_id, CitationORM.number)
        )
        return list(result.scalars().all())
