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
