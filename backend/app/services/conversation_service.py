from collections.abc import Sequence
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ConversationORM
from app.repositories.conversation_repo import ConversationRepository
from app.repositories.message_repo import MessageRepository
from app.schemas.chat import (
    CitationRead,
    ConversationCreate,
    ConversationDetail,
    MessageRead,
)


class ConversationService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.conversations = ConversationRepository(session)
        self.messages = MessageRepository(session)

    async def create(
        self, owner_id: UUID, payload: ConversationCreate
    ) -> ConversationORM:
        conversation = ConversationORM(owner_id=owner_id, title=payload.title)
        await self.conversations.create(conversation)
        await self.session.commit()
        return conversation

    async def list_conversations(
        self, owner_id: UUID, limit: int = 20, offset: int = 0
    ) -> Sequence[ConversationORM]:
        return await self.conversations.list_for_owner(
            owner_id, limit=limit, offset=offset
        )

    async def get(self, owner_id: UUID, conversation_id: UUID) -> ConversationORM:
        conversation = await self.conversations.get_by_id(conversation_id, owner_id)
        if conversation is None:
            # чужой диалог неотличим от несуществующего: не раскрываем его наличие
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Диалог не найден"
            )
        return conversation

    async def get_detail(
        self, owner_id: UUID, conversation_id: UUID
    ) -> ConversationDetail:
        """Диалог целиком: сообщения по порядку, у ответов их цитаты."""
        conversation = await self.get(owner_id, conversation_id)
        messages = await self.messages.list_for_conversation(conversation.id)
        # цитаты всех сообщений одним запросом, а не по запросу на каждое сообщение
        citations = await self.messages.citations_for_messages(
            [message.id for message in messages]
        )

        by_message: dict[UUID, list[CitationRead]] = {}
        for citation in citations:
            by_message.setdefault(citation.message_id, []).append(
                CitationRead.model_validate(citation)
            )

        return ConversationDetail(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at,
            messages=[
                MessageRead(
                    id=message.id,
                    role=message.role,
                    content=message.content,
                    created_at=message.created_at,
                    citations=by_message.get(message.id, []),
                )
                for message in messages
            ],
        )

    async def delete(self, owner_id: UUID, conversation_id: UUID) -> None:
        conversation = await self.get(owner_id, conversation_id)
        await self.conversations.delete(conversation)
        await self.session.commit()
