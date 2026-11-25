from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.models import CitationORM, ConversationORM, MessageORM
from app.models.message import MessageRole
from app.schemas.chat import ConversationCreate
from app.services.conversation_service import ConversationService

OWNER_ID = uuid4()
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def make_service(conversation: ConversationORM | None = None) -> ConversationService:
    service = ConversationService(AsyncMock())
    service.conversations = AsyncMock()
    service.conversations.get_by_id.return_value = conversation
    service.messages = AsyncMock()
    return service


def make_conversation() -> ConversationORM:
    return ConversationORM(
        id=uuid4(), owner_id=OWNER_ID, title="Диалог", created_at=NOW
    )


def make_message(role: MessageRole, content: str) -> MessageORM:
    return MessageORM(id=uuid4(), role=role, content=content, created_at=NOW)


def make_citation(
    message: MessageORM, number: int, url: str | None = None
) -> CitationORM:
    return CitationORM(
        message_id=message.id,
        number=number,
        document_title="Руководство",
        heading_path="Установка > Linux",
        content=f"текст источника {number}",
        document_id=uuid4() if url is None else None,
        chunk_id=None,
        url=url,
    )


async def test_create_saves_conversation_with_given_title() -> None:
    service = make_service()

    conversation = await service.create(
        OWNER_ID, ConversationCreate(title="Мой проект")
    )

    assert (conversation.owner_id, conversation.title) == (OWNER_ID, "Мой проект")
    service.conversations.create.assert_awaited_once()
    service.session.commit.assert_awaited_once()


async def test_unknown_or_foreign_conversation_is_404_for_every_operation() -> None:
    service = make_service(conversation=None)

    for call in (service.get_detail, service.delete, service.get):
        with pytest.raises(HTTPException) as exc_info:
            await call(OWNER_ID, uuid4())
        assert exc_info.value.status_code == 404
    service.conversations.delete.assert_not_awaited()
    service.session.commit.assert_not_awaited()


async def test_detail_groups_citations_under_their_messages_in_order() -> None:
    conversation = make_conversation()
    question = make_message(MessageRole.USER, "Как установить?")
    answer = make_message(MessageRole.ASSISTANT, "Так [1][2].")
    service = make_service(conversation)
    service.messages.list_for_conversation.return_value = [question, answer]
    service.messages.citations_for_messages.return_value = [
        make_citation(answer, 1),
        make_citation(answer, 2),
    ]

    detail = await service.get_detail(OWNER_ID, conversation.id)

    assert [m.role for m in detail.messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert detail.messages[0].citations == []
    assert [c.number for c in detail.messages[1].citations] == [1, 2]
    assert detail.messages[1].citations[0].heading_path == "Установка > Linux"
    # цитаты всех сообщений запрошены одним обращением, а не по одному на сообщение
    service.messages.citations_for_messages.assert_awaited_once_with(
        [question.id, answer.id]
    )


async def test_detail_of_empty_conversation_has_no_messages() -> None:
    conversation = make_conversation()
    service = make_service(conversation)
    service.messages.list_for_conversation.return_value = []
    service.messages.citations_for_messages.return_value = []

    detail = await service.get_detail(OWNER_ID, conversation.id)

    assert detail.messages == []
    assert detail.title == "Диалог"


async def test_delete_removes_own_conversation_and_commits() -> None:
    conversation = make_conversation()
    service = make_service(conversation)

    await service.delete(OWNER_ID, conversation.id)

    service.conversations.delete.assert_awaited_once_with(conversation)
    service.session.commit.assert_awaited_once()


async def test_detail_shows_url_only_for_web_citations() -> None:
    conversation = make_conversation()
    answer = make_message(MessageRole.ASSISTANT, "Так [1][2].")
    service = make_service(conversation)
    service.messages.list_for_conversation.return_value = [answer]
    service.messages.citations_for_messages.return_value = [
        make_citation(answer, 1),
        make_citation(answer, 2, url="https://fastapi.tiangolo.com/tutorial/"),
    ]

    detail = await service.get_detail(OWNER_ID, conversation.id)

    own, web = detail.messages[0].citations
    assert own.url is None and own.document_id is not None
    assert (
        web.url == "https://fastapi.tiangolo.com/tutorial/" and web.document_id is None
    )
