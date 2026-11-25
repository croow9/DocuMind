from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from fastapi.sse import ServerSentEvent

from app.models import MessageORM
from app.models.message import MessageRole
from app.rag.llm import ChatClient, ChatMessage, FakeChat
from app.rag.prompt import NO_ANSWER_TEXT
from app.schemas.chat import MessageCreate
from app.schemas.search import RetrievedChunk
from app.services.chat_service import ChatService

OWNER_ID = uuid4()
CONVERSATION_ID = uuid4()
MESSAGE_ID = uuid4()


def make_chunk(content: str = "текст источника") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_title="Руководство",
        heading_path=None,
        content=content,
    )


def make_service(
    chat: ChatClient, sources: list[RetrievedChunk]
) -> tuple[ChatService, AsyncMock, AsyncMock]:
    """Сервис с подставными поиском, сессией и сохранением ответа."""
    session = AsyncMock()
    search = AsyncMock()
    search.search.return_value = sources
    service = ChatService(session, search, chat)
    save = AsyncMock(return_value=MESSAGE_ID)
    service._save_answer = save  # type: ignore[method-assign]
    return service, session, save


def answer_events(service: ChatService) -> AsyncGenerator[ServerSentEvent]:
    return service.answer(OWNER_ID, CONVERSATION_ID, MessageCreate(question="как?"), [])


async def collect(service: ChatService) -> list[ServerSentEvent]:
    return [event async for event in answer_events(service)]


class BrokenChat(ChatClient):
    """Модель, которая отдаёт начало ответа и падает."""

    async def stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str]:
        yield "Начало ответа "
        raise RuntimeError("OpenAI недоступен")


class TrackingChat(ChatClient):
    """Модель, которая запоминает, закрыли ли её поток."""

    def __init__(self) -> None:
        self.closed = False

    async def stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str]:
        try:
            for piece in ("раз ", "два ", "три "):
                yield piece
        finally:
            self.closed = True


# --- answer: поток событий


async def test_answer_streams_sources_then_tokens_then_done() -> None:
    chunk = make_chunk()
    chat = FakeChat(reply="Ответ по источнику [1].")
    service, _, save = make_service(chat, [chunk])

    events = await collect(service)

    names = [e.event for e in events]
    assert names[0] == "sources"
    assert names[-1] == "done"
    assert set(names[1:-1]) == {"token"}
    assert events[0].data[0].number == 1
    assert events[0].data[0].chunk_id == chunk.chunk_id
    assert "".join(e.data.text for e in events[1:-1]) == "Ответ по источнику [1]."
    assert events[-1].data.message_id == MESSAGE_ID
    assert events[-1].data.citations == [1]
    save.assert_awaited_once_with(
        CONVERSATION_ID, "Ответ по источнику [1].", [chunk], [1]
    )


async def test_db_session_is_released_before_the_first_event() -> None:
    service, session, _ = make_service(FakeChat(), [make_chunk()])
    events = answer_events(service)

    first = await anext(events)
    # поток только начался, а соединение с БД уже отдано: дальше ждём только LLM
    session.close.assert_awaited_once()
    assert first.event == "sources"
    await events.aclose()


async def test_no_sources_means_fixed_answer_without_calling_llm() -> None:
    chat = FakeChat()
    service, _, save = make_service(chat, [])

    events = await collect(service)

    assert [e.event for e in events] == ["sources", "token", "done"]
    assert events[0].data == []
    assert events[1].data.text == NO_ANSWER_TEXT
    assert chat.received == []
    save.assert_awaited_once_with(CONVERSATION_ID, NO_ANSWER_TEXT, [], [])


async def test_citation_numbers_invented_by_model_are_dropped() -> None:
    chat = FakeChat(reply="Факт [1], выдумка [7].")
    service, _, save = make_service(chat, [make_chunk()])

    events = await collect(service)

    assert events[-1].data.citations == [1]
    assert save.await_args.args[3] == [1]


# --- answer: сбои и обрыв


async def test_llm_failure_becomes_error_event_and_nothing_is_saved() -> None:
    service, _, save = make_service(BrokenChat(), [make_chunk()])

    events = await collect(service)

    assert [e.event for e in events] == ["sources", "token", "error"]
    assert "OpenAI" not in events[-1].data.detail  # внутренности клиенту не отдаём
    save.assert_not_awaited()


async def test_search_failure_becomes_error_event() -> None:
    chat = FakeChat()
    service, _, save = make_service(chat, [])
    service.search.search.side_effect = RuntimeError("эмбеддинги недоступны")

    events = await collect(service)

    assert [e.event for e in events] == ["error"]
    assert chat.received == []
    save.assert_not_awaited()


async def test_empty_model_answer_is_an_error_and_is_not_saved() -> None:
    service, _, save = make_service(FakeChat(reply="   "), [make_chunk()])

    events = await collect(service)

    assert events[-1].event == "error"
    save.assert_not_awaited()


async def test_client_disconnect_closes_llm_stream_and_saves_nothing() -> None:
    chat = TrackingChat()
    service, _, save = make_service(chat, [make_chunk()])
    events = answer_events(service)

    assert (await anext(events)).event == "sources"
    assert (await anext(events)).event == "token"
    await events.aclose()  # так генератор закрывается, когда клиент ушёл

    assert chat.closed
    save.assert_not_awaited()


# --- prepare: проверки до начала потока


def make_prepare_service(conversation_exists: bool = True) -> ChatService:
    service = ChatService(AsyncMock(), AsyncMock(), FakeChat())
    service.conversations = AsyncMock()
    service.conversations.get_by_id.return_value = (
        AsyncMock(id=CONVERSATION_ID) if conversation_exists else None
    )
    service.documents = AsyncMock()
    service.messages = AsyncMock()
    service.messages.last_messages.return_value = []
    return service


async def test_prepare_unknown_conversation_is_404_and_saves_nothing() -> None:
    service = make_prepare_service(conversation_exists=False)

    with pytest.raises(HTTPException) as exc_info:
        await service.prepare(OWNER_ID, CONVERSATION_ID, MessageCreate(question="как?"))

    assert exc_info.value.status_code == 404
    service.messages.create.assert_not_awaited()
    service.session.commit.assert_not_awaited()


async def test_prepare_foreign_document_is_404_before_saving_question() -> None:
    service = make_prepare_service()
    foreign: UUID = uuid4()
    service.documents.get_owned_ids.return_value = set()

    with pytest.raises(HTTPException) as exc_info:
        await service.prepare(
            OWNER_ID,
            CONVERSATION_ID,
            MessageCreate(question="как?", document_ids=[foreign]),
        )

    assert exc_info.value.status_code == 404
    service.messages.create.assert_not_awaited()


async def test_prepare_returns_history_read_before_the_question_is_saved() -> None:
    service = make_prepare_service()
    service.messages.last_messages.return_value = [
        MessageORM(role=MessageRole.USER, content="Что такое воркер?"),
        MessageORM(role=MessageRole.ASSISTANT, content="Процесс [1]."),
    ]

    history = await service.prepare(
        OWNER_ID, CONVERSATION_ID, MessageCreate(question="  а подробнее?  ")
    )

    # история не содержит нового вопроса: он сохраняется уже после чтения
    assert history == [
        ChatMessage(role="user", content="Что такое воркер?"),
        ChatMessage(role="assistant", content="Процесс [1]."),
    ]
    assert [call[0] for call in service.messages.mock_calls] == [
        "last_messages",
        "create",
    ]
    # пробелы по краям вопроса срезаны схемой
    service.messages.create.assert_awaited_once_with(
        CONVERSATION_ID, MessageRole.USER, "а подробнее?"
    )
    service.session.commit.assert_awaited_once()
