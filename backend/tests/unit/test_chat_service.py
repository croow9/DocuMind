from collections.abc import AsyncGenerator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from fastapi.sse import ServerSentEvent

from app.models import MessageORM
from app.models.message import MessageRole
from app.rag.llm import ChatClient, ChatMessage, FakeChat
from app.rag.prompt import NO_ANSWER_TEXT, NO_WEB_ANSWER_TEXT, WEB_SYSTEM_PROMPT
from app.rag.web_profiles import WebProfile, WebProfiles
from app.schemas.chat import DEFAULT_CONVERSATION_TITLE, MessageCreate
from app.schemas.search import RetrievedChunk
from app.services.chat_service import ChatService, title_from_question

OWNER_ID = uuid4()
CONVERSATION_ID = uuid4()
MESSAGE_ID = uuid4()

BACKEND = WebProfile(name="backend", title="Backend", domains=("fastapi.tiangolo.com",))
ANALYST = WebProfile(name="analyst", title="Аналитик", domains=("docs.camunda.io",))
PROFILES = WebProfiles([BACKEND, ANALYST], default="backend")


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
    service = ChatService(session, search, chat, AsyncMock(), PROFILES)
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


def make_prepare_service(
    conversation_exists: bool = True, title: str = "Свой заголовок"
) -> ChatService:
    service = ChatService(AsyncMock(), AsyncMock(), FakeChat(), AsyncMock(), PROFILES)
    service.conversations = AsyncMock()
    service.conversations.get_by_id.return_value = (
        SimpleNamespace(id=CONVERSATION_ID, title=title)
        if conversation_exists
        else None
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


async def test_prepare_unknown_web_profile_is_404_before_saving_question() -> None:
    service = make_prepare_service()

    with pytest.raises(HTTPException) as exc_info:
        await service.prepare(
            OWNER_ID,
            CONVERSATION_ID,
            MessageCreate(question="как?", web_search=True, web_profile="nope"),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Профиль не найден"
    service.messages.create.assert_not_awaited()


async def test_prepare_accepts_known_and_omitted_web_profile() -> None:
    service = make_prepare_service()

    for name in ("analyst", None):
        await service.prepare(
            OWNER_ID,
            CONVERSATION_ID,
            MessageCreate(question="как?", web_search=True, web_profile=name),
        )

    assert service.messages.create.await_count == 2


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


# --- заголовок диалога по первому вопросу


async def test_first_question_names_a_conversation_with_default_title() -> None:
    service = make_prepare_service(title=DEFAULT_CONVERSATION_TITLE)

    await service.prepare(
        OWNER_ID, CONVERSATION_ID, MessageCreate(question="Как установить   в Linux?")
    )

    conversation = service.conversations.get_by_id.return_value
    assert conversation.title == "Как установить в Linux?"


async def test_custom_title_is_not_overwritten_by_first_question() -> None:
    service = make_prepare_service(title="Мой проект")

    await service.prepare(OWNER_ID, CONVERSATION_ID, MessageCreate(question="как?"))

    assert service.conversations.get_by_id.return_value.title == "Мой проект"


async def test_later_questions_do_not_rename_the_conversation() -> None:
    service = make_prepare_service(title=DEFAULT_CONVERSATION_TITLE)
    service.messages.last_messages.return_value = [
        MessageORM(role=MessageRole.USER, content="раньше уже спрашивали")
    ]

    await service.prepare(OWNER_ID, CONVERSATION_ID, MessageCreate(question="как?"))

    assert (
        service.conversations.get_by_id.return_value.title == DEFAULT_CONVERSATION_TITLE
    )


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Как работает воркер?", "Как работает воркер?"),
        ("  много \n пробелов\tи   переносов ", "много пробелов и переносов"),
        ("а" * 60, "а" * 60),
        ("а" * 61, "а" * 59 + "…"),
        ("слово " * 30, ("слово " * 30)[:59].rstrip() + "…"),
    ],
)
def test_title_from_question(question: str, expected: str) -> None:
    title = title_from_question(question)

    assert title == expected
    assert len(title) <= 60


# --- поиск по документации в интернете

WEB_URL = "https://fastapi.tiangolo.com/ru/tutorial/dependencies/"


def make_web_chunk(content: str = "текст страницы") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=None,
        document_id=None,
        document_title="Зависимости - FastAPI",
        heading_path="Зависимости > Первые шаги",
        content=content,
        url=WEB_URL,
    )


async def collect_web(service: ChatService) -> list[ServerSentEvent]:
    payload = MessageCreate(question="как?", web_search=True)
    return [e async for e in service.answer(OWNER_ID, CONVERSATION_ID, payload, [])]


async def test_web_search_answers_from_web_sources_not_from_own_documents() -> None:
    chunk = make_web_chunk()
    chat = FakeChat(reply="Ответ по странице [1].")
    service, _, save = make_service(chat, [make_chunk()])
    service.web.search.return_value = [chunk]

    events = await collect_web(service)

    service.web.search.assert_awaited_once_with("как?", BACKEND, context=[])
    service.search.search.assert_not_awaited()
    source = events[0].data[0]
    assert (source.url, source.chunk_id, source.document_id) == (WEB_URL, None, None)
    assert events[-1].data.citations == [1]
    save.assert_awaited_once_with(
        CONVERSATION_ID, "Ответ по странице [1].", [chunk], [1]
    )
    # модель получила правила для документации из интернета, а не для своих документов
    assert chat.received[0][0].content == WEB_SYSTEM_PROMPT


async def test_web_search_uses_the_profile_from_the_request() -> None:
    service, _, _ = make_service(FakeChat(), [])
    service.web.search.return_value = []
    payload = MessageCreate(question="как?", web_search=True, web_profile="analyst")

    [event async for event in service.answer(OWNER_ID, CONVERSATION_ID, payload, [])]

    service.web.search.assert_awaited_once_with("как?", ANALYST, context=[])


async def test_own_documents_search_never_touches_the_web() -> None:
    service, _, _ = make_service(FakeChat(), [make_chunk()])

    await collect(service)

    service.web.search.assert_not_awaited()


async def test_web_search_without_pages_gives_web_refusal_without_calling_llm() -> None:
    chat = FakeChat()
    service, _, save = make_service(chat, [make_chunk()])
    service.web.search.return_value = []

    events = await collect_web(service)

    assert [e.event for e in events] == ["sources", "token", "done"]
    assert events[1].data.text == NO_WEB_ANSWER_TEXT
    assert chat.received == []
    save.assert_awaited_once_with(CONVERSATION_ID, NO_WEB_ANSWER_TEXT, [], [])


async def test_web_search_failure_becomes_error_event_and_nothing_is_saved() -> None:
    service, _, save = make_service(FakeChat(), [])
    service.web.search.side_effect = RuntimeError("OpenAI недоступен")

    events = await collect_web(service)

    assert [e.event for e in events] == ["error"]
    save.assert_not_awaited()


# --- сохранение цитат: адрес страницы и ссылки на документ


def make_store_service(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ChatService, AsyncMock]:
    repo = AsyncMock()
    repo.create.return_value = SimpleNamespace(id=MESSAGE_ID)
    monkeypatch.setattr(
        "app.services.chat_service.MessageRepository", lambda session: repo
    )
    factory = MagicMock()
    factory.return_value.__aenter__.return_value = AsyncMock()
    service = ChatService(
        AsyncMock(),
        AsyncMock(),
        FakeChat(),
        AsyncMock(),
        PROFILES,
        session_factory=factory,
    )
    return service, repo


async def test_web_citation_keeps_url_and_copy_of_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, repo = make_store_service(monkeypatch)

    await service._store_answer(
        CONVERSATION_ID, "Ответ [1].", [make_web_chunk()], [1], keep_links=True
    )

    [citation] = repo.add_citations.await_args.args[0]
    assert citation.url == WEB_URL
    assert citation.document_title == "Зависимости - FastAPI"
    assert citation.heading_path == "Зависимости > Первые шаги"
    assert citation.content == "текст страницы"
    assert citation.document_id is None and citation.chunk_id is None


@pytest.mark.parametrize("keep_links", [True, False])
async def test_citation_of_own_document_drops_links_only_when_asked(
    monkeypatch: pytest.MonkeyPatch, keep_links: bool
) -> None:
    service, repo = make_store_service(monkeypatch)
    chunk = make_chunk()

    await service._store_answer(CONVERSATION_ID, "Ответ [1].", [chunk], [1], keep_links)

    [citation] = repo.add_citations.await_args.args[0]
    assert citation.url is None
    assert (citation.document_id, citation.chunk_id) == (
        (chunk.document_id, chunk.chunk_id) if keep_links else (None, None)
    )
    assert citation.content == chunk.content  # копия текста остаётся в любом случае


async def test_web_search_gets_only_users_previous_questions_as_context() -> None:
    service, _, _ = make_service(FakeChat(), [])
    service.web.search.return_value = []
    history = [
        ChatMessage(role="user", content="первый вопрос"),
        ChatMessage(role="user", content="второй вопрос"),
        ChatMessage(role="user", content="третий вопрос"),
        # ответ модели в конце: он попал бы в «два последних», если бы не фильтр по роли
        ChatMessage(role="assistant", content="ответ, который поиску не нужен"),
    ]
    payload = MessageCreate(question="уточнение", web_search=True)

    [
        event
        async for event in service.answer(OWNER_ID, CONVERSATION_ID, payload, history)
    ]

    # реплики модели не берём, и не больше двух последних вопросов
    service.web.search.assert_awaited_once_with(
        "уточнение", BACKEND, context=["второй вопрос", "третий вопрос"]
    )
