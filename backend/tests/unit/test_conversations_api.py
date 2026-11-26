from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from starlette.testclient import TestClient

from app.api.deps import get_cache, get_current_user, get_web_profiles
from app.api.rate_limit import CHAT_PER_MINUTE, WEB_PER_DAY, WEB_PER_MINUTE
from app.api.v1.conversations import get_chat_service
from app.core.cache import InMemoryCache
from app.main import app
from app.rag.llm import FakeChat
from app.rag.prompt import NO_WEB_ANSWER_TEXT
from app.rag.web_profiles import WebProfile, WebProfiles
from app.services.chat_service import ChatService

MESSAGE_ID = uuid4()

PROFILES = WebProfiles(
    [
        WebProfile(name="backend", title="Backend", domains=("docs.python.org",)),
        WebProfile(
            name="analyst",
            title="Аналитик",
            description="BPMN и спецификации",
            domains=("docs.camunda.io", "omg.org"),
        ),
    ],
    default="backend",
)


def make_service(conversation_exists: bool) -> ChatService:
    service = ChatService(
        AsyncMock(), AsyncMock(), FakeChat(reply="Ответ [1]."), AsyncMock(), PROFILES
    )
    service.conversations = AsyncMock()
    service.conversations.get_by_id.return_value = (
        AsyncMock(id=uuid4()) if conversation_exists else None
    )
    service.messages = AsyncMock()
    service.messages.last_messages.return_value = []
    service.search.search.return_value = []
    service.web.search.return_value = []
    service._save_answer = AsyncMock(return_value=MESSAGE_ID)  # type: ignore[method-assign]
    return service


@pytest.fixture
def client_for() -> Iterator[object]:
    def build(service: ChatService, cache: InMemoryCache | None = None) -> TestClient:
        # один пользователь на клиента: счётчики ограничения привязаны к нему
        user = SimpleNamespace(id=uuid4())
        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[get_chat_service] = lambda: service
        app.dependency_overrides[get_web_profiles] = lambda: PROFILES
        # без lifespan app.state.cache нет: ограничителю подставляем кэш в памяти
        shared_cache = cache or InMemoryCache()  # один на клиента, не на запрос
        app.dependency_overrides[get_cache] = lambda: shared_cache
        return TestClient(app)

    yield build
    app.dependency_overrides.clear()


def test_unknown_conversation_gives_real_404_not_a_stream(client_for) -> None:  # type: ignore[no-untyped-def]
    client = client_for(make_service(conversation_exists=False))

    response = client.post(
        f"/api/v1/conversations/{uuid4()}/messages", json={"question": "как?"}
    )

    # внутри потока HTTPException дал бы 200 и пустой поток: проверки идут раньше
    assert response.status_code == 404
    assert response.json() == {"detail": "Диалог не найден"}
    assert response.headers["content-type"].startswith("application/json")


def test_valid_question_is_answered_with_an_event_stream(client_for) -> None:  # type: ignore[no-untyped-def]
    client = client_for(make_service(conversation_exists=True))

    response = client.post(
        f"/api/v1/conversations/{uuid4()}/messages", json={"question": "как?"}
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    lines = response.text.splitlines()
    events = [
        line.removeprefix("event: ") for line in lines if line.startswith("event:")
    ]
    assert events[0] == "sources"
    assert events[-1] == "done"
    assert f'"message_id":"{MESSAGE_ID}"' in response.text


@pytest.mark.parametrize(
    "body",
    [
        {"question": "   "},
        {"question": "x" * 2001},
        {"question": "как?", "document_ids": []},
        {"question": "как?", "лишнее": 1},
    ],
)
def test_invalid_question_is_rejected_before_any_work(client_for, body: dict) -> None:  # type: ignore[no-untyped-def,type-arg]
    service = make_service(conversation_exists=True)
    client = client_for(service)

    response = client.post(f"/api/v1/conversations/{uuid4()}/messages", json=body)

    assert response.status_code == 422
    service.messages.create.assert_not_awaited()


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "offset=-1", "limit=abc"])
def test_list_rejects_bad_pagination(client_for, query: str) -> None:  # type: ignore[no-untyped-def]
    client = client_for(make_service(conversation_exists=True))

    assert client.get(f"/api/v1/conversations?{query}").status_code == 422


@pytest.mark.parametrize(
    "body", [{"title": ""}, {"title": "   "}, {"title": "x" * 256}, {"name": "x"}]
)
def test_create_rejects_invalid_title(client_for, body: dict) -> None:  # type: ignore[no-untyped-def,type-arg]
    client = client_for(make_service(conversation_exists=True))

    assert client.post("/api/v1/conversations", json=body).status_code == 422


def test_web_search_cannot_be_combined_with_document_ids(client_for) -> None:  # type: ignore[no-untyped-def]
    service = make_service(conversation_exists=True)
    client = client_for(service)

    response = client.post(
        f"/api/v1/conversations/{uuid4()}/messages",
        json={"question": "как?", "web_search": True, "document_ids": [str(uuid4())]},
    )

    assert response.status_code == 422
    service.messages.create.assert_not_awaited()


def test_web_search_question_is_streamed_from_web_not_from_documents(
    client_for,
) -> None:  # type: ignore[no-untyped-def]
    service = make_service(conversation_exists=True)
    client = client_for(service)

    response = client.post(
        f"/api/v1/conversations/{uuid4()}/messages",
        json={"question": "как?", "web_search": True},
    )

    assert response.status_code == 200
    assert NO_WEB_ANSWER_TEXT in response.text
    # по HTTP клиент видит этапы как события потока
    assert "event: status" in response.text
    assert response.text.index("event: status") < response.text.index("event: sources")
    service.search.search.assert_not_awaited()


def test_unknown_web_profile_gives_real_404_not_a_stream(client_for) -> None:  # type: ignore[no-untyped-def]
    service = make_service(conversation_exists=True)
    client = client_for(service)

    response = client.post(
        f"/api/v1/conversations/{uuid4()}/messages",
        json={"question": "как?", "web_search": True, "web_profile": "nope"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Профиль не найден"}
    service.messages.create.assert_not_awaited()


def test_web_profile_without_web_search_is_rejected(client_for) -> None:  # type: ignore[no-untyped-def]
    client = client_for(make_service(conversation_exists=True))

    response = client.post(
        f"/api/v1/conversations/{uuid4()}/messages",
        json={"question": "как?", "web_profile": "analyst"},
    )

    assert response.status_code == 422


def test_profiles_endpoint_lists_choices_without_the_domains(client_for) -> None:  # type: ignore[no-untyped-def]
    client = client_for(make_service(conversation_exists=True))

    response = client.get("/api/v1/web-profiles")

    assert response.status_code == 200
    assert response.json() == [
        {
            "name": "backend",
            "title": "Backend",
            "description": "",
            "domains_count": 1,
            "is_default": True,
        },
        {
            "name": "analyst",
            "title": "Аналитик",
            "description": "BPMN и спецификации",
            "domains_count": 2,
            "is_default": False,
        },
    ]


def test_personal_list_of_someone_else_gives_real_404_not_a_stream(client_for) -> None:  # type: ignore[no-untyped-def]
    service = make_service(conversation_exists=True)
    service.user_profiles = AsyncMock()
    service.user_profiles.get_by_id.return_value = None
    client = client_for(service)

    response = client.post(
        f"/api/v1/conversations/{uuid4()}/messages",
        json={
            "question": "как?",
            "web_search": True,
            "my_web_profile_id": str(uuid4()),
        },
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Список сайтов не найден"}
    service.messages.create.assert_not_awaited()


@pytest.mark.parametrize(
    "extra",
    [
        {},  # без web_search
        {"web_search": True, "web_profile": "analyst"},  # вместе с профилем из файла
        {
            "web_search": True,
            "document_ids": [str(uuid4())],
        },  # вместе со своими документами
    ],
)
def test_personal_list_with_conflicting_fields_is_rejected(client_for, extra) -> None:  # type: ignore[no-untyped-def]
    client = client_for(make_service(conversation_exists=True))

    response = client.post(
        f"/api/v1/conversations/{uuid4()}/messages",
        json={"question": "как?", "my_web_profile_id": str(uuid4()), **extra},
    )

    assert response.status_code == 422


# --- ограничение частоты: сверх лимита 429 до сохранения вопроса


def ask(client, **body):  # type: ignore[no-untyped-def]
    return client.post(
        f"/api/v1/conversations/{uuid4()}/messages", json={"question": "как?", **body}
    )


def test_question_over_the_limit_is_429_and_is_not_saved(
    client_for, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(CHAT_PER_MINUTE, "limit", 2)
    service = make_service(conversation_exists=True)
    client = client_for(service)

    statuses = [ask(client).status_code for _ in range(3)]
    rejected = ask(client)

    assert statuses == [200, 200, 429]
    assert rejected.status_code == 429
    # настоящий ответ с JSON, а не поток; в заголовке сколько ждать
    assert rejected.headers["content-type"].startswith("application/json")
    assert rejected.headers["retry-after"] == "60"
    assert "Слишком много запросов" in rejected.json()["detail"]
    # вопросы сверх лимита не попали в историю диалога
    assert service.messages.create.await_count == 2


def test_web_questions_have_their_own_stricter_limit(client_for, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(WEB_PER_MINUTE, "limit", 1)
    service = make_service(conversation_exists=True)
    client = client_for(service)

    first = ask(client, web_search=True)
    second = ask(client, web_search=True)
    # обычные вопросы этого пользователя лимит интернета не трогает
    usual = ask(client)

    assert (first.status_code, second.status_code, usual.status_code) == (200, 429, 200)
    assert "по интернету" in second.json()["detail"]
    service.web.search.assert_awaited_once()


def test_daily_web_limit_says_how_long_to_wait_in_hours(
    client_for, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(WEB_PER_DAY, "limit", 1)
    client = client_for(make_service(conversation_exists=True))

    ask(client, web_search=True)
    rejected = ask(client, web_search=True)

    assert rejected.status_code == 429
    assert rejected.headers["retry-after"] == str(24 * 60 * 60)
    assert "24 ч" in rejected.json()["detail"]


def test_limit_does_not_hide_a_missing_conversation_from_other_users(
    client_for,
) -> None:  # type: ignore[no-untyped-def]
    # в пределах лимита проверки диалога работают как прежде
    client = client_for(make_service(conversation_exists=False))

    assert ask(client).status_code == 404
