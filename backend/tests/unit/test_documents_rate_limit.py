from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from fastapi import Request
from starlette.testclient import TestClient

from app.api.deps import get_cache, get_current_user, get_producer
from app.api.rate_limit import UPLOAD_PER_MINUTE
from app.core.cache import InMemoryCache
from app.db.session import get_db
from app.main import app

NOW = datetime(2026, 1, 1, tzinfo=UTC)
URL_BODY = {"title": "Руководство", "source_url": "https://docs.example.com/guide"}
FILE = {"file": ("note.txt", b"hello", "text/plain")}


def make_document() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        title="Руководство",
        source_type="url",
        source_url="https://docs.example.com/guide",
        original_filename=None,
        status="pending",
        chunk_count=0,
        error=None,
        created_at=NOW,
    )


class FakeDocumentService:
    """Подставной сервис: считает, сколько раз до него дошёл запрос."""

    created = 0

    def __init__(self, db: object) -> None:
        pass

    async def create_from_url(self, user: object, payload: object) -> SimpleNamespace:
        FakeDocumentService.created += 1
        return make_document()

    async def create_from_file(self, user: object, file: object) -> SimpleNamespace:
        FakeDocumentService.created += 1
        return make_document()

    async def request_indexing(self, document: object, producer: object) -> None:
        pass

    async def list_documents(
        self, user: object, limit: int, offset: int
    ) -> list[object]:
        return []


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    FakeDocumentService.created = 0
    monkeypatch.setattr("app.api.v1.documents.DocumentService", FakeDocumentService)
    monkeypatch.setattr(UPLOAD_PER_MINUTE, "limit", 2)
    cache = InMemoryCache()

    def user_from_header(request: Request) -> SimpleNamespace:
        # пользователь задаётся заголовком, чтобы проверить двух в одном тесте
        return SimpleNamespace(id=UUID(request.headers["x-test-user"]))

    app.dependency_overrides[get_current_user] = user_from_header
    app.dependency_overrides[get_cache] = lambda: cache
    app.dependency_overrides[get_producer] = lambda: AsyncMock()
    app.dependency_overrides[get_db] = lambda: AsyncMock()
    yield TestClient(app)
    app.dependency_overrides.clear()


def as_user(user_id: UUID) -> dict[str, str]:
    return {"x-test-user": str(user_id)}


def test_uploads_over_the_limit_are_429_and_never_reach_the_service(
    client: TestClient,
) -> None:
    me = as_user(uuid4())

    codes = [
        client.post("/api/v1/documents/upload", files=FILE, headers=me).status_code
        for _ in range(3)
    ]
    rejected = client.post("/api/v1/documents/upload", files=FILE, headers=me)

    assert codes == [201, 201, 429]
    assert rejected.headers["retry-after"] == "60"
    assert "загрузка документов" in rejected.json()["detail"]
    assert FakeDocumentService.created == 2


def test_documents_by_link_and_by_file_share_one_counter(client: TestClient) -> None:
    me = as_user(uuid4())

    by_link = client.post("/api/v1/documents", json=URL_BODY, headers=me)
    by_file = client.post("/api/v1/documents/upload", files=FILE, headers=me)
    third = client.post("/api/v1/documents", json=URL_BODY, headers=me)

    assert (by_link.status_code, by_file.status_code, third.status_code) == (
        201,
        201,
        429,
    )


def test_each_user_has_their_own_upload_quota(client: TestClient) -> None:
    first, second = as_user(uuid4()), as_user(uuid4())
    for _ in range(2):
        client.post("/api/v1/documents", json=URL_BODY, headers=first)

    blocked = client.post("/api/v1/documents", json=URL_BODY, headers=first)
    other = client.post("/api/v1/documents", json=URL_BODY, headers=second)

    assert (blocked.status_code, other.status_code) == (429, 201)


def test_reading_documents_is_not_limited(client: TestClient) -> None:
    me = as_user(uuid4())

    codes = [client.get("/api/v1/documents", headers=me).status_code for _ in range(5)]

    assert codes == [200] * 5
