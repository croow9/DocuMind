from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.core.config import settings
from app.models.document import SourceType
from app.repositories.document_repo import DocumentRepository
from app.schemas.document import DocumentCreate
from app.services.document_service import DocumentService

MAX_BYTES = 2 * 1024 * 1024
MAX_DOCUMENTS = 3


class FakeUpload:
    """Подставной файл: запоминает, сколько байт у него просили прочитать."""

    def __init__(self, content: bytes, filename: str = "note.txt") -> None:
        self.filename = filename
        self._content = content
        self.read_sizes: list[int] = []

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return self._content if size < 0 else self._content[:size]


@pytest.fixture(autouse=True)
def small_limits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings, "max_upload_bytes", MAX_BYTES)
    monkeypatch.setattr(settings, "max_documents_per_user", MAX_DOCUMENTS)
    # файлы пишутся во временную папку, а не в настоящее хранилище
    monkeypatch.setattr("app.services.document_service.DATA_DIR", tmp_path)
    monkeypatch.setattr(
        "app.services.document_service.document_file_path",
        lambda document_id, suffix: tmp_path / f"{document_id}{suffix}",
    )


def make_service(owned_documents: int) -> tuple[DocumentService, AsyncMock]:
    session = AsyncMock()
    session.add = MagicMock()
    service = DocumentService(session)
    service.repo = AsyncMock()
    service.repo.count_for_owner.return_value = owned_documents
    service.repo.create.side_effect = lambda document: document
    return service, session


def make_user() -> SimpleNamespace:
    return SimpleNamespace(id=uuid4())


URL_PAYLOAD = DocumentCreate(
    title="Руководство", source_url="https://docs.example.com/g"
)  # type: ignore[arg-type]


# --- размер файла


async def test_a_file_of_exactly_the_limit_is_accepted_and_saved(
    tmp_path: Path,
) -> None:
    service, _ = make_service(owned_documents=0)

    document = await service.create_from_file(
        make_user(),
        FakeUpload(b"x" * MAX_BYTES),  # type: ignore[arg-type]
    )

    assert document.source_type is SourceType.UPLOAD
    [saved] = list(tmp_path.iterdir())
    assert saved.stat().st_size == MAX_BYTES


async def test_a_file_one_byte_over_the_limit_is_413_and_nothing_is_written(
    tmp_path: Path,
) -> None:
    service, session = make_service(owned_documents=0)

    with pytest.raises(HTTPException) as exc_info:
        await service.create_from_file(
            make_user(),
            FakeUpload(b"x" * (MAX_BYTES + 1)),  # type: ignore[arg-type]
        )

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail == "Файл больше 2 МБ"
    assert list(tmp_path.iterdir()) == []
    service.repo.create.assert_not_awaited()
    session.commit.assert_not_awaited()


async def test_the_file_is_read_with_a_bound_so_a_huge_one_never_fills_memory() -> None:
    service, _ = make_service(owned_documents=0)
    upload = FakeUpload(b"x" * 10)

    await service.create_from_file(make_user(), upload)  # type: ignore[arg-type]

    # лимит и ещё один байт: хватает, чтобы понять «больше», и не больше
    assert upload.read_sizes == [MAX_BYTES + 1]


async def test_the_format_is_checked_before_the_file_is_read() -> None:
    service, _ = make_service(owned_documents=0)
    upload = FakeUpload(b"x", filename="virus.exe")

    with pytest.raises(HTTPException) as exc_info:
        await service.create_from_file(make_user(), upload)  # type: ignore[arg-type]

    assert exc_info.value.status_code == 415
    assert upload.read_sizes == []


# --- число документов


async def test_below_the_document_limit_a_file_is_accepted() -> None:
    service, _ = make_service(owned_documents=MAX_DOCUMENTS - 1)

    await service.create_from_file(make_user(), FakeUpload(b"hi"))  # type: ignore[arg-type]

    service.repo.create.assert_awaited_once()


async def test_at_the_document_limit_a_file_is_409_and_nothing_is_written(
    tmp_path: Path,
) -> None:
    service, session = make_service(owned_documents=MAX_DOCUMENTS)

    with pytest.raises(HTTPException) as exc_info:
        await service.create_from_file(make_user(), FakeUpload(b"hi"))  # type: ignore[arg-type]

    assert exc_info.value.status_code == 409
    assert str(MAX_DOCUMENTS) in exc_info.value.detail
    assert list(tmp_path.iterdir()) == []  # отклонённый файл ничего не оставляет
    service.repo.create.assert_not_awaited()
    session.commit.assert_not_awaited()


async def test_below_the_document_limit_a_link_is_accepted() -> None:
    service, _ = make_service(owned_documents=MAX_DOCUMENTS - 1)

    await service.create_from_url(make_user(), URL_PAYLOAD)

    service.repo.create.assert_awaited_once()


async def test_at_the_document_limit_a_link_is_409() -> None:
    service, session = make_service(owned_documents=MAX_DOCUMENTS)

    with pytest.raises(HTTPException) as exc_info:
        await service.create_from_url(make_user(), URL_PAYLOAD)

    assert exc_info.value.status_code == 409
    service.repo.create.assert_not_awaited()
    session.commit.assert_not_awaited()


async def test_the_quota_is_counted_for_the_user_who_uploads() -> None:
    service, _ = make_service(owned_documents=0)
    user = make_user()

    await service.create_from_url(user, URL_PAYLOAD)

    service.repo.count_for_owner.assert_awaited_once_with(user.id)


# --- настоящие значения по умолчанию


def test_defaults_are_20_mb_and_50_documents() -> None:
    from app.core.config import Settings

    fresh = Settings(_env_file=None)  # type: ignore[call-arg]

    assert fresh.max_upload_bytes == 20 * 1024 * 1024
    assert fresh.max_documents_per_user == 50


# --- репозиторий: считаем только свои документы


async def test_counting_is_limited_to_the_owner_and_returns_the_number() -> None:
    session = AsyncMock()
    session.execute.return_value = MagicMock()
    session.execute.return_value.scalar_one.return_value = 7
    owner_id = uuid4()

    count = await DocumentRepository(session).count_for_owner(owner_id)

    [call] = session.execute.await_args_list
    statement = call.args[0]
    assert count == 7
    assert "count" in str(statement).lower()
    assert "documents.owner_id = " in str(statement)
    assert list(statement.compile().params.values()) == [owner_id]
