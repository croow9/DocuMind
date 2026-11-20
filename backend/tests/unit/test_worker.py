from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.messaging.producer import KafkaProducerManager
from app.messaging.topics import IngestionTopics
from app.schemas.events import IngestionRequested
from app.worker import (
    MAX_ATTEMPTS,
    handle_message,
    process_document,
    send_to_dlq,
)


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    # паузы между попытками в тестах не нужны: иначе каждый тест ждал бы секунды
    monkeypatch.setattr("app.worker.RETRY_DELAY_SECONDS", 0)


@pytest.fixture
def index_mock(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock()
    # подменяем имя там, где его использует воркер, а не там, где оно определено
    monkeypatch.setattr("app.worker.index_document", mock)
    return mock


@pytest.fixture
def mark_failed_mock(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock()
    monkeypatch.setattr("app.worker.mark_failed", mock)
    return mock


@pytest.fixture
def producer() -> AsyncMock:
    return AsyncMock(spec=KafkaProducerManager)


def make_event_bytes() -> tuple[bytes, object]:
    event = IngestionRequested(document_id=uuid4())
    return event.model_dump_json().encode(), event.document_id


# --- process_document: повторные попытки


async def test_process_document_succeeds_on_first_attempt(
    index_mock: AsyncMock,
) -> None:
    result = await process_document(uuid4())

    assert result is None
    assert index_mock.await_count == 1


async def test_process_document_retries_until_success(index_mock: AsyncMock) -> None:
    index_mock.side_effect = [ValueError("сбой 1"), ValueError("сбой 2"), None]

    result = await process_document(uuid4())

    assert result is None
    assert index_mock.await_count == 3


async def test_process_document_returns_last_error_after_all_attempts(
    index_mock: AsyncMock,
) -> None:
    index_mock.side_effect = ValueError("битый PDF")

    result = await process_document(uuid4())

    assert result == "битый PDF"
    assert index_mock.await_count == MAX_ATTEMPTS


# --- handle_message: что происходит с сообщением


async def test_garbage_message_goes_to_dlq_without_indexing(
    producer: AsyncMock, index_mock: AsyncMock, mark_failed_mock: AsyncMock
) -> None:
    await handle_message(producer, b"not json at all", None)

    index_mock.assert_not_awaited()
    mark_failed_mock.assert_not_awaited()
    producer.send.assert_awaited_once()
    topic, letter = producer.send.await_args.args
    assert topic == IngestionTopics.DLQ
    assert letter.original_value == "not json at all"


async def test_successful_message_does_not_touch_dlq_or_failed_status(
    producer: AsyncMock, index_mock: AsyncMock, mark_failed_mock: AsyncMock
) -> None:
    value, _ = make_event_bytes()

    await handle_message(producer, value, None)

    index_mock.assert_awaited_once()
    mark_failed_mock.assert_not_awaited()
    producer.send.assert_not_awaited()


async def test_failed_indexing_marks_document_and_sends_to_dlq(
    producer: AsyncMock, index_mock: AsyncMock, mark_failed_mock: AsyncMock
) -> None:
    index_mock.side_effect = ValueError("битый PDF")
    value, document_id = make_event_bytes()

    await handle_message(producer, value, b"key")

    mark_failed_mock.assert_awaited_once_with(document_id, "битый PDF")
    producer.send.assert_awaited_once()
    _, letter = producer.send.await_args.args
    assert letter.error == "битый PDF"


async def test_mark_failed_error_is_not_swallowed(
    producer: AsyncMock, index_mock: AsyncMock, mark_failed_mock: AsyncMock
) -> None:
    # если даже пометить документ не вышло, воркер должен упасть, а не пропустить
    # сообщение: после перезапуска оно придёт снова
    index_mock.side_effect = ValueError("битый PDF")
    mark_failed_mock.side_effect = RuntimeError("БД недоступна")
    value, _ = make_event_bytes()

    with pytest.raises(RuntimeError):
        await handle_message(producer, value, None)

    producer.send.assert_not_awaited()


# --- send_to_dlq: формат письма


@pytest.mark.parametrize(
    ("key", "expected_key"),
    [(b"abc", "abc"), (None, None)],
)
async def test_send_to_dlq_survives_broken_bytes_and_missing_key(
    producer: AsyncMock, key: bytes | None, expected_key: str | None
) -> None:
    await send_to_dlq(producer, b"\xff\xfe broken", key, "причина")

    topic, letter = producer.send.await_args.args
    assert topic == IngestionTopics.DLQ
    assert producer.send.await_args.kwargs["key"] == expected_key
    assert letter.source_topic == IngestionTopics.REQUESTED
    assert "�" in letter.original_value
    assert letter.error == "причина"
