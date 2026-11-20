from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from aiokafka.errors import KafkaConnectionError
from fastapi import HTTPException

from app.messaging.topics import IngestionTopics
from app.models import DocumentORM
from app.models.document import DocumentStatus
from app.services.document_service import DocumentService


def make_document() -> DocumentORM:
    # объект создаётся в памяти, без базы: нужны только id и статус
    return DocumentORM(id=uuid4(), status=DocumentStatus.PENDING)


async def test_request_indexing_sends_event_and_keeps_document_pending() -> None:
    session = AsyncMock()
    producer = AsyncMock()
    document = make_document()

    await DocumentService(session).request_indexing(document, producer)

    producer.send.assert_awaited_once()
    assert producer.send.await_args.args[0] == IngestionTopics.REQUESTED
    assert producer.send.await_args.kwargs["key"] == str(document.id)
    assert document.status is DocumentStatus.PENDING
    session.commit.assert_not_awaited()


async def test_request_indexing_marks_document_failed_when_kafka_is_down() -> None:
    session = AsyncMock()
    producer = AsyncMock()
    producer.send.side_effect = KafkaConnectionError("брокер недоступен")
    document = make_document()

    with pytest.raises(HTTPException) as exc_info:
        await DocumentService(session).request_indexing(document, producer)

    assert exc_info.value.status_code == 503
    assert document.status is DocumentStatus.FAILED
    assert document.error
    session.commit.assert_awaited_once()
