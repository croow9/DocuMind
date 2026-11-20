from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.messaging.producer import KafkaProducerManager
from app.schemas.events import IngestionRequested


async def test_send_before_start_raises() -> None:
    manager = KafkaProducerManager("localhost:9094")

    with pytest.raises(RuntimeError):
        await manager.send("topic", IngestionRequested(document_id=uuid4()))


async def test_send_passes_json_bytes_topic_and_key_to_kafka() -> None:
    manager = KafkaProducerManager("localhost:9094")
    manager.producer = AsyncMock()
    event = IngestionRequested(document_id=uuid4())

    await manager.send("my-topic", event, key="my-key")

    kwargs = manager.producer.send_and_wait.await_args.kwargs
    assert kwargs["topic"] == "my-topic"
    assert kwargs["key"] == "my-key"
    # в Kafka уходят байты JSON, а не сам объект события
    assert isinstance(kwargs["value"], bytes)
    assert IngestionRequested.model_validate_json(kwargs["value"]) == event
