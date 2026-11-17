import time
from datetime import timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas.events import IngestionRequested


def test_event_gets_default_id_and_utc_time() -> None:
    event = IngestionRequested(document_id=uuid4())

    assert event.occurred_at.tzinfo is not None
    assert event.occurred_at.utcoffset() == timedelta(0)


def test_each_event_gets_its_own_id_and_time() -> None:
    first = IngestionRequested(document_id=uuid4())
    time.sleep(0.01)
    second = IngestionRequested(document_id=first.document_id)

    assert first.event_id != second.event_id
    assert second.occurred_at > first.occurred_at


def test_event_survives_json_round_trip() -> None:
    event = IngestionRequested(document_id=uuid4())

    restored = IngestionRequested.model_validate_json(event.model_dump_json())

    assert restored == event


def test_unknown_fields_are_ignored() -> None:
    document_id = uuid4()

    event = IngestionRequested.model_validate_json(
        f'{{"document_id": "{document_id}", "priority": "high"}}'
    )

    assert event.document_id == document_id


@pytest.mark.parametrize(
    "garbage_json",
    [
        "",
        "просто текст, не json",
        '{"document_id": "12345"',
        '{"document_id": undefined}',
        "[]",
        '"just a json string"',
        "12345",
        "null",
        # корректный JSON
        "{}",
        '{"document_id": null}',
        '{"document_id": "12345"}',
    ],
)
def test_garbage_json_raises_validation_error(garbage_json: str) -> None:
    with pytest.raises(ValidationError):
        IngestionRequested.model_validate_json(garbage_json)
