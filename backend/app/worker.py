import asyncio
import logging
import signal
from contextlib import AsyncExitStack
from uuid import UUID

from pydantic import ValidationError

from app.core.config import settings
from app.db.session import engine
from app.messaging.consumer import KafkaConsumerManager
from app.messaging.producer import KafkaProducerManager
from app.messaging.topics import IngestionTopics
from app.schemas.events import DeadLetter, IngestionRequested
from app.services.ingestion_service import index_document, mark_failed

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 3.0
POLL_TIMEOUT_MS = 1000


async def send_to_dlq(
    producer: KafkaProducerManager, value: bytes, key: bytes | None, error: str
) -> None:
    """Отправить неудачное сообщение в DLQ вместе с причиной отказа."""
    letter = DeadLetter(
        source_topic=IngestionTopics.REQUESTED.value,
        original_value=value.decode("utf-8", errors="replace"),
        error=error[:2000],
    )
    dlq_key = key.decode("utf-8", errors="replace") if key else None
    await producer.send(IngestionTopics.DLQ, letter, key=dlq_key)


async def process_document(document_id: UUID) -> str | None:
    """До MAX_ATTEMPTS попыток проиндексировать документ.

    Возвращает None при успехе или текст последней ошибки, если попытки кончились.
    """
    error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            await index_document(document_id)
            return None
        # ловим всё осознанно: любой сбой индексации — повод повторить, а в итоге
        # пометить документ failed, но не ронять воркер
        except Exception as exc:  # noqa: BLE001
            error = str(exc) or type(exc).__name__
            logger.warning(
                "Индексация %s: попытка %d/%d не удалась: %s",
                document_id,
                attempt,
                MAX_ATTEMPTS,
                error,
            )
            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(RETRY_DELAY_SECONDS)
    return error


async def handle_message(
    producer: KafkaProducerManager, value: bytes, key: bytes | None
) -> None:
    """Обработать одно сообщение. Исключение наружу = сообщение НЕ обработано.

    Воркер в этом случае падает, не ставя закладку, и после перезапуска получает
    сообщение снова: молча пропускать сообщения нельзя.
    """
    try:
        event = IngestionRequested.model_validate_json(value)
    except ValidationError as exc:
        # ядовитое сообщение: повторять бессмысленно, сразу в DLQ
        logger.error("Сообщение не разобрано, уходит в DLQ: %s", exc)
        await send_to_dlq(producer, value, key, f"Невалидное сообщение: {exc}")
        return

    error = await process_document(event.document_id)
    if error is None:
        return

    await mark_failed(event.document_id, error)
    await send_to_dlq(producer, value, key, error)


async def main() -> None:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    producer = KafkaProducerManager(settings.kafka_bootstrap_servers)
    consumer = KafkaConsumerManager(
        topic=IngestionTopics.REQUESTED,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_consumer_group,
        auto_offset_reset=settings.kafka_auto_offset_reset,
    )

    # закрытие в обратном порядке: consumer, producer, engine — даже если что-то
    # из этого упадёт или запуск оборвётся посередине
    async with AsyncExitStack() as stack:
        stack.push_async_callback(engine.dispose)
        await producer.start()
        stack.push_async_callback(producer.stop)
        await consumer.start()
        stack.push_async_callback(consumer.stop)

        logger.info("Воркер запущен, жду сообщения")
        while not stop_event.is_set():
            record = await consumer.get_message(timeout_ms=POLL_TIMEOUT_MS)
            if record is None:
                continue
            await handle_message(producer, record.value, record.key)
            # закладка — только после обработки
            await consumer.commit()
        logger.info("Получен сигнал остановки, воркер завершается")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    asyncio.run(main())
