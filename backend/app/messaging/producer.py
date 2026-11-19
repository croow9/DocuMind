import logging

from aiokafka import AIOKafkaProducer
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class KafkaProducerManager:
    def __init__(self, bootstrap_servers: str | list[str]) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            key_serializer=lambda k: str(k).encode("utf-8") if k is not None else None,
            request_timeout_ms=5000,
        )
        try:
            await producer.start()
            self.producer = producer
            logger.info("Kafka producer успешно запущен.")
        except Exception as e:
            await producer.stop()
            logger.error(f"Ошибка при запуске Kafka producer: {e}")
            raise

    async def stop(self) -> None:
        if self.producer:
            try:
                await self.producer.stop()
            finally:
                self.producer = None
                logger.info("Kafka Producer stopped")

    async def send(self, topic: str, value: BaseModel, key: str | None = None) -> None:
        if not self.producer:
            raise RuntimeError("Kafka producer не запущен. Сначала вызовите start().")

        try:
            payload = value.model_dump_json().encode("utf-8")
            await self.producer.send_and_wait(topic=topic, value=payload, key=key)
        except Exception as e:
            logger.error(f"Ошибка при отправке сообщения в топик {topic}: {e}")
            raise
