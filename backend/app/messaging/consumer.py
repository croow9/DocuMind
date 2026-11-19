import logging

from aiokafka import AIOKafkaConsumer, ConsumerRecord

logger = logging.getLogger(__name__)

# Эмбеддинги большого документа могут считаться долго. Если воркер молчит дольше
# этого срока, Kafka решит, что он умер, и отдаст сообщение другому
MAX_POLL_INTERVAL_MS = 600_000


class KafkaConsumerManager:
    """Тонкая обёртка над AIOKafkaConsumer: только чтение и commit.

    Решения «что делать с ошибкой» здесь нет: их принимает воркер. Сообщения
    отдаются сырыми байтами, без десериализации: мусор не должен ронять консьюмер,
    разбирать его — дело обработчика.
    """

    def __init__(
        self,
        topic: str,
        bootstrap_servers: str | list[str],
        group_id: str,
        auto_offset_reset: str = "earliest",
    ) -> None:
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self.consumer: AIOKafkaConsumer | None = None

    async def start(self) -> None:
        consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset=self.auto_offset_reset,
            # закладку ставим сами и только после обработки сообщения
            enable_auto_commit=False,
            max_poll_interval_ms=MAX_POLL_INTERVAL_MS,
        )
        try:
            await consumer.start()
        except Exception:
            await consumer.stop()
            raise
        self.consumer = consumer
        logger.info(
            "Kafka consumer запущен: topic=%s group=%s", self.topic, self.group_id
        )

    async def stop(self) -> None:
        if self.consumer:
            try:
                await self.consumer.stop()
            finally:
                self.consumer = None
                logger.info("Kafka consumer остановлен")

    async def get_message(self, timeout_ms: int = 1000) -> ConsumerRecord | None:
        """Одно следующее сообщение или None, если за timeout_ms ничего не пришло.

        Ограничение по времени позволяет воркеру регулярно проверять, не просят ли
        его остановиться, а не висеть в ожидании сообщения.
        """
        if not self.consumer:
            raise RuntimeError("Kafka consumer не запущен. Сначала вызовите start().")

        batches = await self.consumer.getmany(timeout_ms=timeout_ms, max_records=1)
        for records in batches.values():
            if records:
                return records[0]
        return None

    async def commit(self) -> None:
        """Зафиксировать позицию: все выданные сообщения считаются обработанными."""
        if not self.consumer:
            raise RuntimeError("Kafka consumer не запущен. Сначала вызовите start().")
        await self.consumer.commit()
