import asyncio
from abc import ABC, abstractmethod

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.openai_client import build_openai_client

# Небольшие батчи уходят в OpenAI одновременно: ответ на 150 кусков страницы
# приходит за время самого медленного батча, а не за сумму времён
EMBEDDING_BATCH_SIZE = 50
EMBEDDING_CONCURRENCY = 4


class EmbeddingClient(ABC):
    """Переводчик текстов в векторы смысла: список текстов -> список векторов."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OpenAIEmbedder(EmbeddingClient):
    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        self._client = client or build_openai_client()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        semaphore = asyncio.Semaphore(EMBEDDING_CONCURRENCY)

        async def embed_batch(batch: list[str]) -> list[list[float]]:
            async with semaphore:
                response = await self._client.embeddings.create(
                    input=batch, model=settings.embedding_model
                )
                return [item.embedding for item in response.data]

        tasks = [
            asyncio.create_task(
                embed_batch(texts[start : start + EMBEDDING_BATCH_SIZE])
            )
            for start in range(0, len(texts), EMBEDDING_BATCH_SIZE)
        ]
        try:
            results = await asyncio.gather(*tasks)
        except Exception:
            # gather не отменяет остальные задачи, когда одна упала: без этого
            # запросы к OpenAI работали бы впустую (при отмене всего вызова он
            # отменяет их сам)
            for task in tasks:
                task.cancel()
            raise
        # gather отдаёт результаты в порядке батчей, а не завершения
        return [vector for batch in results for vector in batch]


class FakeEmbedder(EmbeddingClient):
    """Вектор из суммы байтов текста: детерминированный, без сети и ключа.

    Близости смыслов у него нет — он нужен, чтобы тесты и разработка
    без ключа OpenAI прогоняли весь конвейер индексации.
    """

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._fake_vector(text) for text in texts]

    @staticmethod
    def _fake_vector(text: str) -> list[float]:
        dimensions = settings.embedding_dimensions
        vector = [0.0] * dimensions
        for i, byte in enumerate(text.encode("utf-8")):
            vector[i % dimensions] += byte
        return vector


def build_embedder() -> EmbeddingClient:
    """Настоящий API при наличии ключа, детерминированный фейк — без него."""
    if settings.openai_api_key:
        return OpenAIEmbedder()
    return FakeEmbedder()
