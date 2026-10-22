from abc import ABC, abstractmethod

from openai import AsyncOpenAI

from app.core.config import settings

EMBEDDING_BATCH_SIZE = 100


class EmbeddingClient(ABC):
    """Переводчик текстов в векторы смысла: список текстов -> список векторов."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OpenAIEmbedder(EmbeddingClient):
    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        self._client = client or AsyncOpenAI(api_key=settings.openai_api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + EMBEDDING_BATCH_SIZE]
            response = await self._client.embeddings.create(
                input=batch,
                model=settings.embedding_model,
            )
            vectors.extend(item.embedding for item in response.data)
        return vectors


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
