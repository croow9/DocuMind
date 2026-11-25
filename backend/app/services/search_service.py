from collections.abc import Sequence
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.embedder import EmbeddingClient
from app.rag.retrieval import CANDIDATES_PER_SEARCH, TOP_K, reciprocal_rank_fusion
from app.repositories.chunk_repo import ChunkRepository
from app.schemas.search import RetrievedChunk


def _ids(chunks: Sequence[RetrievedChunk]) -> list[UUID]:
    # у чанков из базы id есть всегда; None бывает только у страниц из интернета,
    # а они сюда не попадают. Проверка нужна типам, а не логике
    return [chunk.chunk_id for chunk in chunks if chunk.chunk_id is not None]


class SearchService:
    def __init__(self, session: AsyncSession, embedder: EmbeddingClient) -> None:
        self.repo = ChunkRepository(session)
        # эмбеддер передаётся снаружи: тесты подсовывают фейк, и поиск не ходит в сеть
        self.embedder = embedder

    async def search(
        self,
        owner_id: UUID,
        question: str,
        document_ids: Sequence[UUID] | None = None,
        top_k: int = TOP_K,
    ) -> list[RetrievedChunk]:
        """Лучшие чанки по вопросу среди готовых документов пользователя."""
        if not question.strip():
            # пустой вопрос нечем искать, а OpenAI на пустой вход отвечает ошибкой
            return []

        [query_vector] = await self.embedder.embed([question])

        # Два запроса идут один за другим: в одной сессии параллельные запросы
        # недопустимы (asyncio.gather на общей сессии падает)
        by_meaning = await self.repo.search_by_vector(
            query_vector, owner_id, document_ids, CANDIDATES_PER_SEARCH
        )
        by_words = await self.repo.search_by_text(
            question, owner_id, document_ids, CANDIDATES_PER_SEARCH
        )

        # слияние работает с id, а чанки достаём обратно по словарю
        chunks = {chunk.chunk_id: chunk for chunk in [*by_meaning, *by_words]}
        merged = reciprocal_rank_fusion([_ids(by_meaning), _ids(by_words)])
        return [chunks[chunk_id] for chunk_id in merged[:top_k]]
