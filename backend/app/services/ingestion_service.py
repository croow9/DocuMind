import asyncio
import logging
from pathlib import Path
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.ingestion.chunker import split_into_chunks
from app.ingestion.embedder import build_embedder
from app.ingestion.parser import parse_file
from app.ingestion.storage import document_file_path
from app.models import ChunkORM, DocumentORM
from app.models.document import DocumentStatus, SourceType
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.document_repo import DocumentRepository

logger = logging.getLogger(__name__)


async def index_document(document_id: UUID) -> None:
    """Прогнать документ через конвейер индексации.

    Работает по событию Kafka, поэтому открывает собственную сессию: сессия
    HTTP-запроса к этому моменту уже закрыта. Ошибки НЕ ловит: при сбое бросает
    исключение, а вызывающий (воркер) решает — повторить попытку или вызвать
    mark_failed. Всё незакоммиченное откатывается при выходе из сессии.
    """
    async with AsyncSessionLocal() as session:
        document = await DocumentRepository(session).get_by_document_id(document_id)
        if document is None:
            logger.warning("Document %s not found, indexing skipped", document_id)
            return

        await _run_pipeline(session, document)


async def mark_failed(document_id: UUID, error: str) -> None:
    """Пометить документ как failed с текстом причины.

    Вызывается, когда попытки индексации закончились. Открывает новую сессию:
    сессия упавшей индексации отравлена ошибкой и объект в ней устарел.
    """
    async with AsyncSessionLocal() as session:
        document = await DocumentRepository(session).get_by_document_id(document_id)
        if document is None:
            logger.warning("Document %s not found, mark_failed skipped", document_id)
            return

        document.status = DocumentStatus.FAILED
        document.error = error[:2000]
        await session.commit()
    logger.error("Indexing failed for %s: %s", document_id, error)


async def _run_pipeline(session: AsyncSession, document: DocumentORM) -> None:
    document.status = DocumentStatus.PROCESSING
    await session.commit()

    if document.source_type is SourceType.URL:
        raise ValueError("Индексация документов по URL появится после M2 (отложено)")

    suffix = Path(document.original_filename or "").suffix.lower()
    # парсинг синхронный (pypdf): в потоке, чтобы не вешать event loop
    text = await asyncio.to_thread(parse_file, document_file_path(document.id, suffix))

    pieces = split_into_chunks(text)
    if not pieces:
        # скан без текстового слоя или пустой файл: «ready» без чанков вводит в заблуждение
        raise ValueError("В документе не найден текст для индексации")
    vectors = await build_embedder().embed([piece.content for piece in pieces])

    chunks = [
        ChunkORM(
            document_id=document.id,
            ordinal=piece.ordinal,
            content=piece.content,
            heading_path=piece.heading_path or None,
            embedding=vector,
        )
        for piece, vector in zip(pieces, vectors, strict=True)
    ]
    chunk_repo = ChunkRepository(session)
    # идемпотентность: событие может прийти повторно, старые чанки заменяются
    # новыми в одной транзакции (коммит ниже)
    await chunk_repo.delete_by_document_id(document.id)
    await chunk_repo.create_all(chunks)

    document.chunk_count = len(chunks)
    document.status = DocumentStatus.READY
    document.error = None
    await session.commit()
    logger.info("Document %s indexed: %d chunks", document.id, len(chunks))
