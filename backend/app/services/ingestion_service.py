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

    Работает как фоновая задача (в M3 — по событию Kafka), поэтому открывает
    собственную сессию: сессия HTTP-запроса к этому моменту уже закрыта.
    Любая ошибка переводит документ в failed с текстом причины —
    документ не теряется никогда.
    """
    async with AsyncSessionLocal() as session:
        document = await DocumentRepository(session).get_by_document_id(document_id)
        if document is None:
            logger.warning("Document %s not found, indexing skipped", document_id)
            return

        try:
            await _run_pipeline(session, document)
        # ловим всё осознанно: конвейер не должен падать — ошибка становится статусом
        except Exception as exc:  # noqa: BLE001
            await session.rollback()
            # после rollback объект протух: заново загружаем и помечаем failed
            fresh = await DocumentRepository(session).get_by_document_id(document_id)
            if fresh is not None:
                fresh.status = DocumentStatus.FAILED
                fresh.error = str(exc)[:2000]
                await session.commit()
            logger.error("Indexing failed for %s: %s", document_id, exc)


async def _run_pipeline(session: AsyncSession, document: DocumentORM) -> None:
    document.status = DocumentStatus.PROCESSING
    await session.commit()

    if document.source_type is SourceType.URL:
        raise ValueError("Индексация документов по URL появится после M2 (отложено)")

    suffix = Path(document.original_filename or "").suffix.lower()
    text = parse_file(document_file_path(document.id, suffix))

    pieces = split_into_chunks(text)
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
    await ChunkRepository(session).create_all(chunks)

    document.chunk_count = len(chunks)
    document.status = DocumentStatus.READY
    document.error = None
    await session.commit()
    logger.info("Document %s indexed: %d chunks", document.id, len(chunks))
