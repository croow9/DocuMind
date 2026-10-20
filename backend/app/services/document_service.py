from collections.abc import Sequence
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.storage import DATA_DIR, document_file_path
from app.models import DocumentORM, UserORM
from app.models.document import SourceType
from app.repositories.document_repo import DocumentRepository
from app.schemas.document import DocumentCreate

ALLOWED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}


class DocumentService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repo = DocumentRepository(session)

    async def create_from_url(
        self, user: UserORM, payload: DocumentCreate
    ) -> DocumentORM:
        document = DocumentORM(
            owner_id=user.id,
            title=payload.title,
            source_type=payload.source_type,
            source_url=str(payload.source_url),
        )
        return await self._create(document)

    async def create_from_file(self, user: UserORM, file: UploadFile) -> DocumentORM:
        original = file.filename or "upload"
        suffix = Path(original).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Поддерживаются только файлы PDF, Markdown и TXT",
            )

        # id задаём сами: файл на диске хранится под именем документа,
        # а default в модели сработал бы только на flush — поздно для имени файла
        document = DocumentORM(
            id=uuid4(),
            owner_id=user.id,
            title=Path(original).stem or "document",
            source_type=SourceType.UPLOAD,
            original_filename=original,
        )

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        document_file_path(document.id, suffix).write_bytes(await file.read())

        return await self._create(document)

    async def get(self, user: UserORM, document_id: UUID) -> DocumentORM:
        document = await self.repo.get_by_id(document_id, user.id)
        if document is None:
            # Чужой документ неотличим от несуществующего — не раскрываем факт существования
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Документ не найден",
            )
        return document

    async def list_documents(
        self, user: UserORM, limit: int = 20, offset: int = 0
    ) -> Sequence[DocumentORM]:
        return await self.repo.list_for_owner(user.id, limit=limit, offset=offset)

    async def delete(self, user: UserORM, document_id: UUID) -> None:
        document = await self.get(user, document_id)
        await self.repo.delete(document)
        await self.session.commit()

    async def _create(self, document: DocumentORM) -> DocumentORM:
        await self.repo.create(document)
        await self.session.commit()
        return document
