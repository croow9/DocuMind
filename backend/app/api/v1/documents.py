from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, UploadFile, status

from app.api.deps import ProducerDep, SessionDep, UserDep
from app.api.rate_limit import UPLOAD_PER_MINUTE
from app.schemas.document import DocumentCreate, DocumentRead
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(UPLOAD_PER_MINUTE)],
)
async def create_document(
    payload: DocumentCreate, producer: ProducerDep, user: UserDep, db: SessionDep
) -> DocumentRead:
    """Документ по ссылке. Частота ограничена на пользователя: сверх лимита 429 с `Retry-After`."""
    service = DocumentService(db)
    document = await service.create_from_url(user, payload)
    await service.request_indexing(document, producer)
    return DocumentRead.model_validate(document)


@router.post(
    "/upload",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(UPLOAD_PER_MINUTE)],
)
async def upload_document(
    producer: ProducerDep,
    file: UploadFile,
    user: UserDep,
    db: SessionDep,
) -> DocumentRead:
    """Загрузка файла. Частота ограничена на пользователя: сверх лимита 429 с `Retry-After`."""
    service = DocumentService(db)
    document = await service.create_from_file(user, file)
    await service.request_indexing(document, producer)
    return DocumentRead.model_validate(document)


@router.get("", response_model=list[DocumentRead])
async def list_documents(
    user: UserDep,
    db: SessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[DocumentRead]:
    service = DocumentService(db)
    documents = await service.list_documents(user, limit=limit, offset=offset)
    return [DocumentRead.model_validate(d) for d in documents]


@router.get("/{document_id}", response_model=DocumentRead)
async def get_document(
    document_id: UUID, user: UserDep, db: SessionDep
) -> DocumentRead:
    service = DocumentService(db)
    document = await service.get(user, document_id)
    return DocumentRead.model_validate(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: UUID, user: UserDep, db: SessionDep) -> None:
    service = DocumentService(db)
    await service.delete(user, document_id)
