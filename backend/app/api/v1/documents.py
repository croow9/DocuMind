from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Query, UploadFile, status

from app.api.deps import SessionDep, UserDep
from app.schemas.document import DocumentCreate, DocumentRead
from app.services.document_service import DocumentService
from app.services.ingestion_service import index_document

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def create_document(
    payload: DocumentCreate, user: UserDep, db: SessionDep
) -> DocumentRead:
    service = DocumentService(db)
    document = await service.create_from_url(user, payload)
    return DocumentRead.model_validate(document)


@router.post(
    "/upload", response_model=DocumentRead, status_code=status.HTTP_201_CREATED
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    user: UserDep,
    db: SessionDep,
) -> DocumentRead:
    service = DocumentService(db)
    document = await service.create_from_file(user, file)
    background_tasks.add_task(index_document, document.id)
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
