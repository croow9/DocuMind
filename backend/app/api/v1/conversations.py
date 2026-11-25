from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.api.deps import (
    ChatClientDep,
    EmbedderDep,
    SessionDep,
    UserDep,
    WebSearchDep,
)
from app.rag.llm import ChatMessage
from app.schemas.chat import (
    ConversationCreate,
    ConversationDetail,
    ConversationRead,
    MessageCreate,
)
from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationService
from app.services.search_service import SearchService

router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.post("", response_model=ConversationRead, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    user: UserDep, db: SessionDep, payload: ConversationCreate | None = None
) -> ConversationRead:
    """Тело можно не передавать: тогда заголовок по умолчанию заменится первым вопросом."""
    service = ConversationService(db)
    conversation = await service.create(user.id, payload or ConversationCreate())
    return ConversationRead.model_validate(conversation)


@router.get("", response_model=list[ConversationRead])
async def list_conversations(
    user: UserDep,
    db: SessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ConversationRead]:
    service = ConversationService(db)
    conversations = await service.list_conversations(
        user.id, limit=limit, offset=offset
    )
    return [ConversationRead.model_validate(c) for c in conversations]


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: UUID, user: UserDep, db: SessionDep
) -> ConversationDetail:
    return await ConversationService(db).get_detail(user.id, conversation_id)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: UUID, user: UserDep, db: SessionDep
) -> None:
    await ConversationService(db).delete(user.id, conversation_id)


def get_chat_service(
    db: SessionDep, embedder: EmbedderDep, chat: ChatClientDep, web: WebSearchDep
) -> ChatService:
    return ChatService(db, SearchService(db, embedder), chat, web)


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]


async def prepare_message(
    conversation_id: UUID,
    payload: MessageCreate,
    user: UserDep,
    service: ChatServiceDep,
) -> list[ChatMessage]:
    """Проверки и сохранение вопроса: зависимость выполняется до начала ответа.

    Если вопрос не прошёл проверку, клиент получит обычный 404, а не поток.
    """
    return await service.prepare(user.id, conversation_id, payload)


HistoryDep = Annotated[list[ChatMessage], Depends(prepare_message)]


@router.post(
    "/{conversation_id}/messages",
    response_class=EventSourceResponse,
    summary="Задать вопрос: ответ приходит потоком SSE",
)
async def send_message(
    conversation_id: UUID,
    payload: MessageCreate,
    user: UserDep,
    service: ChatServiceDep,
    history: HistoryDep,
) -> AsyncIterator[ServerSentEvent]:
    """События потока: `sources` (источники), `token` (кусочки ответа),
    затем `done` (id сообщения и номера использованных источников) или `error`.

    По умолчанию ищет в документах пользователя. С `web_search: true` ищет в
    официальной документации из интернета (список сайтов задан в настройках):
    у таких источников вместо id документа и чанка приходит `url`. Вместе с
    `document_ids` флаг не принимается.
    """
    async for event in service.answer(user.id, conversation_id, payload, history):
        yield event
