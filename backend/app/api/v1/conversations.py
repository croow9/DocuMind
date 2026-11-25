from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.api.deps import ChatClientDep, EmbedderDep, SessionDep, UserDep
from app.rag.llm import ChatMessage
from app.schemas.chat import MessageCreate
from app.services.chat_service import ChatService
from app.services.search_service import SearchService

router = APIRouter(prefix="/conversations", tags=["Conversations"])


def get_chat_service(
    db: SessionDep, embedder: EmbedderDep, chat: ChatClientDep
) -> ChatService:
    return ChatService(db, SearchService(db, embedder), chat)


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
    """
    async for event in service.answer(user.id, conversation_id, payload, history):
        yield event
