import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from uuid import UUID

from fastapi import HTTPException, status
from fastapi.sse import ServerSentEvent
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.session import AsyncSessionLocal
from app.models import CitationORM
from app.models.message import MessageRole
from app.rag.llm import ChatClient, ChatMessage
from app.rag.prompt import (
    MAX_HISTORY_MESSAGES,
    NO_ANSWER_TEXT,
    NO_WEB_ANSWER_TEXT,
    build_messages,
    extract_citation_numbers,
)
from app.repositories.conversation_repo import ConversationRepository
from app.repositories.document_repo import DocumentRepository
from app.repositories.message_repo import MessageRepository
from app.schemas.chat import (
    DEFAULT_CONVERSATION_TITLE,
    DoneData,
    ErrorData,
    MessageCreate,
    SourceRead,
    TokenData,
)
from app.schemas.search import RetrievedChunk
from app.services.search_service import SearchService
from app.services.web_search_service import WebSearchService

logger = logging.getLogger(__name__)

STREAM_ERROR_TEXT = "Не удалось получить ответ. Попробуйте задать вопрос ещё раз."

# Сколько прежних вопросов диалога отдаём поиску по интернету как контекст
MAX_WEB_CONTEXT_QUESTIONS = 2

# Длина заголовка диалога, названного по первому вопросу
AUTO_TITLE_LENGTH = 60


def title_from_question(question: str) -> str:
    """Заголовок диалога из вопроса: в одну строку и не длиннее AUTO_TITLE_LENGTH."""
    text = " ".join(question.split())
    if len(text) <= AUTO_TITLE_LENGTH:
        return text
    return text[: AUTO_TITLE_LENGTH - 1].rstrip() + "…"


class ChatService:
    def __init__(
        self,
        session: AsyncSession,
        search: SearchService,
        chat: ChatClient,
        web: WebSearchService,
        session_factory: async_sessionmaker[AsyncSession] = AsyncSessionLocal,
    ) -> None:
        self.session = session
        self.conversations = ConversationRepository(session)
        self.messages = MessageRepository(session)
        self.documents = DocumentRepository(session)
        self.search = search
        self.chat = chat
        self.web = web
        # ответ сохраняется в собственной короткой сессии (см. _save_answer)
        self.session_factory = session_factory

    async def prepare(
        self, owner_id: UUID, conversation_id: UUID, payload: MessageCreate
    ) -> list[ChatMessage]:
        """Всё, что может закончиться обычной HTTP-ошибкой, до начала потока.

        Внутри потока HTTPException не сработает: статус 200 уже отправлен.
        Поэтому проверки и сохранение вопроса идут здесь, а поток начинается,
        только если всё в порядке. Возвращает недавнюю переписку для промпта.
        """
        conversation = await self.conversations.get_by_id(conversation_id, owner_id)
        if conversation is None:
            # чужой диалог неотличим от несуществующего: не раскрываем его наличие
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Диалог не найден"
            )

        if payload.document_ids is not None:
            owned = await self.documents.get_owned_ids(payload.document_ids, owner_id)
            if owned != set(payload.document_ids):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Документ не найден",
                )

        # история читается до сохранения вопроса: в промпт он попадёт отдельно
        history = await self.messages.last_messages(
            conversation.id, MAX_HISTORY_MESSAGES
        )
        await self.messages.create(conversation.id, MessageRole.USER, payload.question)
        if not history and conversation.title == DEFAULT_CONVERSATION_TITLE:
            # первый вопрос диалога без своего заголовка: называем диалог по нему,
            # иначе в списке диалогов все они выглядели бы одинаково
            conversation.title = title_from_question(payload.question)
        # вопрос сохраняется сразу: он не пропадёт, даже если ответ не получится
        await self.session.commit()
        return [
            ChatMessage(role=message.role.value, content=message.content)
            for message in history
        ]

    async def answer(
        self,
        owner_id: UUID,
        conversation_id: UUID,
        payload: MessageCreate,
        history: list[ChatMessage],
    ) -> AsyncIterator[ServerSentEvent]:
        """Поток событий: sources -> token... -> done (или error).

        Любой сбой внутри превращается в событие error: исключение из потока
        клиент не увидит, соединение просто оборвётся. Отмена (клиент ушёл)
        исключением Exception не является и проходит насквозь: в этом случае
        неполный ответ не сохраняется.
        """
        try:
            if payload.web_search:
                # поиску нужны прежние вопросы, чтобы понять короткое уточнение
                previous = [m.content for m in history if m.role == "user"]
                sources = await self.web.search(
                    payload.question, context=previous[-MAX_WEB_CONTEXT_QUESTIONS:]
                )
            else:
                sources = await self.search.search(
                    owner_id, payload.question, payload.document_ids
                )
            # Сессия запроса живёт до конца потока и держит соединение с БД.
            # Дальше мы только ждём LLM, поэтому отдаём соединение обратно
            await self.session.close()

            yield ServerSentEvent(
                event="sources",
                data=[
                    SourceRead(number=number, **chunk.model_dump())
                    for number, chunk in enumerate(sources, start=1)
                ],
            )

            parts: list[str] = []
            if sources:
                messages = build_messages(
                    payload.question, sources, history, web=payload.web_search
                )
                # aclosing: если читатель ушёл, поток LLM закрывается сразу,
                # а не когда до него доберётся сборщик мусора
                async with aclosing(self.chat.stream(messages)) as stream:
                    async for piece in stream:
                        parts.append(piece)
                        yield ServerSentEvent(event="token", data=TokenData(text=piece))
            else:
                # искать нечего: LLM не зовём, это дешевле и без выдумок
                no_answer = NO_WEB_ANSWER_TEXT if payload.web_search else NO_ANSWER_TEXT
                parts.append(no_answer)
                yield ServerSentEvent(event="token", data=TokenData(text=no_answer))

            answer = "".join(parts)
            if not answer.strip():
                raise ValueError("Модель вернула пустой ответ")

            numbers = extract_citation_numbers(answer, len(sources))
            message_id = await self._save_answer(
                conversation_id, answer, sources, numbers
            )
            yield ServerSentEvent(
                event="done", data=DoneData(message_id=message_id, citations=numbers)
            )
        except Exception:
            logger.exception(
                "Не удалось ответить на вопрос, диалог %s", conversation_id
            )
            yield ServerSentEvent(
                event="error", data=ErrorData(detail=STREAM_ERROR_TEXT)
            )

    async def _save_answer(
        self,
        conversation_id: UUID,
        answer: str,
        sources: list[RetrievedChunk],
        numbers: list[int],
    ) -> UUID:
        try:
            return await self._store_answer(
                conversation_id, answer, sources, numbers, keep_links=True
            )
        except IntegrityError:
            # Пока модель отвечала, документ могли удалить или переиндексировать:
            # его чанки исчезли, и ссылка на них нарушает внешний ключ. Ответ
            # важнее ссылок, а копия источника в цитате всё равно сохранится
            logger.warning(
                "Источник пропал во время ответа, сохраняю цитаты без ссылок"
            )
            return await self._store_answer(
                conversation_id, answer, sources, numbers, keep_links=False
            )

    async def _store_answer(
        self,
        conversation_id: UUID,
        answer: str,
        sources: list[RetrievedChunk],
        numbers: list[int],
        keep_links: bool,
    ) -> UUID:
        # собственная сессия: сессия запроса к этому моменту уже закрыта
        async with self.session_factory() as session:
            repo = MessageRepository(session)
            message = await repo.create(conversation_id, MessageRole.ASSISTANT, answer)
            await repo.add_citations(
                [
                    CitationORM(
                        message_id=message.id,
                        number=number,
                        # копия источника: переиндексация не сломает историю
                        document_title=sources[number - 1].document_title,
                        heading_path=sources[number - 1].heading_path,
                        content=sources[number - 1].content,
                        document_id=sources[number - 1].document_id
                        if keep_links
                        else None,
                        chunk_id=sources[number - 1].chunk_id if keep_links else None,
                        # адрес страницы не ссылка на строку БД: внешнего ключа нет,
                        # и он остаётся даже в запасном сохранении без ссылок
                        url=sources[number - 1].url,
                    )
                    for number in numbers
                ]
            )
            await session.commit()
            return message.id
