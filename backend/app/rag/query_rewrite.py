import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.openai_client import build_openai_client

logger = logging.getLogger(__name__)

# Запрос из нескольких слов: лимит защищает от рассуждений вместо запроса
REWRITE_MAX_TOKENS = 60
MAX_QUERY_LENGTH = 200

REWRITE_INSTRUCTIONS = (
    "Ты готовишь поисковый запрос для поиска по официальной документации и "
    "спецификациям. Преврати вопрос пользователя в короткий запрос на "
    "английском из 3-8 слов: назови продукт, технологию или стандарт и суть "
    "вопроса. Не добавляй общих слов вроде usage, example, guide, tutorial, "
    "how to: они уводят поиск на посторонние страницы. Если вопрос только "
    "уточняет предыдущие, возьми продукт и тему из них. Ответь одним запросом: "
    "без кавычек, пояснений и вопросительного знака."
)


class QueryRewriter(ABC):
    """Вопрос пользователя -> запрос для поиска по документации.

    Документация в основном английская, а вопросы бывают русскими и короткими
    («а как с yield?»): для поиска их нужно переформулировать.
    """

    @abstractmethod
    async def rewrite(self, question: str, context: Sequence[str] = ()) -> str:
        """context — прежние вопросы диалога, от старых к новым."""
        raise NotImplementedError


# Слова, которыми модель разбавляет запрос, хотя мы просили не делать этого. Для
# поиска они вредны: «pydantic field validator syntax» находил страницы FastAPI,
# а «pydantic field validator» — документацию Pydantic. Список взят из замеров
FILLER_WORDS = frozenset(
    {"usage", "syntax", "example", "examples", "guide", "tutorial", "overview"}
)
MIN_QUERY_WORDS = 2


def _drop_filler(query: str) -> str:
    words = [word for word in query.split() if word.lower() not in FILLER_WORDS]
    # от запроса «tutorial» ничего не остаётся: лучше с лишним словом, чем без темы
    return " ".join(words) if len(words) >= MIN_QUERY_WORDS else query


def _clean_query(raw: str | None, fallback: str) -> str:
    """Первая непустая строка ответа модели без кавычек и слов-паразитов.

    Пусто — исходный вопрос.
    """
    line = next((line.strip() for line in (raw or "").splitlines() if line.strip()), "")
    # кавычки и «?» снимаются одним проходом: иначе «запрос»? оставил бы кавычку
    line = _drop_filler(line.strip("\"'`«»? "))
    return line[:MAX_QUERY_LENGTH] if line else fallback


class OpenAIQueryRewriter(QueryRewriter):
    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        self._client = client or build_openai_client()

    async def rewrite(self, question: str, context: Sequence[str] = ()) -> str:
        prompt = question
        if context:
            previous = "\n".join(f"- {text}" for text in context)
            prompt = (
                f"Предыдущие вопросы пользователя в этом диалоге:\n{previous}\n\n"
                f"Текущий вопрос: {question}"
            )
        try:
            response = await self._client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {"role": "system", "content": REWRITE_INSTRUCTIONS},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=REWRITE_MAX_TOKENS,
                # один и тот же вопрос должен давать один и тот же запрос: от этого
                # зависит, попадём ли мы в кэш поиска
                temperature=0,
            )
        except Exception:
            # без перефразирования поиск хуже, но он работает: сбой этого шага не
            # должен отнимать у пользователя ответ
            logger.warning(
                "Не удалось перефразировать вопрос, ищу по исходному", exc_info=True
            )
            return question
        return _clean_query(response.choices[0].message.content, question)


class FakeQueryRewriter(QueryRewriter):
    """Без сети: отдаёт заданный запрос или сам вопрос. Запоминает обращения."""

    def __init__(self, reply: str | None = None) -> None:
        self.reply = reply
        self.calls: list[tuple[str, list[str]]] = []

    async def rewrite(self, question: str, context: Sequence[str] = ()) -> str:
        self.calls.append((question, list(context)))
        return question if self.reply is None else self.reply


def build_query_rewriter() -> QueryRewriter:
    if settings.openai_api_key:
        return OpenAIQueryRewriter()
    return FakeQueryRewriter()
