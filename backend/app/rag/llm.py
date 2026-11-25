import asyncio
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Literal, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict

from app.core.config import settings

# Пауза между кусочками у фейка: без неё весь «поток» прилетает мгновенно, и в
# curl -N не видно, что ответ идёт частями
FAKE_DELAY_SECONDS = 0.05
FAKE_REPLY = "Это тестовый ответ без обращения к LLM. Он ссылается на источник [1]."


class ChatMessage(BaseModel):
    """Одно сообщение для LLM: чьё оно и что в нём написано."""

    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class ChatClient(ABC):
    """Собеседник-LLM: список сообщений -> ответ, который приходит кусочками.

    Поток объявлен как AsyncGenerator, а не AsyncIterator: у него есть aclose(),
    которым читатель закрывает поток, если ответ дочитывать не нужно.
    """

    @abstractmethod
    def stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str]:
        raise NotImplementedError


class OpenAIChat(ChatClient):
    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        self._client = client or AsyncOpenAI(api_key=settings.openai_api_key)

    async def stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str]:
        response = await self._client.chat.completions.create(
            model=settings.chat_model,
            # SDK различает типы сообщений по ролям, а у нас роль строка: сообщаем
            # проверке типов, что словари подходят
            messages=cast(
                "list[ChatCompletionMessageParam]",
                [{"role": m.role, "content": m.content} for m in messages],
            ),
            max_completion_tokens=settings.chat_max_tokens,
            stream=True,
        )
        # async with закрывает соединение с OpenAI, даже если читатель бросил
        # поток на полуслове (клиент оборвал связь)
        async with response:
            async for chunk in response:
                # последний служебный чанк бывает без choices
                if not chunk.choices:
                    continue
                # у первого чанка (там только роль) и у последнего (там причина
                # остановки) текста нет: content равен None или пустой строке
                text = chunk.choices[0].delta.content
                if text:
                    yield text


class FakeChat(ChatClient):
    """Ответ из заготовленного текста: без сети, ключа и денег.

    Для разработки и тестов. Запоминает, с какими сообщениями его спросили, чтобы
    тест мог проверить промпт или убедиться, что LLM вообще не вызывали.
    """

    def __init__(self, reply: str = FAKE_REPLY, delay_seconds: float = 0.0) -> None:
        self.reply = reply
        self.delay_seconds = delay_seconds
        self.received: list[list[ChatMessage]] = []

    async def stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str]:
        self.received.append(messages)
        # по слову вместе с пробелами после него: склейка кусочков равна reply
        for piece in re.findall(r"\s*\S+\s*", self.reply):
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)
            yield piece


def build_chat_client() -> ChatClient:
    """Настоящий API при наличии ключа, фейк с имитацией потока — без него."""
    if settings.openai_api_key:
        return OpenAIChat()
    return FakeChat(delay_seconds=FAKE_DELAY_SECONDS)
