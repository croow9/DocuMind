from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

from app.core.config import settings
from app.rag.llm import (
    ChatMessage,
    FakeChat,
    OpenAIChat,
    build_chat_client,
)

MESSAGES = [
    ChatMessage(role="system", content="Отвечай по источникам."),
    ChatMessage(role="user", content="Что такое RRF?"),
]


def make_chunk(content: str | None, *, with_choice: bool = True) -> ChatCompletionChunk:
    """Чанк в том же виде, в каком его отдаёт стриминг OpenAI."""
    choices = (
        [Choice(index=0, delta=ChoiceDelta(content=content))] if with_choice else []
    )
    return ChatCompletionChunk(
        id="chunk-1",
        choices=choices,
        created=0,
        model="test-model",
        object="chat.completion.chunk",
    )


class FakeOpenAIStream:
    """Замена AsyncStream: умеет async with и async for, помнит, что её закрыли."""

    def __init__(self, chunks: list[ChatCompletionChunk]) -> None:
        self.chunks = chunks
        self.closed = False

    async def __aenter__(self) -> "FakeOpenAIStream":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.closed = True

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[ChatCompletionChunk]:
        for chunk in self.chunks:
            yield chunk


def make_client(stream: FakeOpenAIStream) -> AsyncMock:
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value=stream)
    return client


async def test_openai_chat_yields_only_pieces_with_text() -> None:
    stream = FakeOpenAIStream(
        [
            make_chunk(None),  # первый чанк: пришла только роль
            make_chunk("Привет"),
            make_chunk(""),
            make_chunk(", мир"),
            make_chunk(None),  # последний чанк: причина остановки, текста нет
            make_chunk(None, with_choice=False),  # служебный чанк без choices
        ]
    )

    pieces = [p async for p in OpenAIChat(client=make_client(stream)).stream(MESSAGES)]

    assert pieces == ["Привет", ", мир"]


async def test_openai_chat_sends_model_messages_and_stream_flag() -> None:
    client = make_client(FakeOpenAIStream([]))

    _ = [p async for p in OpenAIChat(client=client).stream(MESSAGES)]

    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["stream"] is True
    assert kwargs["model"] == settings.chat_model
    assert kwargs["max_completion_tokens"] == settings.chat_max_tokens
    assert kwargs["messages"] == [
        {"role": "system", "content": "Отвечай по источникам."},
        {"role": "user", "content": "Что такое RRF?"},
    ]


async def test_openai_chat_closes_connection_when_reader_stops_early() -> None:
    stream = FakeOpenAIStream([make_chunk("раз"), make_chunk("два"), make_chunk("три")])
    pieces = OpenAIChat(client=make_client(stream)).stream(MESSAGES)

    assert await anext(pieces) == "раз"
    await pieces.aclose()  # type: ignore[attr-defined]

    assert stream.closed


async def test_fake_chat_streams_pieces_that_join_into_reply() -> None:
    chat = FakeChat(reply="Первый  второй третий [1]")

    pieces = [p async for p in chat.stream(MESSAGES)]

    assert len(pieces) > 1
    assert "".join(pieces) == "Первый  второй третий [1]"
    # фейк помнит, с чем его спросили: по этому тесты проверяют промпт
    assert chat.received == [MESSAGES]


@pytest.mark.parametrize(
    ("api_key", "expected_name"),
    [("", "FakeChat"), ("sk-test", "OpenAIChat")],
)
def test_build_chat_client_depends_on_api_key(
    monkeypatch: pytest.MonkeyPatch, api_key: str, expected_name: str
) -> None:
    monkeypatch.setattr(settings, "openai_api_key", api_key)

    assert type(build_chat_client()).__name__ == expected_name
