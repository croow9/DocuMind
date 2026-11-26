from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import settings
from app.core.openai_client import build_openai_client
from app.ingestion.embedder import OpenAIEmbedder
from app.rag.llm import OpenAIChat
from app.rag.query_rewrite import (
    MAX_QUERY_LENGTH,
    REWRITE_INSTRUCTIONS,
    FakeQueryRewriter,
    OpenAIQueryRewriter,
    build_query_rewriter,
)
from app.rag.web_search import OpenAIPageFinder


def make_rewriter(reply: object) -> tuple[OpenAIQueryRewriter, AsyncMock]:
    client = AsyncMock()
    if isinstance(reply, Exception):
        client.chat.completions.create = AsyncMock(side_effect=reply)
    else:
        message = SimpleNamespace(content=reply)
        client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(choices=[SimpleNamespace(message=message)])
        )
    return OpenAIQueryRewriter(client=client), client.chat.completions.create


async def test_question_is_sent_with_instructions_and_zero_temperature() -> None:
    rewriter, create = make_rewriter("BPMN gateway types")

    query = await rewriter.rewrite("Какие бывают типы шлюзов в BPMN?")

    assert query == "BPMN gateway types"
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == settings.chat_model
    assert kwargs["messages"] == [
        {"role": "system", "content": REWRITE_INSTRUCTIONS},
        {"role": "user", "content": "Какие бывают типы шлюзов в BPMN?"},
    ]
    # один вопрос — один запрос, иначе кэш поиска почти не будет срабатывать
    assert kwargs["temperature"] == 0


async def test_previous_questions_are_given_as_context() -> None:
    rewriter, create = make_rewriter("FastAPI dependency yield")

    await rewriter.rewrite("А как с yield?", ["Как объявить Depends в FastAPI?"])

    prompt = create.await_args.kwargs["messages"][1]["content"]
    assert "- Как объявить Depends в FastAPI?" in prompt
    assert prompt.endswith("Текущий вопрос: А как с yield?")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('"BPMN gateway types"', "BPMN gateway types"),
        ("«BPMN gateway types»?", "BPMN gateway types"),
        (
            "\n\n  BPMN gateway types  \nпояснение во второй строке",
            "BPMN gateway types",
        ),
        ("x" * (MAX_QUERY_LENGTH + 50), "x" * MAX_QUERY_LENGTH),
        # слова-паразиты уходят, регистр не важен, остальное и порядок на месте
        ("pydantic field validator syntax", "pydantic field validator"),
        ("FastAPI dependency yield Usage", "FastAPI dependency yield"),
        ("Alembic migration example guide", "Alembic migration"),
        ("pydantic   field  validator syntax", "pydantic field validator"),
        ("tutorial Kafka consumer group overview", "Kafka consumer group"),
        # части слов и слова с чем-то ещё не трогаем
        ("exampleapp usage-guide patterns", "exampleapp usage-guide patterns"),
        ("BPMN gateway types", "BPMN gateway types"),
        # без паразитов остаётся меньше двух слов — оставляем как было
        ("Kafka tutorial", "Kafka tutorial"),
        ("syntax", "syntax"),
    ],
)
async def test_model_reply_is_reduced_to_one_clean_line(
    raw: str, expected: str
) -> None:
    rewriter, _ = make_rewriter(raw)

    assert await rewriter.rewrite("вопрос") == expected


@pytest.mark.parametrize("raw", ["", "   \n ", None, '""', "?"])
async def test_empty_reply_falls_back_to_the_question(raw: str | None) -> None:
    rewriter, _ = make_rewriter(raw)

    assert await rewriter.rewrite("исходный вопрос") == "исходный вопрос"


async def test_failure_of_the_model_does_not_cost_the_user_the_answer() -> None:
    rewriter, _ = make_rewriter(RuntimeError("OpenAI недоступен"))

    assert await rewriter.rewrite("исходный вопрос") == "исходный вопрос"


async def test_fake_rewriter_returns_question_or_given_reply_and_remembers_calls() -> (
    None
):
    plain, scripted = FakeQueryRewriter(), FakeQueryRewriter(reply="query")

    assert await plain.rewrite("вопрос", ["раньше"]) == "вопрос"
    assert await scripted.rewrite("вопрос") == "query"
    assert plain.calls == [("вопрос", ["раньше"])]


def test_rewriter_is_chosen_by_key_presence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "")
    assert isinstance(build_query_rewriter(), FakeQueryRewriter)

    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    assert isinstance(build_query_rewriter(), OpenAIQueryRewriter)


# --- таймаут OpenAI: по умолчанию SDK ждёт 10 минут


def test_every_openai_client_of_the_app_has_the_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "openai_timeout_seconds", 12.5)

    clients = [
        build_openai_client(),
        OpenAIEmbedder()._client,
        OpenAIChat()._client,
        OpenAIPageFinder()._client,
        OpenAIQueryRewriter()._client,
    ]

    assert [client.timeout for client in clients] == [12.5] * len(clients)


def test_instruction_forbids_filler_words_that_derail_the_search() -> None:
    # «pydantic field validator usage» уводил поиск на страницы FastAPI, а без
    # «usage» находил документацию Pydantic: слова-паразиты запрещены явно
    assert "usage" in REWRITE_INSTRUCTIONS
