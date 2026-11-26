import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.core.config import settings
from app.rag.web_search import (
    TAVILY_MAX_RESULTS,
    TAVILY_SEARCH_URL,
    FakePageFinder,
    FoundPage,
    OpenAIPageFinder,
    SearchBackendError,
    TavilyPageFinder,
    build_page_finder,
)

DOMAINS = ["fastapi.tiangolo.com", "docs.python.org"]


def search_call(*urls: str) -> SimpleNamespace:
    sources = [SimpleNamespace(url=url) for url in urls]
    return SimpleNamespace(
        type="web_search_call", action=SimpleNamespace(type="search", sources=sources)
    )


def message(*urls: str) -> SimpleNamespace:
    annotations = [SimpleNamespace(type="url_citation", url=url) for url in urls]
    # посторонняя аннотация (не ссылка) не должна попасть в результат
    annotations.append(
        SimpleNamespace(type="file_citation", url="https://not-a-link.test/")
    )
    part = SimpleNamespace(type="output_text", annotations=annotations)
    return SimpleNamespace(type="message", content=[part])


def make_finder(*output: SimpleNamespace) -> tuple[OpenAIPageFinder, AsyncMock]:
    client = AsyncMock()
    client.responses.create = AsyncMock(
        return_value=SimpleNamespace(output=list(output))
    )
    return OpenAIPageFinder(client=client), client.responses.create


async def test_cited_pages_come_first_then_the_rest_without_repeats_and_tracking() -> (
    None
):
    finder, _ = make_finder(
        SimpleNamespace(type="reasoning"),
        search_call(
            "https://fastapi.tiangolo.com/a/?utm_source=openai",
            "https://fastapi.tiangolo.com/b/",
            "https://fastapi.tiangolo.com/c/#frag",
        ),
        message("https://fastapi.tiangolo.com/c/?utm_source=openai"),
    )

    urls = [page.url for page in await finder.find("как?", DOMAINS)]

    assert urls == [
        "https://fastapi.tiangolo.com/c/",  # процитирована — значит, важнее
        "https://fastapi.tiangolo.com/a/",
        "https://fastapi.tiangolo.com/b/",
    ]


async def test_actions_without_sources_are_ignored() -> None:
    open_page = SimpleNamespace(
        type="web_search_call",
        action=SimpleNamespace(type="open_page", url="https://x.test/"),
    )
    finder, _ = make_finder(open_page, search_call("https://docs.python.org/3/"))

    assert await finder.find("как?", DOMAINS) == [
        FoundPage("https://docs.python.org/3/")
    ]


async def test_request_forces_search_and_names_the_sites() -> None:
    finder, create = make_finder(search_call())

    await finder.find("Как объявить зависимость?", DOMAINS)

    kwargs = create.await_args.kwargs
    assert kwargs["input"] == "Как объявить зависимость?"
    assert kwargs["model"] == settings.web_search_model
    assert kwargs["tools"] == [{"type": "web_search"}]
    # без «required» модель могла бы ответить по памяти и ничего не искать
    assert kwargs["tool_choice"] == "required"
    assert "web_search_call.action.sources" in kwargs["include"]
    assert "fastapi.tiangolo.com" in kwargs["instructions"]
    assert "docs.python.org" in kwargs["instructions"]


async def test_fake_finder_returns_given_urls_and_remembers_questions() -> None:
    finder = FakePageFinder(["https://docs.python.org/3/"])

    assert await finder.find("как?", DOMAINS) == [
        FoundPage("https://docs.python.org/3/")
    ]
    assert finder.queries == ["как?"]


def test_finder_is_chosen_by_the_keys_that_are_set(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    client = httpx.AsyncClient()
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "tavily_api_key", "")
    assert isinstance(build_page_finder(client), FakePageFinder)

    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    assert isinstance(build_page_finder(client), OpenAIPageFinder)

    # Tavily важнее: он умеет фильтровать по доменам, поиск OpenAI на длинных списках нет
    monkeypatch.setattr(settings, "tavily_api_key", "tvly-test")
    assert isinstance(build_page_finder(client), TavilyPageFinder)


# --- Tavily

KEY = "tvly-secret-key"


def make_tavily(handler) -> TavilyPageFinder:  # type: ignore[no-untyped-def]
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TavilyPageFinder(client, KEY)


def tavily_answer(*results: object) -> httpx.Response:
    return httpx.Response(200, json={"results": list(results)})


async def test_tavily_request_restricts_search_to_profile_domains_and_asks_for_markdown() -> (
    None
):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return tavily_answer()

    await make_tavily(handler).find("Как объявить зависимость?", DOMAINS)

    [request] = seen
    assert str(request.url) == TAVILY_SEARCH_URL
    assert request.headers["authorization"] == f"Bearer {KEY}"
    body = json.loads(request.content)
    assert body["query"] == "Как объявить зависимость?"
    assert body["include_domains"] == DOMAINS
    # «prefer» пускает в выдачу и чужие сайты: проверено на живом запросе
    assert body["include_domains_mode"] == "restrict"
    assert body["include_raw_content"] == "markdown"
    assert body["search_depth"] == "basic"
    assert body["max_results"] == TAVILY_MAX_RESULTS


async def test_tavily_results_become_pages_with_titles_texts_and_snippets() -> None:
    finder = make_tavily(
        lambda request: tavily_answer(
            {
                "url": "https://fastapi.tiangolo.com/tutorial/dependencies/",
                "title": "Зависимости",
                "raw_content": "# Зависимости\n\nтекст",
                "content": "отрывок",
            }
        )
    )

    assert await finder.find("вопрос", DOMAINS) == [
        FoundPage(
            "https://fastapi.tiangolo.com/tutorial/dependencies/",
            title="Зависимости",
            text="# Зависимости\n\nтекст",
            snippet="отрывок",
        )
    ]


async def test_http_address_is_upgraded_and_tracking_and_anchor_are_dropped() -> None:
    finder = make_tavily(
        lambda request: tavily_answer(
            {
                "url": "http://www.omg.org/spec/BPMN/?utm_source=x#top",
                "raw_content": "t",
            }
        )
    )

    [page] = await finder.find("вопрос", ["omg.org"])

    assert page.url == "https://www.omg.org/spec/BPMN/"


async def test_results_of_one_page_with_different_anchors_are_merged() -> None:
    page = "https://docs.pydantic.dev/latest/concepts/validators"
    finder = make_tavily(
        lambda request: tavily_answer(
            {
                "url": f"{page}#before",
                "title": "Validators",
                "raw_content": "короткий",
                "content": "про before",
            },
            {"url": "https://docs.pydantic.dev/latest/other/", "raw_content": "другая"},
            {
                "url": f"{page}#wrap",
                "raw_content": "самый длинный текст",
                "content": "про wrap",
            },
            {"url": f"{page}#after", "raw_content": "средний", "content": "про before"},
        )
    )

    pages = await finder.find("вопрос", ["docs.pydantic.dev"])

    # порядок по первому появлению; текст самый длинный; отрывки всех якорей, без повторов
    assert [p.url for p in pages] == [page, "https://docs.pydantic.dev/latest/other/"]
    assert pages[0] == FoundPage(
        page,
        title="Validators",
        text="самый длинный текст",
        snippet="про before\n\nпро wrap",
    )


async def test_unusable_results_are_skipped_and_blank_text_means_no_text() -> None:
    finder = make_tavily(
        lambda request: tavily_answer(
            "не словарь",
            {"title": "без адреса"},
            {"url": 123},
            {
                "url": "https://fastapi.tiangolo.com/a/",
                "raw_content": "   ",
                "content": None,
            },
        )
    )

    assert await finder.find("вопрос", DOMAINS) == [
        FoundPage("https://fastapi.tiangolo.com/a/")
    ]


@pytest.mark.parametrize("status", [401, 429, 432, 500])
async def test_tavily_refusal_becomes_backend_error_without_leaking_the_key(
    status: int,
) -> None:
    finder = make_tavily(lambda request: httpx.Response(status, text=f"плохо {KEY}"))

    with pytest.raises(SearchBackendError, match=str(status)) as exc_info:
        await finder.find("вопрос", DOMAINS)

    assert KEY not in str(exc_info.value)


async def test_network_failure_becomes_backend_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("нет связи")

    with pytest.raises(SearchBackendError, match="сеть"):
        await make_tavily(handler).find("вопрос", DOMAINS)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="это не JSON"),
        httpx.Response(200, json={"нет": "results"}),
        httpx.Response(200, json={"results": "не список"}),
        httpx.Response(200, json=["не", "словарь"]),
    ],
)
async def test_unexpected_tavily_body_becomes_backend_error(
    response: httpx.Response,
) -> None:
    with pytest.raises(SearchBackendError, match="неожиданном"):
        await make_tavily(lambda request: response).find("вопрос", DOMAINS)
