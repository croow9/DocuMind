from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.core.config import settings
from app.rag.web_search import (
    FakePageFinder,
    OpenAIPageFinder,
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

    urls = await finder.find("как?", DOMAINS)

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

    assert await finder.find("как?", DOMAINS) == ["https://docs.python.org/3/"]


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

    assert await finder.find("как?", DOMAINS) == ["https://docs.python.org/3/"]
    assert finder.questions == ["как?"]


def test_finder_is_chosen_by_key_presence(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(settings, "openai_api_key", "")
    assert isinstance(build_page_finder(), FakePageFinder)

    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    assert isinstance(build_page_finder(), OpenAIPageFinder)


async def test_previous_questions_are_given_to_the_search_model_as_context() -> None:
    finder, create = make_finder(search_call())

    await finder.find("А как с yield?", DOMAINS, ["Как объявить Depends в FastAPI?"])

    prompt = create.await_args.kwargs["input"]
    assert "Как объявить Depends в FastAPI?" in prompt
    assert prompt.endswith("Текущий вопрос: А как с yield?")
