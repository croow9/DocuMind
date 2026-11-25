from unittest.mock import AsyncMock

from app.ingestion.embedder import EmbeddingClient
from app.ingestion.web_pages import PageFetchError
from app.rag.web_search import FakePageFinder
from app.services.web_search_service import (
    MAX_CHUNKS_PER_PAGE,
    WebSearchService,
    build_http_client,
)

DOMAINS = ["fastapi.tiangolo.com"]
GOOD = "https://fastapi.tiangolo.com/tutorial/dependencies/"
OTHER = "https://fastapi.tiangolo.com/tutorial/security/"
EVIL = "https://evil.com/fastapi.tiangolo.com"


def page(title: str, **sections: str) -> str:
    """HTML страницы: заголовок и разделы «название раздела -> текст»."""
    body = "".join(f"<h2>{name}</h2><p>{text}</p>" for name, text in sections.items())
    return f"<title>{title}</title><article><h1>{title}</h1>{body}</article>"


class VectorEmbedder(EmbeddingClient):
    """Вектор по тексту из заданной таблицы: близость смыслов задаём сами."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self.vectors.get(text, [0.0, 0.0, 1.0]) for text in texts]


def make_service(
    pages: dict[str, str | Exception],
    found: list[str],
    embedder: EmbeddingClient | None = None,
    max_pages: int = 4,
) -> tuple[WebSearchService, FakePageFinder, AsyncMock]:
    finder = FakePageFinder(found)
    fetcher = AsyncMock()

    async def fetch(url: str) -> str:
        result = pages[url]
        if isinstance(result, Exception):
            raise result
        return result

    fetcher.fetch.side_effect = fetch
    service = WebSearchService(
        finder, fetcher, embedder or VectorEmbedder({}), DOMAINS, max_pages
    )
    return service, finder, fetcher.fetch


async def test_only_allowed_unique_urls_are_downloaded_up_to_the_limit() -> None:
    third = "https://fastapi.tiangolo.com/third/"
    service, _, fetch = make_service(
        {
            GOOD: page("A", s="текст"),
            OTHER: page("B", s="текст"),
            third: page("C", s="текст"),
        },
        found=[EVIL, GOOD, GOOD, OTHER, third],
        max_pages=2,
    )

    await service.search("вопрос")

    # чужой сайт отброшен, повтор склеен, лишнее сверх лимита не качается
    assert [call.args[0] for call in fetch.await_args_list] == [GOOD, OTHER]


async def test_nearest_fragments_win_and_carry_page_title_path_and_url() -> None:
    embedder = VectorEmbedder(
        {
            "вопрос": [1.0, 0.0, 0.0],
            "про зависимости": [0.9, 0.1, 0.0],
            "про безопасность": [0.1, 0.9, 0.0],
        }
    )
    service, _, _ = make_service(
        {
            GOOD: page("Зависимости", **{"Первые шаги": "про зависимости"}),
            OTHER: page("Безопасность", **{"Токены": "про безопасность"}),
        },
        found=[OTHER, GOOD],
        embedder=embedder,
    )

    [best] = await service.search("вопрос", top_k=1)

    assert best.content == "про зависимости"
    assert best.document_title == "Зависимости"
    assert best.heading_path == "Зависимости > Первые шаги"
    assert best.url == GOOD
    assert best.chunk_id is None and best.document_id is None


async def test_question_and_all_fragments_are_embedded_in_one_request() -> None:
    embedder = VectorEmbedder({})
    service, _, _ = make_service(
        {GOOD: page("A", x="раз", y="два")}, found=[GOOD], embedder=embedder
    )

    await service.search("вопрос")

    assert embedder.calls == [["вопрос", "раз", "два"]]


async def test_broken_page_is_skipped_and_the_rest_is_used() -> None:
    service, _, _ = make_service(
        {GOOD: PageFetchError("HTTP 500"), OTHER: page("B", s="текст")},
        found=[GOOD, OTHER],
    )

    results = await service.search("вопрос")

    assert [r.url for r in results] == [OTHER]


async def test_no_pages_means_no_sources_and_no_embedding_call() -> None:
    embedder = VectorEmbedder({})
    service, _, _ = make_service(
        {GOOD: PageFetchError("HTTP 404")}, found=[GOOD, EVIL], embedder=embedder
    )

    assert await service.search("вопрос") == []
    assert embedder.calls == []


async def test_blank_question_or_empty_allowlist_does_not_even_search() -> None:
    service, finder, _ = make_service({}, found=[GOOD])

    assert await service.search("   ") == []

    service.allowed_domains = []
    assert await service.search("вопрос") == []
    assert finder.questions == []


async def test_huge_page_is_cut_to_the_chunk_limit() -> None:
    sections = {f"Раздел {i}": f"текст {i}" for i in range(MAX_CHUNKS_PER_PAGE + 20)}
    embedder = VectorEmbedder({})
    service, _, _ = make_service(
        {GOOD: page("Справочник", **sections)}, [GOOD], embedder
    )

    await service.search("вопрос")

    # вопрос + не больше MAX_CHUNKS_PER_PAGE кусков
    assert len(embedder.calls[0]) == 1 + MAX_CHUNKS_PER_PAGE


async def test_pages_in_preferred_languages_and_current_version_are_downloaded_first() -> (
    None
):
    # страницы разные (a..e): это про порядок, а не про склейку вариантов одной страницы
    urls = {
        "ja": "https://docs.python.org/ja/3/library/a.html",
        "old": "https://docs.python.org/3.7/library/b.html",
        "pl_old": "https://docs.python.org/pl/3.7/library/c.html",
        "latest": "https://docs.python.org/3/library/d.html",
        "ru": "https://docs.python.org/ru/3/library/e.html",
    }
    service = WebSearchService(
        FakePageFinder(list(urls.values())),
        fetch := AsyncMock(),
        VectorEmbedder({}),
        ["docs.python.org"],
        max_pages=3,
    )
    fetch.fetch.return_value = page("asyncio", s="текст")

    await service.search("вопрос")

    fetched = [call.args[0] for call in fetch.fetch.await_args_list]
    # впереди «нужный язык, актуальная версия»; закреплённая версия хуже,
    # чужой язык ещё хуже; при равных оценках порядок поиска сохраняется
    assert fetched == [urls["latest"], urls["ru"], urls["old"]]


async def test_foreign_language_page_is_still_used_when_nothing_better_exists() -> None:
    ja = "https://fastapi.tiangolo.com/ja/tutorial/"
    service, _, fetch = make_service({ja: page("Страница", s="текст")}, found=[ja])

    results = await service.search("вопрос")

    assert [r.url for r in results] == [ja]


async def test_http_client_of_the_app_can_be_built_with_an_ascii_user_agent() -> None:
    # с кириллицей в User-Agent приложение не запускалось: httpx кодирует
    # заголовки в ASCII и падает прямо при создании клиента
    client = build_http_client()
    try:
        assert client.headers["user-agent"].isascii()
    finally:
        await client.aclose()


async def test_translations_and_versions_of_one_page_take_one_place() -> None:
    task = "https://docs.python.org/{}/library/asyncio-task.html"
    index = "https://docs.python.org/3/library/asyncio-api-index.html"
    found = [task.format("3.13"), task.format("3"), task.format("3.11"), index]
    service = WebSearchService(
        FakePageFinder(found),
        fetch := AsyncMock(),
        VectorEmbedder({}),
        ["docs.python.org"],
        max_pages=4,
    )
    fetch.fetch.return_value = page("asyncio", s="текст")

    await service.search("вопрос")

    # из трёх вариантов одной страницы взят лучший (/3/), место осталось другой
    assert [call.args[0] for call in fetch.fetch.await_args_list] == [
        task.format("3"),
        index,
    ]


async def test_previous_questions_are_passed_to_the_finder() -> None:
    service, finder, _ = make_service({GOOD: page("A", s="текст")}, found=[GOOD])

    await service.search("а как с yield?", context=["как объявить Depends в FastAPI?"])

    assert finder.questions == ["а как с yield?"]
    assert finder.contexts == [["как объявить Depends в FastAPI?"]]
