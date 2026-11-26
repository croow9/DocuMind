import json
from collections.abc import Sequence
from unittest.mock import AsyncMock

import pytest

from app.core.cache import InMemoryCache
from app.ingestion.embedder import EmbeddingClient
from app.ingestion.web_pages import FetchedPage, PageFetchError, html_to_markdown
from app.rag.query_rewrite import FakeQueryRewriter
from app.rag.web_profiles import WebProfile
from app.rag.web_search import FakePageFinder, FoundPage
from app.services.web_search_service import (
    MAX_CHUNKS_PER_PAGE,
    MAX_FULL_TEXT_CHARS,
    WebSearchService,
    build_http_client,
)

DOMAINS = ["fastapi.tiangolo.com"]
TTL = 3600
PROFILE = WebProfile(name="backend", title="Backend", domains=tuple(DOMAINS))
PYTHON_PROFILE = WebProfile(name="python", title="Python", domains=("docs.python.org",))
GOOD = "https://fastapi.tiangolo.com/tutorial/dependencies/"
OTHER = "https://fastapi.tiangolo.com/tutorial/security/"
EVIL = "https://evil.com/fastapi.tiangolo.com"


def fetched(html: str) -> FetchedPage:
    return FetchedPage(*html_to_markdown(html))


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
    found: list[str | FoundPage],
    embedder: EmbeddingClient | None = None,
    max_pages: int = 4,
    rewriter: FakeQueryRewriter | None = None,
) -> tuple[WebSearchService, FakePageFinder, AsyncMock]:
    finder = FakePageFinder(found)
    fetcher = AsyncMock()

    async def fetch(url: str, domains: list[str]) -> FetchedPage:
        result = pages[url]
        if isinstance(result, Exception):
            raise result
        return fetched(result)

    fetcher.fetch.side_effect = fetch
    service = WebSearchService(
        finder,
        fetcher,
        embedder or VectorEmbedder({}),
        rewriter or FakeQueryRewriter(),
        InMemoryCache(),
        max_pages,
        cache_ttl_seconds=TTL,
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

    await service.search("вопрос", PROFILE)

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

    [best] = await service.search("вопрос", PROFILE, top_k=1)

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

    await service.search("вопрос", PROFILE)

    assert embedder.calls == [["вопрос", "раз", "два"]]


async def test_broken_page_is_skipped_and_the_rest_is_used() -> None:
    service, _, _ = make_service(
        {GOOD: PageFetchError("HTTP 500"), OTHER: page("B", s="текст")},
        found=[GOOD, OTHER],
    )

    results = await service.search("вопрос", PROFILE)

    assert [r.url for r in results] == [OTHER]


async def test_no_pages_means_no_sources_and_no_embedding_call() -> None:
    embedder = VectorEmbedder({})
    service, _, _ = make_service(
        {GOOD: PageFetchError("HTTP 404")}, found=[GOOD, EVIL], embedder=embedder
    )

    assert await service.search("вопрос", PROFILE) == []
    assert embedder.calls == []


async def test_blank_question_does_not_even_search() -> None:
    service, finder, _ = make_service({}, found=[GOOD])

    assert await service.search("   ", PROFILE) == []
    assert finder.queries == []


async def test_finder_and_fetcher_get_the_domains_of_the_chosen_profile() -> None:
    service, finder, fetch = make_service({GOOD: page("A", s="текст")}, found=[GOOD])

    await service.search("вопрос", PROFILE)

    assert finder.domains == [DOMAINS]
    assert fetch.await_args.args == (GOOD, PROFILE.domains)


async def test_the_same_found_url_is_dropped_when_the_profile_does_not_allow_it() -> (
    None
):
    service, _, fetch = make_service({GOOD: page("A", s="текст")}, found=[GOOD])

    # страница с сайта из профиля backend, но спрашивали по профилю python
    assert await service.search("вопрос", PYTHON_PROFILE) == []
    fetch.assert_not_awaited()


async def test_huge_page_is_cut_to_the_chunk_limit() -> None:
    sections = {f"Раздел {i}": f"текст {i}" for i in range(MAX_CHUNKS_PER_PAGE + 20)}
    embedder = VectorEmbedder({})
    service, _, _ = make_service(
        {GOOD: page("Справочник", **sections)}, [GOOD], embedder
    )

    await service.search("вопрос", PROFILE)

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
        FakeQueryRewriter(),
        InMemoryCache(),
        max_pages=3,
        cache_ttl_seconds=TTL,
    )
    fetch.fetch.return_value = fetched(page("asyncio", s="текст"))

    await service.search("вопрос", PYTHON_PROFILE)

    downloaded = [call.args[0] for call in fetch.fetch.await_args_list]
    # впереди «нужный язык, актуальная версия»; закреплённая версия хуже,
    # чужой язык ещё хуже; при равных оценках порядок поиска сохраняется
    assert downloaded == [urls["latest"], urls["ru"], urls["old"]]


async def test_foreign_language_page_is_still_used_when_nothing_better_exists() -> None:
    ja = "https://fastapi.tiangolo.com/ja/tutorial/"
    service, _, fetch = make_service({ja: page("Страница", s="текст")}, found=[ja])

    results = await service.search("вопрос", PROFILE)

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
        FakeQueryRewriter(),
        InMemoryCache(),
        max_pages=4,
        cache_ttl_seconds=TTL,
    )
    fetch.fetch.return_value = fetched(page("asyncio", s="текст"))

    await service.search("вопрос", PYTHON_PROFILE)

    # из трёх вариантов одной страницы взят лучший (/3/), место осталось другой
    assert [call.args[0] for call in fetch.fetch.await_args_list] == [
        task.format("3"),
        index,
    ]


async def test_search_and_ranking_use_the_rewritten_query_not_the_raw_question() -> (
    None
):
    rewriter = FakeQueryRewriter(reply="fastapi yield dependency")
    embedder = VectorEmbedder({})
    service, finder, _ = make_service(
        {GOOD: page("A", s="текст")}, found=[GOOD], embedder=embedder, rewriter=rewriter
    )

    await service.search(
        "а как с yield?", PROFILE, context=["как объявить Depends в FastAPI?"]
    )

    assert rewriter.calls == [("а как с yield?", ["как объявить Depends в FastAPI?"])]
    assert finder.queries == ["fastapi yield dependency"]
    assert embedder.calls[0][0] == "fastapi yield dependency"


# --- страницы, которые поисковик отдал вместе с текстом

TEXT_PAGE = "# Зависимости\n\nтекст про Depends"


async def test_page_with_text_from_the_finder_is_not_downloaded() -> None:
    service, _, fetch = make_service(
        {}, found=[FoundPage(GOOD, title="Зависимости - FastAPI", text=TEXT_PAGE)]
    )

    [result] = await service.search("вопрос", PROFILE)

    fetch.assert_not_awaited()
    assert (result.url, result.document_title) == (GOOD, "Зависимости - FastAPI")
    assert result.heading_path == "Зависимости" and "Depends" in result.content


async def test_page_without_text_is_downloaded_with_its_title_from_the_finder() -> None:
    service, _, fetch = make_service(
        {GOOD: page("Название со страницы", s="текст")},
        found=[FoundPage(GOOD, title="Название от поисковика")],
    )

    [result] = await service.search("вопрос", PROFILE)

    fetch.assert_awaited_once()
    assert result.document_title == "Название от поисковика"


async def test_text_from_the_finder_is_cleaned_before_chunking() -> None:
    text = '[Skip to content](#x)\n\n# Заголовок[¶](#z "Permanent link")\n\nтекст'
    service, _, _ = make_service({}, found=[FoundPage(GOOD, text=text)])

    [result] = await service.search("вопрос", PROFILE)

    assert result.heading_path == "Заголовок"


async def test_text_of_a_foreign_site_is_never_used_even_if_the_finder_supplied_it() -> (
    None
):
    embedder = VectorEmbedder({})
    service, _, fetch = make_service(
        {}, found=[FoundPage(EVIL, text="# Чужое\n\nвредный текст")], embedder=embedder
    )

    assert await service.search("вопрос", PROFILE) == []
    assert embedder.calls == []
    fetch.assert_not_awaited()


async def test_huge_document_is_replaced_by_the_snippets_chosen_for_the_question() -> (
    None
):
    huge = "# Спецификация\n\n" + "вводная часть " * (MAX_FULL_TEXT_CHARS // 10)
    snippet = '## Exclusive gateway[¶](#e "Permanent link")\n\nотрывок про шлюз'
    embedder = VectorEmbedder({})
    service, _, _ = make_service(
        {}, found=[FoundPage(GOOD, text=huge, snippet=snippet)], embedder=embedder
    )

    [result] = await service.search("вопрос", PROFILE)

    assert embedder.calls == [["вопрос", "отрывок про шлюз"]]
    assert result.heading_path == "Exclusive gateway"


async def test_page_exactly_at_the_size_limit_is_used_in_full_one_more_is_not() -> None:
    head = "# Раздел\n\n"
    for extra, expected_in_calls in ((0, False), (1, True)):
        text = head + "x" * (MAX_FULL_TEXT_CHARS - len(head) + extra)
        assert len(text) == MAX_FULL_TEXT_CHARS + extra
        embedder = VectorEmbedder({})
        service, _, _ = make_service(
            {}, found=[FoundPage(GOOD, text=text, snippet="отрывок")], embedder=embedder
        )

        await service.search("вопрос", PROFILE)

        # на самом лимите берём страницу целиком, на символ больше уже отрывки
        assert ("отрывок" in embedder.calls[0]) is expected_in_calls


async def test_huge_document_without_snippets_is_cut_to_the_chunk_limit() -> None:
    sections = "".join(
        f"## Раздел {i}\n\nтекст {i}\n\n" for i in range(MAX_CHUNKS_PER_PAGE + 20)
    )
    text = sections + "я" * MAX_FULL_TEXT_CHARS
    embedder = VectorEmbedder({})
    service, _, _ = make_service(
        {}, found=[FoundPage(GOOD, text=text)], embedder=embedder
    )

    await service.search("вопрос", PROFILE)

    assert len(embedder.calls[0]) == 1 + MAX_CHUNKS_PER_PAGE


# --- кэш: найденные страницы и скачанные страницы

SEARCH_PREFIX = "web:search:v2:"
PAGE_PREFIX = "web:page:v1:"


def cached(service: WebSearchService, prefix: str) -> dict[str, str]:
    assert isinstance(service.cache, InMemoryCache)
    return {k: v for k, v in service.cache.values.items() if k.startswith(prefix)}


async def test_same_question_is_answered_from_cache_without_asking_the_finder_again() -> (
    None
):
    service, finder, _ = make_service(
        {}, found=[FoundPage(GOOD, title="Зависимости", text="# Раздел\n\nтекст")]
    )

    first = await service.search("вопрос", PROFILE)
    second = await service.search("вопрос", PROFILE)

    assert len(finder.queries) == 1
    assert second == first and len(first) == 1


async def test_cache_key_ignores_case_and_spaces_but_not_the_query_itself() -> None:
    service, finder, _ = make_service(
        {}, found=[FoundPage(GOOD, text="# Раздел\n\nтекст")]
    )

    await service.search("FastAPI   Yield", PROFILE)
    await service.search("fastapi yield", PROFILE)
    await service.search("fastapi depends", PROFILE)

    assert finder.queries == ["FastAPI   Yield", "fastapi depends"]


async def test_cache_depends_on_the_set_of_sites_not_on_the_profile_name() -> None:
    python_page = "https://docs.python.org/3/library/asyncio.html"
    service, finder, _ = make_service(
        {},
        found=[
            FoundPage(GOOD, text="# Раздел\n\nтекст про FastAPI"),
            FoundPage(python_page, text="# Раздел\n\nтекст про asyncio"),
        ],
    )
    same_sites = WebProfile(name="other", title="Другой", domains=tuple(DOMAINS))
    other_sites = WebProfile(
        name="backend", title="Тот же", domains=("docs.python.org",)
    )

    first = await service.search("вопрос", PROFILE)
    again = await service.search("вопрос", same_sites)  # те же сайты: готовое
    other = await service.search("вопрос", other_sites)  # то же имя, другие сайты

    assert len(finder.queries) == 2
    assert len(cached(service, SEARCH_PREFIX)) == 2
    assert again == first
    # страницы чужого набора сайтов из общего кэша не просачиваются
    assert [r.url for r in first] == [GOOD]
    assert [r.url for r in other] == [python_page]


async def test_order_of_sites_does_not_change_the_cache_entry() -> None:
    first = WebProfile(
        name="a", title="A", domains=("docs.python.org", "peps.python.org")
    )
    second = WebProfile(
        name="b", title="B", domains=("peps.python.org", "docs.python.org")
    )
    service, finder, _ = make_service(
        {}, found=[FoundPage("https://docs.python.org/3/x", text="# Раздел\n\nтекст")]
    )

    await service.search("вопрос", first)
    await service.search("вопрос", second)

    assert len(finder.queries) == 1


async def test_entry_is_stored_with_the_configured_lifetime() -> None:
    service, _, _ = make_service({}, found=[FoundPage(GOOD, text="# Раздел\n\nтекст")])

    await service.search("вопрос", PROFILE)

    assert set(service.cache.ttls.values()) == {TTL}


async def test_empty_search_result_is_not_remembered() -> None:
    service, finder, _ = make_service({}, found=[EVIL])

    assert await service.search("вопрос", PROFILE) == []
    await service.search("вопрос", PROFILE)

    # чужой сайт отброшен, получилось пусто: повторяем поиск, а не помним пустоту
    assert len(finder.queries) == 2
    assert cached(service, SEARCH_PREFIX) == {}


async def test_failed_search_leaves_nothing_in_cache() -> None:
    service, finder, _ = make_service({}, found=[GOOD])
    finder.find = AsyncMock(side_effect=RuntimeError("поиск недоступен"))  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        await service.search("вопрос", PROFILE)

    assert service.cache.values == {}


async def test_cache_holds_only_allowed_trimmed_pages() -> None:
    huge = "# Спецификация\n\n" + "вводная часть " * MAX_FULL_TEXT_CHARS
    service, _, _ = make_service(
        {},
        found=[FoundPage(EVIL, text="# Чужое\n\nвредное"), FoundPage(GOOD, text=huge)],
    )

    await service.search("вопрос", PROFILE)

    [raw] = cached(service, SEARCH_PREFIX).values()
    stored = json.loads(raw)
    assert [item["url"] for item in stored] == [GOOD]  # чужой сайт не сохранён
    assert (
        len(stored[0]["text"]) == MAX_FULL_TEXT_CHARS
    )  # хвост, который не прочтём, тоже


@pytest.mark.parametrize(
    "garbage", ["не json", '{"a": 1}', "[1, 2]", '[{"нет": "url"}]']
)
async def test_damaged_cache_entry_is_ignored_and_replaced(garbage: str) -> None:
    service, finder, _ = make_service(
        {}, found=[FoundPage(GOOD, text="# Раздел\n\nтекст")]
    )
    await service.search("вопрос", PROFILE)
    [key] = cached(service, SEARCH_PREFIX)
    service.cache.values[key] = garbage

    results = await service.search("вопрос", PROFILE)

    assert len(finder.queries) == 2 and len(results) == 1
    assert json.loads(service.cache.values[key])[0]["url"] == GOOD


async def test_too_big_entry_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.web_search_service.MAX_CACHED_BYTES", 50)
    service, finder, _ = make_service(
        {}, found=[FoundPage(GOOD, text="# Раздел\n\n" + "текст " * 50)]
    )

    await service.search("вопрос", PROFILE)
    await service.search("вопрос", PROFILE)

    assert len(finder.queries) == 2


async def test_zero_lifetime_turns_the_cache_off() -> None:
    service, finder, _ = make_service({}, found=[FoundPage(GOOD, text="# А\n\nтекст")])
    service.cache_ttl_seconds = 0

    await service.search("вопрос", PROFILE)
    await service.search("вопрос", PROFILE)

    assert len(finder.queries) == 2 and service.cache.values == {}


async def test_downloaded_page_is_cached_and_not_downloaded_twice() -> None:
    service, _, fetch = make_service(
        {GOOD: page("Название", s="текст")}, found=[FoundPage(GOOD)]
    )
    # поиск каждый раз новый (ключ другой), страница одна и та же
    first = await service.search("первый вопрос", PROFILE)
    second = await service.search("второй вопрос", PROFILE)

    assert fetch.await_count == 1
    assert [r.content for r in second] == [r.content for r in first] == ["текст"]
    assert len(cached(service, PAGE_PREFIX)) == 1


async def test_failed_download_is_not_cached() -> None:
    service, _, fetch = make_service(
        {GOOD: PageFetchError("HTTP 500")}, found=[FoundPage(GOOD)]
    )

    await service.search("первый", PROFILE)
    await service.search("второй", PROFILE)

    assert fetch.await_count == 2
    assert cached(service, PAGE_PREFIX) == {}


async def test_damaged_page_entry_triggers_a_new_download() -> None:
    service, _, fetch = make_service(
        {GOOD: page("Название", s="текст")}, found=[FoundPage(GOOD)]
    )
    await service.search("первый", PROFILE)
    [key] = cached(service, PAGE_PREFIX)
    service.cache.values[key] = '{"title": "только название"}'

    await service.search("второй", PROFILE)

    assert fetch.await_count == 2


async def test_downloaded_huge_page_is_cut_to_what_is_read_before_it_is_cached() -> (
    None
):
    huge = page("Справочник", s="я" * (MAX_FULL_TEXT_CHARS * 2))
    service, _, _ = make_service({GOOD: huge}, found=[FoundPage(GOOD)])

    await service.search("вопрос", PROFILE)

    [raw] = cached(service, PAGE_PREFIX).values()
    assert len(json.loads(raw)["text"]) == MAX_FULL_TEXT_CHARS


async def test_switching_the_cache_off_stops_reading_already_stored_entries() -> None:
    service, finder, fetch = make_service(
        {GOOD: page("Название", s="текст")}, found=[FoundPage(GOOD)]
    )
    await service.search("вопрос", PROFILE)  # записи в кэше есть
    assert fetch.await_count == 1 and len(finder.queries) == 1

    service.cache_ttl_seconds = 0
    await service.search("вопрос", PROFILE)

    # ни найденные страницы, ни скачанные из кэша не берутся
    assert len(finder.queries) == 2 and fetch.await_count == 2


# --- кэш перефразирования


class ChangingRewriter(FakeQueryRewriter):
    """Каждый раз формулирует по-новому, как живая модель."""

    async def rewrite(self, question: str, context: Sequence[str] = ()) -> str:
        await super().rewrite(question, context)
        return f"запрос номер {len(self.calls)}"


async def test_rewriting_is_remembered_so_the_search_query_stays_the_same() -> None:
    rewriter = ChangingRewriter()
    service, finder, _ = make_service(
        {}, found=[FoundPage(GOOD, text="# Раздел\n\nтекст")], rewriter=rewriter
    )

    await service.search("как объявить зависимость?", PROFILE)
    await service.search("как объявить зависимость?", PROFILE)

    # без запоминания вторая формулировка отличалась бы и поиск шёл бы заново
    assert len(rewriter.calls) == 1
    assert finder.queries == ["запрос номер 1"]


async def test_rewriting_key_ignores_case_and_spaces_of_the_question() -> None:
    rewriter = FakeQueryRewriter(reply="fastapi yield")
    service, _, _ = make_service(
        {}, found=[FoundPage(GOOD, text="# Раздел\n\nтекст")], rewriter=rewriter
    )

    await service.search("Как   объявить YIELD?", PROFILE)
    await service.search("как объявить yield?", PROFILE)

    assert len(rewriter.calls) == 1


async def test_same_follow_up_in_other_conversation_is_rewritten_separately() -> None:
    rewriter = FakeQueryRewriter(reply="fastapi yield")
    service, _, _ = make_service(
        {}, found=[FoundPage(GOOD, text="# Раздел\n\nтекст")], rewriter=rewriter
    )
    follow_up = "а как с yield?"

    await service.search(follow_up, PROFILE, context=["про Depends"])
    await service.search(follow_up, PROFILE, context=["про Depends"])
    await service.search(follow_up, PROFILE, context=["про Kafka"])
    await service.search(follow_up, PROFILE)

    # одинаковый контекст — одно обращение, разный или пустой — отдельные
    assert [c[1] for c in rewriter.calls] == [["про Depends"], ["про Kafka"], []]


async def test_question_returned_unchanged_is_not_remembered() -> None:
    # так выглядит запасной вариант при сбое модели: сутки им пользоваться нельзя
    rewriter = FakeQueryRewriter()
    service, _, _ = make_service(
        {}, found=[FoundPage(GOOD, text="# Раздел\n\nтекст")], rewriter=rewriter
    )

    await service.search("вопрос", PROFILE)
    await service.search("вопрос", PROFILE)

    assert len(rewriter.calls) == 2
    assert cached(service, "web:rewrite:") == {}


async def test_rewriting_is_stored_for_the_configured_time_without_the_question_text() -> (
    None
):
    rewriter = FakeQueryRewriter(reply="fastapi yield")
    service, _, _ = make_service(
        {}, found=[FoundPage(GOOD, text="# Раздел\n\nтекст")], rewriter=rewriter
    )

    await service.search("секретный вопрос", PROFILE, context=["секретный контекст"])

    [(key, value)] = cached(service, "web:rewrite:v2:").items()
    assert value == "fastapi yield"
    assert service.cache.ttls[key] == TTL
    assert "секретный" not in key


async def test_zero_lifetime_rewrites_every_time() -> None:
    rewriter = FakeQueryRewriter(reply="fastapi yield")
    service, _, _ = make_service(
        {}, found=[FoundPage(GOOD, text="# Раздел\n\nтекст")], rewriter=rewriter
    )
    service.cache_ttl_seconds = 0

    await service.search("вопрос", PROFILE)
    await service.search("вопрос", PROFILE)

    assert len(rewriter.calls) == 2


# --- личные списки: свои скачивания нет, кэш общий по набору сайтов

MINE = WebProfile(
    name="my-0123456789ab",
    title="Мой список",
    domains=tuple(DOMAINS),
    fetch_pages=False,
)


async def test_personal_list_never_downloads_pages_itself() -> None:
    # поисковик прислал только адрес: у обычного профиля мы бы скачали страницу
    service, _, fetch = make_service(
        {GOOD: page("Название", s="текст")}, found=[FoundPage(GOOD)]
    )

    results = await service.search("вопрос", MINE)

    assert results == [] and fetch.await_count == 0
    assert cached(service, PAGE_PREFIX) == {}


async def test_personal_list_uses_the_excerpt_when_there_is_no_full_text() -> None:
    service, _, fetch = make_service(
        {}, found=[FoundPage(GOOD, title="Статья", snippet="Короткий отрывок статьи")]
    )

    [result] = await service.search("вопрос", MINE)

    assert result.content == "Короткий отрывок статьи"
    assert result.document_title == "Статья" and result.url == GOOD
    assert fetch.await_count == 0


async def test_personal_list_still_uses_full_text_from_the_search() -> None:
    service, _, fetch = make_service(
        {}, found=[FoundPage(GOOD, text="# Раздел\n\nполный текст", snippet="отрывок")]
    )

    [result] = await service.search("вопрос", MINE)

    assert "полный текст" in result.content
    assert fetch.await_count == 0


async def test_personal_list_still_filters_pages_by_its_own_sites() -> None:
    service, _, _ = make_service(
        {},
        found=[
            FoundPage(EVIL, text="# Чужое\n\nвредное"),
            FoundPage(GOOD, text="# А\n\nтекст"),
        ],
    )

    results = await service.search("вопрос", MINE)

    assert [r.url for r in results] == [GOOD]


async def test_excerpt_is_cleaned_before_it_is_cached() -> None:
    service, _, _ = make_service(
        {}, found=[FoundPage(GOOD, snippet="Заголовок [¶](#anchor)\n\nтекст")]
    )

    await service.search("вопрос", MINE)

    [raw] = cached(service, SEARCH_PREFIX).values()
    assert "¶" not in json.loads(raw)[0]["snippet"]


async def test_page_without_text_and_excerpt_is_skipped_for_personal_list() -> None:
    service, _, fetch = make_service({}, found=[FoundPage(GOOD)])

    assert await service.search("вопрос", MINE) == []
    assert fetch.await_count == 0


async def test_excerpt_is_cut_to_what_is_read_before_it_is_cached() -> None:
    service, _, _ = make_service(
        {}, found=[FoundPage(GOOD, snippet="я" * (MAX_FULL_TEXT_CHARS * 2))]
    )

    await service.search("вопрос", MINE)

    [raw] = cached(service, SEARCH_PREFIX).values()
    assert len(json.loads(raw)[0]["snippet"]) == MAX_FULL_TEXT_CHARS
