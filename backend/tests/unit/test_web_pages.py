import httpx
import pytest

from app.ingestion.chunker import split_into_chunks
from app.ingestion.web_pages import (
    MAX_REDIRECTS,
    FetchedPage,
    PageFetcher,
    PageFetchError,
    clean_markdown,
    html_to_markdown,
    is_allowed_url,
    normalize_url,
)

ALLOWED = ["fastapi.tiangolo.com", "docs.python.org"]
HTML_HEADERS = {"content-type": "text/html; charset=utf-8"}


# --- is_allowed_url: главная граница безопасности


@pytest.mark.parametrize(
    "url",
    [
        "https://fastapi.tiangolo.com/tutorial/dependencies/",
        "https://FastAPI.Tiangolo.com./ru/",  # регистр и точка в конце имени хоста
        "https://docs.python.org/3/library/asyncio.html",
        "https://ru.docs.python.org/3/",  # поддомен разрешённого домена
    ],
)
def test_documentation_urls_are_allowed(url: str) -> None:
    assert is_allowed_url(url, ALLOWED)


@pytest.mark.parametrize(
    "url",
    [
        "http://fastapi.tiangolo.com/",  # не https
        "https://evilfastapi.tiangolo.com/",  # домен лишь оканчивается так же
        "https://fastapi.tiangolo.com.evil.com/",  # разрешённое имя только в начале
        "https://evil.com/fastapi.tiangolo.com",  # и только в пути
        "https://fastapi.tiangolo.com@evil.com/",  # логин в адресе
        "https://evil.com\\@fastapi.tiangolo.com/",  # обратный слеш: расхождение разборщиков
        "https://user:pw@fastapi.tiangolo.com/",
        "https://fastapi.tiangolo.com:8443/",  # нестандартный порт
        "https://127.0.0.1/",
        "https://[::1]/",
        "ftp://fastapi.tiangolo.com/",
        "//fastapi.tiangolo.com/",
        "не адрес",
        "",
    ],
)
def test_lookalike_and_unsafe_urls_are_rejected(url: str) -> None:
    assert not is_allowed_url(url, ALLOWED)


def test_empty_allowlist_allows_nothing() -> None:
    assert not is_allowed_url("https://fastapi.tiangolo.com/", [])


def test_normalize_url_drops_tracking_query_and_fragment() -> None:
    url = "https://fastapi.tiangolo.com/ru/tutorial/?utm_source=openai#section"

    assert normalize_url(url) == "https://fastapi.tiangolo.com/ru/tutorial/"
    # поиск иногда пишет «//» в пути: это та же страница
    assert (
        normalize_url("https://fastapi.tiangolo.com//reference/dependencies/")
        == "https://fastapi.tiangolo.com/reference/dependencies/"
    )
    assert normalize_url("не адрес://\x00") is None


# --- PageFetcher


def make_fetcher(handler, **kwargs: int) -> PageFetcher:  # type: ignore[no-untyped-def]
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return PageFetcher(client, **kwargs)


async def test_fetch_returns_page_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers=HTML_HEADERS, content="<p>Привет</p>".encode()
        )

    page = await make_fetcher(handler).fetch("https://fastapi.tiangolo.com/", ALLOWED)

    assert page == FetchedPage(title="", text="Привет")


async def test_fetch_follows_redirect_inside_allowed_domains() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old/":
            return httpx.Response(301, headers={"location": "/new/"})
        return httpx.Response(200, headers=HTML_HEADERS, content="<p>новая</p>")

    page = await make_fetcher(handler).fetch(
        "https://fastapi.tiangolo.com/old/", ALLOWED
    )

    assert page.text == "новая"


async def test_redirect_to_foreign_host_is_refused_without_connecting() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.host)
        return httpx.Response(302, headers={"location": "https://evil.com/steal"})

    with pytest.raises(PageFetchError, match="не из списка"):
        await make_fetcher(handler).fetch("https://fastapi.tiangolo.com/", ALLOWED)

    # к чужому хосту запроса не было: адрес проверен до соединения
    assert requested == ["fastapi.tiangolo.com"]


async def test_redirect_loop_stops() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "/again/"})

    with pytest.raises(PageFetchError, match="слишком много"):
        await make_fetcher(handler).fetch("https://fastapi.tiangolo.com/", ALLOWED)

    assert len(calls) == MAX_REDIRECTS + 1


async def test_disallowed_url_makes_no_request_at_all() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("сеть не должна вызываться")

    with pytest.raises(PageFetchError, match="не из списка"):
        await make_fetcher(handler).fetch("http://fastapi.tiangolo.com/", ALLOWED)


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (httpx.Response(404, headers=HTML_HEADERS, content="нет"), "404"),
        (
            httpx.Response(
                200, headers={"content-type": "image/png"}, content=b"\x89PNG"
            ),
            "не HTML и не PDF",
        ),
        (
            httpx.Response(
                200,
                headers={"content-type": "application/pdf"},
                content=b"%PDF-1.4 garbage",
            ),
            "не удалось прочитать PDF",
        ),
        (httpx.Response(200, content="<p>без типа</p>"), "не HTML"),
        (
            httpx.Response(200, headers=HTML_HEADERS, content=b"x" * 500),
            "слишком большая",
        ),
    ],
)
async def test_unusable_responses_are_refused(
    response: httpx.Response, reason: str
) -> None:
    fetcher = make_fetcher(lambda request: response, max_bytes=100)

    with pytest.raises(PageFetchError, match=reason):
        await fetcher.fetch("https://fastapi.tiangolo.com/", ALLOWED)


async def test_allowed_sites_are_chosen_per_call_not_per_fetcher() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=HTML_HEADERS, content="<p>ок</p>")

    fetcher = make_fetcher(handler)

    assert await fetcher.fetch("https://docs.camunda.io/", ["docs.camunda.io"])
    # тот же скачиватель, но сайт не из списка этого вызова
    with pytest.raises(PageFetchError, match="не из списка"):
        await fetcher.fetch("https://docs.camunda.io/", ALLOWED)


async def test_network_failure_becomes_fetch_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("нет связи")

    with pytest.raises(PageFetchError, match="сеть"):
        await make_fetcher(handler).fetch("https://fastapi.tiangolo.com/", ALLOWED)


# --- html_to_markdown

SAMPLE_PAGE = """
<html><head><title>Зависимости - FastAPI</title><style>.x { color: red }</style></head>
<body>
<header>Шапка сайта</header>
<nav>Меню навигации</nav>
<main>
  <div>Хлебные крошки</div>
  <article>
    <h1>Зависимости<a class="headerlink" href="#x" title="Permanent link">¶</a></h1>
    <p>Текст про <code>Depends</code> &amp; другое.</p>
    <h2>Первые шаги</h2>
    <ul><li>первый пункт</li><li>второй пункт</li></ul>
    <div class="highlight"><pre><span></span><code># это комментарий, а не заголовок
def dependency(): ...
</code></pre></div>
    <script>alert(1)</script>
  </article>
</main>
<footer>Подвал сайта</footer>
</body></html>
"""


def test_page_becomes_markdown_with_headings_lists_and_code() -> None:
    title, text = html_to_markdown(SAMPLE_PAGE)

    assert title == "Зависимости - FastAPI"
    assert "# Зависимости\n" in text and "¶" not in text
    assert "Текст про `Depends` & другое." in text
    assert "## Первые шаги" in text
    assert "- первый пункт\n\n- второй пункт" in text
    assert "```\n# это комментарий, а не заголовок\ndef dependency(): ...\n```" in text


def test_navigation_scripts_and_everything_outside_article_are_dropped() -> None:
    _, text = html_to_markdown(SAMPLE_PAGE)

    for junk in ("Шапка", "Меню", "Хлебные", "Подвал", "alert", "color: red"):
        assert junk not in text


def test_markdown_from_page_splits_into_chunks_by_headings() -> None:
    _, text = html_to_markdown(SAMPLE_PAGE)

    chunks = split_into_chunks(text)

    assert [c.heading_path for c in chunks] == [
        "Зависимости",
        "Зависимости > Первые шаги",
    ]
    # комментарий внутри кода не стал заголовком
    assert not any("это комментарий" in c.heading_path for c in chunks)


def test_page_without_article_or_main_falls_back_to_body() -> None:
    _, text = html_to_markdown("<body><nav>меню</nav><p>Просто   текст</p></body>")

    assert text == "Просто текст"


def test_code_fence_is_longer_than_backticks_inside_the_code() -> None:
    _, text = html_to_markdown("<article><pre>print('```')</pre></article>")

    assert text == "````\nprint('```')\n````"


def test_empty_or_broken_html_gives_empty_text() -> None:
    assert html_to_markdown("") == ("", "")
    assert html_to_markdown("<div><p>не закрыто") == ("", "не закрыто")


# --- clean_markdown: чистка готового markdown от поискового сервиса

SEARCH_MARKDOWN = """[Skip to content](#dependencies-with-yield)

* [Первый раздел](#pervyj)
  + [Вложенный](#vlozhennyj)

# Dependencies with yield[¶](#dependencies-with-yield "Permanent link")

Текст со [ссылкой](https://example.com/) и [якорем](#x) внутри абзаца.

## Первый раздел[¶](#pervyj "Permanent link")

```python
* [не оглавление](#x)
# comment[¶](#keep)
```
"""


def test_toc_skip_links_and_heading_anchors_are_removed() -> None:
    text = clean_markdown(SEARCH_MARKDOWN)

    assert "Skip to content" not in text and "Вложенный" not in text
    assert "# Dependencies with yield\n" in text and "¶" not in text.split("```")[0]
    assert [c.heading_path for c in split_into_chunks(text)] == [
        "Dependencies with yield",
        "Dependencies with yield > Первый раздел",
    ]


def test_links_inside_text_and_code_blocks_are_left_alone() -> None:
    text = clean_markdown(SEARCH_MARKDOWN)

    assert "[ссылкой](https://example.com/) и [якорем](#x) внутри" in text
    assert "* [не оглавление](#x)\n# comment[¶](#keep)" in text


def test_clean_markdown_collapses_blank_lines_left_by_removed_ones() -> None:
    assert clean_markdown("a\n\n* [x](#y)\n\n\n\nb") == "a\n\nb"


# --- PDF: запасной путь, когда поисковик не принёс текст страницы

PDF_HEADERS = {"content-type": "application/pdf"}


def pdf_response(data: bytes, **headers: str) -> httpx.Response:
    return httpx.Response(200, headers={**PDF_HEADERS, **headers}, content=data)


async def test_pdf_becomes_markdown_with_a_section_per_page_and_its_title(
    pdf_bytes,  # type: ignore[no-untyped-def]
) -> None:
    data = pdf_bytes("First page text", "Second page text", title="BPMN Specification")
    fetcher = make_fetcher(lambda request: pdf_response(data))

    page = await fetcher.fetch("https://fastapi.tiangolo.com/spec.pdf", ALLOWED)

    assert page.title == "BPMN Specification"
    assert page.text == (
        "# Страница 1\n\nFirst page text\n\n# Страница 2\n\nSecond page text"
    )
    # номер страницы доходит до цитаты через путь раздела
    assert [c.heading_path for c in split_into_chunks(page.text)] == [
        "Страница 1",
        "Страница 2",
    ]


async def test_pages_without_text_are_skipped_but_keep_their_numbers(
    pdf_bytes,  # type: ignore[no-untyped-def]
) -> None:
    fetcher = make_fetcher(lambda request: pdf_response(pdf_bytes("one", "", "three")))

    page = await fetcher.fetch("https://fastapi.tiangolo.com/a.pdf", ALLOWED)

    assert page.text == "# Страница 1\n\none\n\n# Страница 3\n\nthree"


async def test_pdf_without_any_text_is_refused_as_a_possible_scan(
    pdf_bytes,  # type: ignore[no-untyped-def]
) -> None:
    fetcher = make_fetcher(lambda request: pdf_response(pdf_bytes("", "")))

    with pytest.raises(PageFetchError, match="нет текста"):
        await fetcher.fetch("https://fastapi.tiangolo.com/scan.pdf", ALLOWED)


async def test_only_the_first_pages_of_a_long_pdf_are_read(
    pdf_bytes,  # type: ignore[no-untyped-def]
) -> None:
    data = pdf_bytes("a", "b", "c", "d")
    fetcher = make_fetcher(lambda request: pdf_response(data), max_pdf_pages=2)

    page = await fetcher.fetch("https://fastapi.tiangolo.com/long.pdf", ALLOWED)

    assert page.text == "# Страница 1\n\na\n\n# Страница 2\n\nb"


async def test_pdf_has_its_own_size_limit_larger_than_the_one_for_html(
    pdf_bytes,  # type: ignore[no-untyped-def]
) -> None:
    data = pdf_bytes("text")
    assert len(data) > 100  # лимит HTML ниже: PDF под него не подпадает

    allowed = make_fetcher(lambda request: pdf_response(data), max_bytes=100)
    assert (await allowed.fetch("https://fastapi.tiangolo.com/a.pdf", ALLOWED)).text

    too_big = make_fetcher(lambda request: pdf_response(data), max_pdf_bytes=100)
    with pytest.raises(PageFetchError, match="слишком большая"):
        await too_big.fetch("https://fastapi.tiangolo.com/a.pdf", ALLOWED)


async def test_pdf_redirect_to_foreign_host_is_still_refused(
    pdf_bytes,  # type: ignore[no-untyped-def]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://evil.com/a.pdf"})

    with pytest.raises(PageFetchError, match="не из списка"):
        await make_fetcher(handler).fetch("https://fastapi.tiangolo.com/a.pdf", ALLOWED)


@pytest.mark.parametrize("content_type", ["TEXT/HTML; charset=UTF-8", "Text/Html"])
async def test_content_type_is_compared_without_regard_to_case(
    content_type: str,
) -> None:
    fetcher = make_fetcher(
        lambda request: httpx.Response(
            200, headers={"content-type": content_type}, content="<p>текст</p>"
        )
    )

    assert (
        await fetcher.fetch("https://fastapi.tiangolo.com/", ALLOWED)
    ).text == "текст"


async def test_pdf_content_type_is_compared_without_regard_to_case(
    pdf_bytes,  # type: ignore[no-untyped-def]
) -> None:
    data = pdf_bytes("Hello")
    fetcher = make_fetcher(
        lambda request: pdf_response(data, **{"content-type": "Application/PDF"})
    )

    page = await fetcher.fetch("https://fastapi.tiangolo.com/a.pdf", ALLOWED)

    assert page.text == "# Страница 1\n\nHello"
