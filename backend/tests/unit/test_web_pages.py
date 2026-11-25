import httpx
import pytest

from app.ingestion.chunker import split_into_chunks
from app.ingestion.web_pages import (
    MAX_REDIRECTS,
    PageFetcher,
    PageFetchError,
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
    return PageFetcher(client, ALLOWED, **kwargs)


async def test_fetch_returns_page_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers=HTML_HEADERS, content="<p>Привет</p>".encode()
        )

    html = await make_fetcher(handler).fetch("https://fastapi.tiangolo.com/")

    assert html == "<p>Привет</p>"


async def test_fetch_follows_redirect_inside_allowed_domains() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old/":
            return httpx.Response(301, headers={"location": "/new/"})
        return httpx.Response(200, headers=HTML_HEADERS, content="<p>новая</p>")

    html = await make_fetcher(handler).fetch("https://fastapi.tiangolo.com/old/")

    assert html == "<p>новая</p>"


async def test_redirect_to_foreign_host_is_refused_without_connecting() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.host)
        return httpx.Response(302, headers={"location": "https://evil.com/steal"})

    with pytest.raises(PageFetchError, match="не из списка"):
        await make_fetcher(handler).fetch("https://fastapi.tiangolo.com/")

    # к чужому хосту запроса не было: адрес проверен до соединения
    assert requested == ["fastapi.tiangolo.com"]


async def test_redirect_loop_stops() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "/again/"})

    with pytest.raises(PageFetchError, match="слишком много"):
        await make_fetcher(handler).fetch("https://fastapi.tiangolo.com/")

    assert len(calls) == MAX_REDIRECTS + 1


async def test_disallowed_url_makes_no_request_at_all() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("сеть не должна вызываться")

    with pytest.raises(PageFetchError, match="не из списка"):
        await make_fetcher(handler).fetch("http://fastapi.tiangolo.com/")


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (httpx.Response(404, headers=HTML_HEADERS, content="нет"), "404"),
        (
            httpx.Response(
                200, headers={"content-type": "application/pdf"}, content=b"%PDF"
            ),
            "не HTML",
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
        await fetcher.fetch("https://fastapi.tiangolo.com/")


async def test_network_failure_becomes_fetch_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("нет связи")

    with pytest.raises(PageFetchError, match="сеть"):
        await make_fetcher(handler).fetch("https://fastapi.tiangolo.com/")


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
