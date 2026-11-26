import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from io import BytesIO

import httpx

from app.ingestion.parser import read_pdf

# только ASCII: HTTP-заголовки кириллицу не принимают, httpx падает при создании клиента
USER_AGENT = "DocuMindBot/0.1 (documentation search)"
# Страница документации весит сотни килобайт; больше двух мегабайт — это уже не она
MAX_PAGE_BYTES = 2_000_000
# Спецификации и стандарты в PDF весят единицы мегабайт и сотни страниц
MAX_PDF_BYTES = 15_000_000
MAX_PDF_PAGES = 200
MAX_REDIRECTS = 3


class PageFetchError(Exception):
    """Страницу нельзя скачать: адрес не разрешён, ответ не тот или сеть подвела."""


@dataclass(frozen=True)
class FetchedPage:
    """Скачанная страница: название (может быть пустым) и текст в виде markdown."""

    title: str
    text: str


def is_allowed_url(url: str, allowed_domains: Sequence[str]) -> bool:
    """Можно ли ходить по этому адресу: https и домен из списка (или его поддомен).

    Адрес разбирает сам httpx, тот же, что потом откроет соединение: если бы мы
    проверяли одним разборщиком, а соединялись другим, их расхождение дало бы
    лазейку (на этом строятся трюки вроде https://good.com\\@evil.com).
    """
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL:
        return False

    # логин:пароль в адресе и нестандартный порт нам не нужны, а подвоха допускают
    if parsed.scheme != "https" or parsed.userinfo or parsed.port is not None:
        return False

    host = parsed.host.lower().rstrip(".")
    for domain in allowed_domains:
        domain = domain.lower().strip(".")
        # точка перед доменом обязательна: «evilfastapi.tiangolo.com» — чужой сайт
        if domain and (host == domain or host.endswith("." + domain)):
            return True
    return False


def normalize_url(url: str) -> str | None:
    """Адрес без параметров, якоря и двойных слешей: для документации они ничего
    не меняют, а поиск добавляет к ссылкам метки (?utm_source=...) и иногда
    пишет «//» в пути, из-за чего одна страница выглядела бы разными."""
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL:
        return None
    path = re.sub(r"/{2,}", "/", parsed.path)
    return str(parsed.copy_with(path=path, query=None, fragment=None))


class PageFetcher:
    """Скачивает страницы только с разрешённых доменов: HTML и PDF.

    Клиент приходит снаружи: приложение создаёт один на всё время работы (у него
    пул соединений), а тесты подставляют клиент с поддельным транспортом.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        max_bytes: int = MAX_PAGE_BYTES,
        max_pdf_bytes: int = MAX_PDF_BYTES,
        max_pdf_pages: int = MAX_PDF_PAGES,
    ) -> None:
        self._client = client
        self._max_bytes = max_bytes
        self._max_pdf_bytes = max_pdf_bytes
        self._max_pdf_pages = max_pdf_pages

    async def fetch(self, url: str, allowed_domains: Sequence[str]) -> FetchedPage:
        """Страница в виде markdown или PageFetchError с причиной.

        Список сайтов передаётся при каждом вызове: у разных профилей он разный,
        а скачиватель один на всё приложение.
        """
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            # проверяем каждый адрес в цепочке, а не только первый: разрешённый
            # сайт мог бы перенаправить нас на любой другой
            if not is_allowed_url(current, allowed_domains):
                raise PageFetchError(f"адрес не из списка разрешённых: {current}")
            try:
                # follow_redirects=False: переходы ведём сами, с проверкой выше
                async with self._client.stream(
                    "GET", current, follow_redirects=False
                ) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise PageFetchError("перенаправление без адреса")
                        current = str(response.url.join(location))
                        continue
                    return await self._read(response)
            except httpx.HTTPError as exc:
                raise PageFetchError(f"сеть: {exc!r}") from exc
        raise PageFetchError("слишком много перенаправлений")

    async def _read(self, response: httpx.Response) -> FetchedPage:
        if response.status_code != 200:
            raise PageFetchError(f"ответ сайта: {response.status_code}")
        content_type = response.headers.get("content-type", "").lower()
        is_pdf = content_type.startswith("application/pdf")
        if not is_pdf and not content_type.startswith("text/html"):
            raise PageFetchError(
                f"это не HTML и не PDF: {content_type or 'тип не указан'}"
            )

        limit = self._max_pdf_bytes if is_pdf else self._max_bytes
        body = bytearray()
        # читаем кусками и останавливаемся на лимите: заголовку Content-Length
        # верить нельзя, а целиком читать неизвестно что — значит дать себя
        # завалить огромным ответом
        async for part in response.aiter_bytes():
            body += part
            if len(body) > limit:
                raise PageFetchError("страница слишком большая")

        if is_pdf:
            # разбор PDF занимает процессор и может быть долгим: не в цикле событий
            return await asyncio.to_thread(self._pdf_to_page, bytes(body))
        try:
            html = bytes(body).decode(response.encoding or "utf-8", errors="replace")
        except LookupError:  # сайт указал кодировку, которой Python не знает
            html = bytes(body).decode("utf-8", errors="replace")
        title, text = html_to_markdown(html)
        return FetchedPage(title, text)

    def _pdf_to_page(self, data: bytes) -> FetchedPage:
        try:
            title, pages = read_pdf(BytesIO(data), self._max_pdf_pages)
        except (
            Exception
        ) as exc:  # битый или хитро собранный файл ломает разбор по-разному
            raise PageFetchError("не удалось прочитать PDF") from exc
        # номер страницы в заголовке первого уровня попадёт в путь раздела: по цитате
        # видно, где искать (заголовки второго уровня чанкер вложил бы друг в друга)
        sections = [
            f"# Страница {number}\n\n{text}"
            for number, text in enumerate(pages, start=1)
            if text
        ]
        if not sections:
            raise PageFetchError("в PDF нет текста (возможно, это скан)")
        return FetchedPage(title, "\n\n".join(sections))


# --- из HTML в markdown, который понимает наш чанкер

# Вся эта разметка — навигация, оформление и скрипты, а не текст страницы
_SKIPPED_TAGS = frozenset(
    {
        "script", "style", "nav", "header", "footer", "aside", "svg", "button",
        "form", "noscript", "template", "iframe", "select",
    }
)  # fmt: skip
_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_BLOCK_TAGS = frozenset(
    {
        "p", "div", "ul", "ol", "tr", "table", "section", "blockquote", "details",
        "summary", "dl", "dt", "dd", "figure", "figcaption", "caption", "thead",
        "tbody", "br", "hr",
    }
)  # fmt: skip


def _collapse(text: str) -> str:
    """Любые пробелы, переносы и табы одним пробелом."""
    return re.sub(r"\s+", " ", text).strip()


class _PageParser(HTMLParser):
    """Собирает текст страницы в markdown: заголовки, абзацы, списки и код.

    Основной текст берётся из <article>, а если его нет, из <main>, иначе из всей
    страницы: так отбрасывается шапка, меню и подвал сайта.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._title_parts: list[str] = []
        self._in_title = False
        # пропускаемый элемент: его имя и глубина вложенности одноимённых тегов
        self._skip_tag: str | None = None
        self._skip_depth = 0
        self._depth = {"article": 0, "main": 0}
        self._buckets: dict[str, list[str]] = {"article": [], "main": [], "body": []}
        # текст текущего абзаца: копится по кусочкам и выписывается в _flush
        self._inline: list[str] = []
        self._prefix = ""
        self._heading: int | None = None
        self._pre: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._skip_tag is not None:
            if tag == self._skip_tag:
                self._skip_depth += 1
            return
        # якорь «¶» рядом с заголовком в MkDocs: к тексту отношения не имеет
        classes = dict(attrs).get("class") or ""
        if tag in _SKIPPED_TAGS or (tag == "a" and "headerlink" in classes):
            self._skip_tag, self._skip_depth = tag, 1
            return

        if tag == "title":
            self._in_title = True
        elif tag in ("article", "main"):
            self._flush()
            self._depth[tag] += 1
        elif tag in _HEADING_TAGS:
            self._flush()
            self._heading = int(tag[1])
        elif tag == "pre":
            self._flush()
            self._pre = []
        elif tag == "code" and self._pre is None:
            self._inline.append("`")
        elif tag == "li":
            self._flush()
            self._prefix = "- "
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if self._skip_tag is not None:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if self._skip_depth == 0:
                    self._skip_tag = None
            return

        if tag == "title":
            self._in_title = False
        elif tag in ("article", "main"):
            self._flush()
            self._depth[tag] = max(0, self._depth[tag] - 1)
        elif tag in _HEADING_TAGS:
            self._end_heading()
        elif tag == "pre":
            self._end_pre()
        elif tag == "code" and self._pre is None:
            self._inline.append("`")
        elif tag in ("td", "th"):
            self._inline.append(" | ")
        elif tag == "li":
            self._flush()
            self._prefix = ""
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_tag is not None:
            return
        if self._in_title:
            self._title_parts.append(data)
        elif self._pre is not None:
            # в блоке кода пробелы и переносы строк значимы: не сворачиваем
            self._pre.append(data)
        else:
            self._inline.append(data)

    def _bucket(self) -> list[str]:
        if self._depth["article"]:
            return self._buckets["article"]
        if self._depth["main"]:
            return self._buckets["main"]
        return self._buckets["body"]

    def _emit(self, block: str) -> None:
        # пустая строка после блока: абзацы в markdown разделяются ею, и по ней
        # чанкер режет слишком длинную секцию
        self._bucket().extend([block, ""])

    def _flush(self) -> None:
        text = _collapse("".join(self._inline))
        self._inline.clear()
        if text:
            self._emit(self._prefix + text)
            self._prefix = ""

    def _end_heading(self) -> None:
        text = _collapse("".join(self._inline))
        self._inline.clear()
        if text and self._heading is not None:
            self._emit("#" * self._heading + " " + text)
        self._heading = None

    def _end_pre(self) -> None:
        code = "".join(self._pre or []).strip("\n")
        self._pre = None
        if not code.strip():
            return
        # «забор» длиннее любой цепочки обратных кавычек внутри кода, чтобы код
        # не мог закрыть его раньше времени; внутри забора «# …» — это комментарий,
        # а не заголовок (чанкер так и читает)
        fence = "```"
        while fence in code:
            fence += "`"
        self._emit(f"{fence}\n{code}\n{fence}")

    def result(self) -> tuple[str, str]:
        self._flush()
        for name in ("article", "main", "body"):
            lines = self._buckets[name]
            if any(line.strip() for line in lines):
                text = "\n".join(lines)
                break
        else:
            text = ""
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return _collapse("".join(self._title_parts)), text


def html_to_markdown(html: str) -> tuple[str, str]:
    """Название и основной текст страницы в виде markdown."""
    parser = _PageParser()
    parser.feed(html)
    parser.close()
    return parser.result()


# --- чистка markdown, который уже пришёл готовым (от поискового сервиса)

# якорь-ссылка у заголовков: «Заголовок[¶](#zagolovok "Permanent link")»
_PERMALINK = re.compile(r"\[[¶#]\]\([^)]*\)")
# строка из одной ссылки на место этой же страницы: пункт оглавления, «Skip to content»
_ANCHOR_ONLY_LINE = re.compile(r"^\s*(?:[*+-]\s+)?\[[^\]]*\]\(#[^)]*\)\s*$")


def clean_markdown(text: str) -> str:
    """Убирает из готового markdown страницы то, что только мешает поиску по смыслу.

    Оглавление из ссылок на разделы и якоря у заголовков попали бы в куски текста и
    в названия разделов (heading_path). Код внутри блоков не трогаем.
    """
    lines: list[str] = []
    in_code = False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_code = not in_code
        elif not in_code:
            if _ANCHOR_ONLY_LINE.match(line):
                continue
            line = _PERMALINK.sub("", line)
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
