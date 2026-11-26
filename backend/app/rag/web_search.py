import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx
from openai import AsyncOpenAI
from openai.types.responses import ResponseIncludable, WebSearchToolParam

from app.core.config import settings
from app.core.openai_client import build_openai_client
from app.ingestion.web_pages import normalize_url

logger = logging.getLogger(__name__)

# Ответ модели нам не нужен, нужен только список найденных страниц: он приходит
# отдельно от текста. Лимит защищает от лишних (и платных) токенов
FINDER_MAX_OUTPUT_TOKENS = 150

# Параметры запроса заданы типизированными константами, а не словарями на месте:
# так mypy проверяет их по описанию API
_WEB_SEARCH_TOOL: WebSearchToolParam = {"type": "web_search"}
# полный список просмотренных страниц, а не только процитированных в тексте
_INCLUDE_SOURCES: list[ResponseIncludable] = ["web_search_call.action.sources"]

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
# Поиск с текстами страниц занимает до 9 секунд (замер на живых запросах)
TAVILY_TIMEOUT_SECONDS = 30.0
# Один результат Tavily на каждый якорь (#раздел) страницы: пять результатов бывают
# одной страницей. Просим с запасом, а повторы склеиваем; кредиты от этого не растут
TAVILY_MAX_RESULTS = 10


class SearchBackendError(Exception):
    """Поисковый сервис не ответил как нужно: сеть, ключ, лимит или сбой на его стороне."""


@dataclass(frozen=True)
class FoundPage:
    """Найденная страница. Тексты есть, только если поисковик сразу их отдаёт."""

    url: str
    title: str | None = None
    # полный текст страницы в markdown: тогда скачивать её самим не нужно
    text: str | None = None
    # самые подходящие вопросу отрывки: выручают, когда страница слишком велика
    snippet: str | None = None


class PageFinder(ABC):
    """Находит страницы документации по вопросу.

    Ничему из найденного не доверяем: список сайтов проверяет сервис, который
    этим пользуется, а не сам поисковик.
    """

    @abstractmethod
    async def find(self, query: str, domains: Sequence[str]) -> list[FoundPage]:
        """query — поисковый запрос (его готовит QueryRewriter), domains — сайты профиля."""
        raise NotImplementedError


def _unique_urls(urls: Iterable[str]) -> list[str]:
    """Адреса без параметров и повторов, в порядке появления."""
    result: dict[str, None] = {}
    for url in urls:
        normalized = normalize_url(url)
        if normalized is not None:
            result[normalized] = None
    return list(result)


class OpenAIPageFinder(PageFinder):
    """Встроенный веб-поиск OpenAI (Responses API) как «разведчик».

    Запасной вариант на случай, когда ключа Tavily нет. Мы не берём у модели
    готовый ответ: он не вписывается в наши источники со сносками. От неё нужны
    только адреса страниц, а тексты скачиваем и режем сами. Ограничение «только
    эти сайты» у выбранной модели (gpt-4.1-mini) в API недоступно, поэтому просим
    словами в инструкции, а проверяем по-настоящему потом, в коде. На длинных
    списках сайтов он ненадёжен: нередко не находит ничего.
    """

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        self._client = client or build_openai_client()

    async def find(self, query: str, domains: Sequence[str]) -> list[FoundPage]:
        sites = ", ".join(domains)
        response = await self._client.responses.create(
            model=settings.web_search_model,
            instructions=(
                "Найди в интернете страницы официальной документации или "
                "стандартов, по которым можно ответить на вопрос. Ищи только на "
                f"этих сайтах: {sites}. Выбери один-два самых подходящих по теме "
                "вопроса и ищи на них оператором site:, а не на всех сразу. "
                "Называй в запросе продукт, технологию или стандарт, лучше по-"
                "английски: документация в основном на нём. Предпочитай "
                "актуальную версию. Отвечать на вопрос не нужно: достаточно "
                "найти страницы."
            ),
            input=query,
            tools=[_WEB_SEARCH_TOOL],
            # «required»: инструмент один, значит поиск обязателен; без этого
            # модель могла бы ответить по памяти и не искать вовсе
            tool_choice="required",
            include=_INCLUDE_SOURCES,
            max_output_tokens=FINDER_MAX_OUTPUT_TOKENS,
        )

        cited: list[str] = []
        consulted: list[str] = []
        for item in response.output:
            if item.type == "web_search_call":
                # sources есть не у каждого вида действия (открытие страницы, поиск
                # по тексту их не имеют), поэтому getattr
                sources = getattr(item.action, "sources", None) or []
                consulted += [source.url for source in sources]
            elif item.type == "message":
                for part in item.content:
                    if part.type == "output_text":
                        cited += [
                            annotation.url
                            for annotation in part.annotations
                            if annotation.type == "url_citation"
                        ]
        # на что модель сослалась, важнее того, что она лишь просмотрела
        return [FoundPage(url) for url in _unique_urls([*cited, *consulted])]


def _https(url: str) -> str:
    """Tavily иногда отдаёт http:// для сайтов, которые отвечают и по https."""
    if url.lower().startswith("http://"):
        return "https://" + url[len("http://") :]
    return url


def _text_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _pages_from_tavily(results: Iterable[Any]) -> list[FoundPage]:
    """По одной странице на адрес, в порядке релевантности.

    Один и тот же адрес приходит с разными якорями, и у каждого свой отрывок:
    склеиваем их, а из полных текстов берём самый длинный.
    """
    titles: dict[str, str | None] = {}
    texts: dict[str, str | None] = {}
    snippets: dict[str, dict[str, None]] = {}
    for item in results:
        if not isinstance(item, dict) or not isinstance(item.get("url"), str):
            continue
        url = normalize_url(_https(item["url"]))
        if url is None:
            continue
        titles[url] = titles.get(url) or _text_or_none(item.get("title"))
        # из нескольких текстов одной страницы нужен самый полный
        candidates = (texts.get(url), _text_or_none(item.get("raw_content")))
        texts[url] = max(
            (text for text in candidates if text is not None), key=len, default=None
        )
        parts = snippets.setdefault(url, {})
        if (snippet := _text_or_none(item.get("content"))) is not None:
            parts[snippet] = None  # словарь вместо множества: сохраняет порядок
    return [
        FoundPage(url, titles[url], texts[url], "\n\n".join(snippets[url]) or None)
        for url in titles
    ]


class TavilyPageFinder(PageFinder):
    """Поисковый API Tavily: фильтр по доменам на стороне поисковика и тексты страниц.

    Сайты профиля отдаём в include_domains (до 300) в режиме «только они», поэтому
    поиск не уходит в сторону, как это делал «разведчик» на модели. Страницы, в том
    числе PDF, приходят уже готовым markdown: скачивать их самим не нужно.
    """

    def __init__(self, client: httpx.AsyncClient, api_key: str) -> None:
        self._client = client
        self._api_key = api_key

    async def find(self, query: str, domains: Sequence[str]) -> list[FoundPage]:
        body = {
            "query": query,
            "include_domains": list(domains),
            # «prefer» добавляет в выдачу и чужие сайты: нам это не нужно
            "include_domains_mode": "restrict",
            # «advanced» вдвое дороже и на проверке не дал лучшего результата
            "search_depth": "basic",
            "max_results": TAVILY_MAX_RESULTS,
            "include_raw_content": "markdown",
        }
        try:
            response = await self._client.post(
                TAVILY_SEARCH_URL,
                json=body,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=TAVILY_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError as exc:
            raise SearchBackendError(f"сеть: {exc!r}") from exc
        if response.status_code != 200:
            # тело ответа не берём в сообщение: оно уходит в лог
            raise SearchBackendError(f"Tavily ответил {response.status_code}")
        try:
            results = response.json()["results"]
        except (ValueError, KeyError, TypeError) as exc:
            raise SearchBackendError("Tavily вернул ответ в неожиданном виде") from exc
        if not isinstance(results, list):
            raise SearchBackendError("Tavily вернул ответ в неожиданном виде")
        return _pages_from_tavily(results)


class FakePageFinder(PageFinder):
    """Отдаёт заранее заданные страницы: для тестов и работы без ключей."""

    def __init__(self, pages: Sequence[str | FoundPage] = ()) -> None:
        self._pages = [
            page if isinstance(page, FoundPage) else FoundPage(page) for page in pages
        ]
        self.queries: list[str] = []
        self.domains: list[list[str]] = []

    async def find(self, query: str, domains: Sequence[str]) -> list[FoundPage]:
        self.queries.append(query)
        self.domains.append(list(domains))
        return list(self._pages)


def build_page_finder(http_client: httpx.AsyncClient) -> PageFinder:
    """Tavily при его ключе, иначе поиск OpenAI при его ключе, иначе пустой фейк."""
    if settings.tavily_api_key:
        return TavilyPageFinder(http_client, settings.tavily_api_key)
    if settings.openai_api_key:
        logger.warning("Ключа Tavily нет: поиск идёт через OpenAI, он менее надёжен")
        return OpenAIPageFinder()
    return FakePageFinder()
