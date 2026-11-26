import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from app.core.cache import Cache
from app.core.config import settings
from app.ingestion.chunker import MAX_CHUNK_LENGTH, ChunkData, split_into_chunks
from app.ingestion.embedder import EmbeddingClient
from app.ingestion.web_pages import (
    USER_AGENT,
    FetchedPage,
    PageFetcher,
    PageFetchError,
    clean_markdown,
    is_allowed_url,
)
from app.rag.query_rewrite import QueryRewriter, build_query_rewriter
from app.rag.retrieval import TOP_K, top_by_similarity
from app.rag.web_profiles import WebProfile
from app.rag.web_search import FoundPage, PageFinder, build_page_finder
from app.schemas.search import RetrievedChunk

logger = logging.getLogger(__name__)

# Страница-справочник бывает огромной; берём начало, чтобы один вопрос не превратился
# в сотни платных эмбеддингов
MAX_CHUNKS_PER_PAGE = 80
# Страница длиннее этого (спецификация на сотни страниц) не помещается в лимит
# кусков: начало почти ничего не скажет о вопросе, поэтому вместо неё берём
# отрывки, которые поисковик выбрал под вопрос
MAX_FULL_TEXT_CHARS = MAX_CHUNKS_PER_PAGE * MAX_CHUNK_LENGTH
# Больше этого в кэш не кладём: он для быстрого доступа к небольшим записям
MAX_CACHED_BYTES = 1_000_000

# Поиск возвращает одну и ту же страницу во всех переводах и версиях. Читаем
# русские и английские; остальное идёт в конец очереди, а не отбрасывается: если
# больше ничего не нашлось, лучше страница на другом языке, чем отказ
PREFERRED_LANGUAGES = frozenset({"ru", "en"})
_LANGUAGE_SEGMENT = re.compile(r"^[a-z]{2}(?:-[a-zA-Z]{2,4})?$")
# закреплённая версия вроде /3.11/: адрес вида /3/ ведёт на актуальную
_PINNED_VERSION_SEGMENT = re.compile(r"^\d+\.\d+$")


# Переводы и версии одной и той же страницы: /ru/, /3.11/, /latest/, /pl/3/
_VARIANT_SEGMENT = re.compile(
    r"^(?:[a-z]{2}(?:-[a-zA-Z]{2,4})?|\d+(?:\.\d+)?|latest|stable|dev)$"
)


def _page_identity(url: str) -> tuple[str, str]:
    """Что за страница, независимо от языка и версии: сайт и путь без /ru/, /3.11/ и т.п."""
    parsed = httpx.URL(url)
    segments = [segment for segment in parsed.path.split("/") if segment]
    skipped = 0
    while segments and skipped < 2 and _VARIANT_SEGMENT.match(segments[0]):
        segments.pop(0)
        skipped += 1
    return parsed.host, "/".join(segments)


def _url_penalty(url: str) -> int:
    """Чем меньше, тем лучше: 0 — нужный язык и актуальная версия."""
    segments = [segment for segment in httpx.URL(url).path.split("/") if segment]
    penalty = 0
    if (
        segments
        and _LANGUAGE_SEGMENT.match(segments[0])
        and segments[0] not in PREFERRED_LANGUAGES
    ):
        penalty += 2
    # версия бывает первым сегментом (/3.11/) или вторым после языка (/pl/3.11/)
    if any(_PINNED_VERSION_SEGMENT.match(segment) for segment in segments[:2]):
        penalty += 1
    return penalty


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _search_key(profile: str, max_pages: int, query: str) -> str:
    # регистр и пробелы в запросе роли не играют; профиль и число страниц важны:
    # от них зависит, какие страницы попадут в запись
    return f"web:search:v1:{profile}:{max_pages}:{_digest(_normalized(query))}"


def _page_key(url: str) -> str:
    return f"web:page:v1:{_digest(url)}"


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def _rewrite_key(question: str, context: Sequence[str]) -> str:
    # уточнение «а как с yield?» значит разное в разных диалогах: в ключе и прежние
    # вопросы. Сами вопросы в Redis не лежат, только их отпечаток
    previous = "\n".join(_normalized(text) for text in context)
    # v2: инструкция рерайтера изменилась; при её правках версию надо поднимать
    return f"web:rewrite:v2:{_digest(previous + chr(10) * 2 + _normalized(question))}"


@dataclass(frozen=True)
class _Fragment:
    """Кусок скачанной страницы вместе с тем, откуда он."""

    url: str
    title: str
    chunk: ChunkData


class WebSearchService:
    """Ответ на «источники» из документации в интернете.

    Поиск находит адреса, мы их фильтруем по сайтам профиля, скачиваем, режем на
    куски и выбираем ближайшие к вопросу по смыслу. На выходе те же
    RetrievedChunk, что и у поиска по своим документам, поэтому промпт, сноски
    и сохранение цитат работают без изменений.
    """

    def __init__(
        self,
        finder: PageFinder,
        fetcher: PageFetcher,
        embedder: EmbeddingClient,
        rewriter: QueryRewriter,
        cache: Cache,
        max_pages: int,
        cache_ttl_seconds: int,
    ) -> None:
        self.finder = finder
        self.fetcher = fetcher
        self.embedder = embedder
        self.rewriter = rewriter
        self.cache = cache
        self.max_pages = max_pages
        self.cache_ttl_seconds = cache_ttl_seconds

    async def search(
        self,
        question: str,
        profile: WebProfile,
        top_k: int = TOP_K,
        context: Sequence[str] = (),
    ) -> list[RetrievedChunk]:
        """Источники из документации сайтов профиля.

        context — прежние вопросы диалога: без них «а как с yield?» нечем искать.
        """
        if not question.strip():
            return []

        # ищем и сравниваем по перефразированному запросу, а не по вопросу как есть:
        # он английский, названия продукта в нём не потеряны, а уточнение вроде
        # «а как с yield?» дополнено темой из прежних вопросов
        query = await self._query_for(question, context)
        picked = await self._find_pages(query, profile)
        # страницы качаются одновременно: это сеть, а не общая сессия БД
        pages = await asyncio.gather(
            *(self._load_page(page, profile.domains) for page in picked)
        )
        fragments = [fragment for page in pages for fragment in page]
        if not fragments:
            return []

        # вопрос и все куски одним запросом к API: это быстрее и дешевле
        query_vector, *vectors = await self.embedder.embed(
            [query, *(fragment.chunk.content for fragment in fragments)]
        )
        best = top_by_similarity(query_vector, vectors, top_k)
        return [
            RetrievedChunk(
                chunk_id=None,
                document_id=None,
                document_title=fragments[i].title,
                heading_path=fragments[i].chunk.heading_path or None,
                content=fragments[i].chunk.content,
                url=fragments[i].url,
            )
            for i in best
        ]

    async def _find_pages(self, query: str, profile: WebProfile) -> list[FoundPage]:
        """Страницы для запроса: из кэша или от поисковика (и тогда в кэш).

        В кэше лежит уже отобранное и сжатое: только разрешённые сайты профиля,
        не больше max_pages страниц и тексты не длиннее того, что мы читаем.
        Публичные страницы одинаковы для всех, поэтому запись общая, но ключ
        включает профиль: у профилей разные списки сайтов.
        """
        key = _search_key(profile.name, self.max_pages, query)
        cached = await self._cache_get_json(key)
        if isinstance(cached, list):
            try:
                return [FoundPage(**item) for item in cached]
            except TypeError:  # не словари, лишние или недостающие поля
                logger.warning("Запись кэша %s в неожиданном виде, игнорирую", key)

        found = await self.finder.find(query, profile.domains)
        pages = [
            self._compact(page) for page in self._pick_pages(found, profile.domains)
        ]
        # пустой результат не запоминаем: это мог быть случайный сбой поиска, и
        # он не должен повторяться целые сутки
        if pages:
            await self._cache_put_json(key, [asdict(page) for page in pages])
        return pages

    async def _query_for(self, question: str, context: Sequence[str]) -> str:
        """Запрос для поиска; модель перефразирует не всегда одинаково, а от
        одинакового запроса зависит попадание в кэш поиска, поэтому и
        перефразирование запоминаем."""
        key = _rewrite_key(question, context)
        cached = await self._cache_get(key)
        if cached:
            return cached
        query = await self.rewriter.rewrite(question, context)
        # запрос, совпавший с вопросом, мог быть запасным вариантом при сбое модели:
        # сутки пользоваться им нельзя
        if _normalized(query) != _normalized(question):
            await self._cache_put(key, query)
        return query

    def _compact(self, page: FoundPage) -> FoundPage:
        """Текст страницы, очищенный и не длиннее того, что мы всё равно прочтём."""
        if page.text is None:
            return page
        text = clean_markdown(page.text)
        if len(text) > MAX_FULL_TEXT_CHARS and page.snippet:
            text = clean_markdown(page.snippet)
        # читаем всё равно не больше MAX_CHUNKS_PER_PAGE кусков: остальное только
        # раздуло бы запись в кэше
        return FoundPage(page.url, page.title, text[:MAX_FULL_TEXT_CHARS], None)

    async def _cache_get(self, key: str) -> str | None:
        if self.cache_ttl_seconds <= 0:
            return None
        return await self.cache.get(key)

    async def _cache_put(self, key: str, raw: str) -> None:
        if self.cache_ttl_seconds <= 0:
            return
        if len(raw.encode()) > MAX_CACHED_BYTES:
            logger.info("Запись %s слишком велика для кэша", key)
            return
        await self.cache.set(key, raw, self.cache_ttl_seconds)

    async def _cache_get_json(self, key: str) -> Any | None:
        raw = await self._cache_get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            logger.warning("Запись кэша %s повреждена, игнорирую", key)
            return None

    async def _cache_put_json(self, key: str, value: object) -> None:
        await self._cache_put(key, json.dumps(value, ensure_ascii=False))

    def _pick_pages(
        self, found: Sequence[FoundPage], domains: Sequence[str]
    ) -> list[FoundPage]:
        """Страницы с разрешённых сайтов, без повторов, не больше max_pages.

        Это главная проверка: поисковику мы не верим, поэтому всё, что вне списка
        сайтов, отбрасываем здесь вместе с присланным им текстом. Фильтр по доменам
        у самого поисковика лишь экономит запросы. Модель с доступом к интернету
        и правда выходит за рамки: на вопрос не про документацию возвращает совсем
        другие сайты.
        """
        allowed: dict[str, FoundPage] = {}
        for page in found:
            if is_allowed_url(page.url, domains):
                allowed.setdefault(page.url, page)
        # сортировка устойчивая: при равных оценках порядок поиска сохраняется
        ordered = sorted(allowed.values(), key=lambda page: _url_penalty(page.url))
        # одна страница в трёх версиях заняла бы три места из нескольких: берём
        # лучший вариант каждой, остальные места достанутся другим страницам
        unique: dict[tuple[str, str], FoundPage] = {}
        for page in ordered:
            unique.setdefault(_page_identity(page.url), page)
        return list(unique.values())[: self.max_pages]

    async def _load_page(
        self, page: FoundPage, domains: Sequence[str]
    ) -> list[_Fragment]:
        """Куски одной страницы; недоступная страница не ломает остальные."""
        title, text = page.title or "", ""
        if page.text is not None:
            text = page.text
        else:
            fetched = await self._fetch_page(page.url, domains)
            if fetched is None:
                return []
            title, text = title or fetched.title, fetched.text
        chunks = split_into_chunks(text)[:MAX_CHUNKS_PER_PAGE]
        return [
            _Fragment(url=page.url, title=title or page.url, chunk=chunk)
            for chunk in chunks
        ]

    async def _fetch_page(self, url: str, domains: Sequence[str]) -> FetchedPage | None:
        """Скачанная страница (из кэша или из сети); None, если скачать не удалось."""
        key = _page_key(url)
        cached = await self._cache_get_json(key)
        if isinstance(cached, dict):
            try:
                return FetchedPage(title=cached["title"], text=cached["text"])
            except KeyError:
                logger.warning("Запись кэша %s в неожиданном виде, игнорирую", key)
        try:
            fetched = await self.fetcher.fetch(url, domains)
        except PageFetchError as exc:
            logger.warning("Страница %s пропущена: %s", url, exc)
            return None
        fetched = FetchedPage(fetched.title, fetched.text[:MAX_FULL_TEXT_CHARS])
        await self._cache_put_json(key, asdict(fetched))
        return fetched


def build_web_search_service(
    embedder: EmbeddingClient, http_client: httpx.AsyncClient, cache: Cache
) -> WebSearchService:
    return WebSearchService(
        finder=build_page_finder(http_client),
        fetcher=PageFetcher(http_client),
        embedder=embedder,
        rewriter=build_query_rewriter(),
        cache=cache,
        max_pages=settings.web_max_pages,
        cache_ttl_seconds=settings.web_cache_ttl_seconds,
    )


def build_http_client() -> httpx.AsyncClient:
    """Клиент для скачивания страниц: один на всё приложение, с пулом соединений."""
    return httpx.AsyncClient(
        timeout=settings.web_fetch_timeout_seconds,
        headers={"User-Agent": USER_AGENT},
    )
