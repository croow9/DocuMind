import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

from app.core.config import settings
from app.ingestion.chunker import ChunkData, split_into_chunks
from app.ingestion.embedder import EmbeddingClient
from app.ingestion.web_pages import (
    USER_AGENT,
    PageFetcher,
    PageFetchError,
    html_to_markdown,
    is_allowed_url,
)
from app.rag.retrieval import TOP_K, top_by_similarity
from app.rag.web_search import PageFinder, build_page_finder
from app.schemas.search import RetrievedChunk

logger = logging.getLogger(__name__)

# Страница-справочник бывает огромной; берём начало, чтобы один вопрос не превратился
# в сотни платных эмбеддингов
MAX_CHUNKS_PER_PAGE = 80

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


@dataclass(frozen=True)
class _Fragment:
    """Кусок скачанной страницы вместе с тем, откуда он."""

    url: str
    title: str
    chunk: ChunkData


class WebSearchService:
    """Ответ на «источники» из документации в интернете.

    Поиск находит адреса, мы их фильтруем по списку сайтов, скачиваем, режем на
    куски и выбираем ближайшие к вопросу по смыслу. На выходе те же
    RetrievedChunk, что и у поиска по своим документам, поэтому промпт, сноски
    и сохранение цитат работают без изменений.
    """

    def __init__(
        self,
        finder: PageFinder,
        fetcher: PageFetcher,
        embedder: EmbeddingClient,
        allowed_domains: Sequence[str],
        max_pages: int,
    ) -> None:
        self.finder = finder
        self.fetcher = fetcher
        self.embedder = embedder
        self.allowed_domains = list(allowed_domains)
        self.max_pages = max_pages

    async def search(
        self, question: str, top_k: int = TOP_K, context: Sequence[str] = ()
    ) -> list[RetrievedChunk]:
        """context — прежние вопросы диалога: без них «а как с yield?» нечем искать."""
        # пустой список сайтов — способ выключить функцию: искать негде
        if not question.strip() or not self.allowed_domains:
            return []

        urls = self._pick_urls(await self.finder.find(question, context))
        # страницы качаются одновременно: это сеть, а не общая сессия БД
        pages = await asyncio.gather(*(self._load_page(url) for url in urls))
        fragments = [fragment for page in pages for fragment in page]
        if not fragments:
            return []

        # вопрос и все куски одним запросом к API: это быстрее и дешевле
        query_vector, *vectors = await self.embedder.embed(
            [question, *(fragment.chunk.content for fragment in fragments)]
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

    def _pick_urls(self, found: Sequence[str]) -> list[str]:
        """Адреса с разрешённых сайтов, без повторов, не больше max_pages.

        Это главная проверка: поисковику (то есть модели с доступом к
        интернету) мы не верим, поэтому всё, что вне списка сайтов, отбрасываем
        здесь, и качать будем только отобранное. Он и правда выходит за рамки:
        на вопрос не про документацию возвращает совсем другие сайты.
        """
        allowed = [
            url
            for url in dict.fromkeys(found)
            if is_allowed_url(url, self.allowed_domains)
        ]
        # сортировка устойчивая: при равных оценках порядок поиска сохраняется
        allowed.sort(key=_url_penalty)
        # одна страница в трёх версиях заняла бы три места из нескольких: берём
        # лучший вариант каждой, остальные места достанутся другим страницам
        unique: dict[tuple[str, str], str] = {}
        for url in allowed:
            unique.setdefault(_page_identity(url), url)
        return list(unique.values())[: self.max_pages]

    async def _load_page(self, url: str) -> list[_Fragment]:
        """Куски одной страницы; недоступная страница не ломает остальные."""
        try:
            html = await self.fetcher.fetch(url)
        except PageFetchError as exc:
            logger.warning("Страница %s пропущена: %s", url, exc)
            return []
        title, text = html_to_markdown(html)
        chunks = split_into_chunks(text)[:MAX_CHUNKS_PER_PAGE]
        return [_Fragment(url=url, title=title or url, chunk=chunk) for chunk in chunks]


def build_web_search_service(
    embedder: EmbeddingClient, http_client: httpx.AsyncClient
) -> WebSearchService:
    return WebSearchService(
        finder=build_page_finder(),
        fetcher=PageFetcher(http_client, settings.web_allowed_domains),
        embedder=embedder,
        allowed_domains=settings.web_allowed_domains,
        max_pages=settings.web_max_pages,
    )


def build_http_client() -> httpx.AsyncClient:
    """Клиент для скачивания страниц: один на всё приложение, с пулом соединений."""
    return httpx.AsyncClient(
        timeout=settings.web_fetch_timeout_seconds,
        headers={"User-Agent": USER_AGENT},
    )
