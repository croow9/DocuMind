from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

from openai import AsyncOpenAI
from openai.types.responses import ResponseIncludable, WebSearchToolParam

from app.core.config import settings
from app.ingestion.web_pages import normalize_url

# Ответ модели нам не нужен, нужен только список найденных страниц: он приходит
# отдельно от текста. Лимит защищает от лишних (и платных) токенов
FINDER_MAX_OUTPUT_TOKENS = 150

# Параметры запроса заданы типизированными константами, а не словарями на месте:
# так mypy проверяет их по описанию API
_WEB_SEARCH_TOOL: WebSearchToolParam = {"type": "web_search"}
# полный список просмотренных страниц, а не только процитированных в тексте
_INCLUDE_SOURCES: list[ResponseIncludable] = ["web_search_call.action.sources"]


class PageFinder(ABC):
    """Находит страницы документации по вопросу: вопрос -> список адресов.

    Ничему из найденного не доверяем: список сайтов проверяет сервис, который
    этим пользуется, а не сам поисковик.
    """

    @abstractmethod
    async def find(self, question: str, context: Sequence[str] = ()) -> list[str]:
        """context — прежние вопросы пользователя в этом диалоге, от старых к новым."""
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

    Мы не берём у модели готовый ответ: он не вписывается в наши источники со
    сносками. От неё нужны только адреса страниц, а тексты скачиваем и режем
    сами. Ограничение «только эти сайты» у выбранной модели (gpt-4.1-mini) в API
    недоступно, поэтому просим словами в инструкции, а проверяем по-настоящему
    потом, в коде.
    """

    def __init__(
        self, allowed_domains: Sequence[str], client: AsyncOpenAI | None = None
    ) -> None:
        self._allowed_domains = list(allowed_domains)
        self._client = client or AsyncOpenAI(api_key=settings.openai_api_key)

    async def find(self, question: str, context: Sequence[str] = ()) -> list[str]:
        sites = ", ".join(self._allowed_domains)
        prompt = question
        if context:
            # короткий вопрос вроде «а как с yield?» без предыдущих непонятен:
            # о какой библиотеке речь, подсказывает только диалог
            previous = "\n".join(f"- {text}" for text in context)
            prompt = (
                f"Предыдущие вопросы пользователя в этом диалоге:\n{previous}\n\n"
                f"Текущий вопрос: {question}"
            )
        response = await self._client.responses.create(
            model=settings.web_search_model,
            instructions=(
                "Найди в интернете страницы официальной документации, по которым "
                "можно ответить на вопрос. Ищи только на этих сайтах, "
                f"используя оператор site: {sites}. Называй в запросе библиотеку и "
                "предпочитай актуальную версию документации. Отвечать на вопрос "
                "не нужно: достаточно найти страницы."
            ),
            input=prompt,
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
        return _unique_urls([*cited, *consulted])


class FakePageFinder(PageFinder):
    """Отдаёт заранее заданные адреса: для тестов и работы без ключа OpenAI."""

    def __init__(self, urls: Sequence[str] = ()) -> None:
        self._urls = list(urls)
        self.questions: list[str] = []
        self.contexts: list[list[str]] = []

    async def find(self, question: str, context: Sequence[str] = ()) -> list[str]:
        self.questions.append(question)
        self.contexts.append(list(context))
        return list(self._urls)


def build_page_finder() -> PageFinder:
    """Настоящий поиск при наличии ключа OpenAI, пустой фейк без него."""
    if settings.openai_api_key:
        return OpenAIPageFinder(settings.web_allowed_domains)
    return FakePageFinder()
