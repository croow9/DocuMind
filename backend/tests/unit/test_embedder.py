import asyncio
from types import SimpleNamespace

import pytest

from app.ingestion.embedder import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_CONCURRENCY,
    OpenAIEmbedder,
)


class FakeEmbeddings:
    """Подделка client.embeddings: вектор текста «7» — это [7.0]."""

    def __init__(
        self, delays: dict[int, float] | None = None, default_delay: float = 0
    ) -> None:
        self.batches: list[list[str]] = []
        # номер батча -> сколько он «думает»; остальные думают default_delay
        self.delays = delays or {}
        self.default_delay = default_delay
        self.in_flight = 0
        self.max_in_flight = 0
        self.cancelled = 0

    async def create(self, input: list[str], model: str) -> SimpleNamespace:  # noqa: A002
        index = len(self.batches)
        self.batches.append(input)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delays.get(index, self.default_delay))
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.in_flight -= 1
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(t)]) for t in input]
        )


def make_embedder(fake: FakeEmbeddings) -> OpenAIEmbedder:
    client = SimpleNamespace(embeddings=fake)
    return OpenAIEmbedder(client)  # type: ignore[arg-type]


def texts(count: int) -> list[str]:
    return [str(i) for i in range(count)]


async def test_empty_input_makes_no_requests() -> None:
    fake = FakeEmbeddings()

    assert await make_embedder(fake).embed([]) == []
    assert fake.batches == []


@pytest.mark.parametrize(
    ("count", "batches"),
    [
        (1, 1),
        (EMBEDDING_BATCH_SIZE, 1),
        (EMBEDDING_BATCH_SIZE + 1, 2),
        (EMBEDDING_BATCH_SIZE * 3, 3),
    ],
)
async def test_texts_are_split_into_batches_of_the_limit(
    count: int, batches: int
) -> None:
    fake = FakeEmbeddings()

    await make_embedder(fake).embed(texts(count))

    assert len(fake.batches) == batches
    assert all(len(batch) <= EMBEDDING_BATCH_SIZE for batch in fake.batches)


async def test_vectors_keep_the_order_of_texts_even_if_early_batches_finish_last() -> (
    None
):
    # первый батч самый медленный: он заканчивает последним, но результат — первым
    fake = FakeEmbeddings(delays={0: 0.05})
    items = texts(EMBEDDING_BATCH_SIZE * 2 + 7)

    vectors = await make_embedder(fake).embed(items)

    assert vectors == [[float(i)] for i in range(len(items))]


async def test_batches_run_at_the_same_time_up_to_the_limit() -> None:
    fake = FakeEmbeddings(default_delay=0.02)
    # батчей втрое больше лимита: очередь доходит до потолка и упирается в него
    count = EMBEDDING_BATCH_SIZE * EMBEDDING_CONCURRENCY * 3

    await make_embedder(fake).embed(texts(count))

    assert fake.max_in_flight == EMBEDDING_CONCURRENCY
    # параллельно, но скромно: десятки одновременных запросов упрутся в лимиты OpenAI
    assert 1 < EMBEDDING_CONCURRENCY <= 8


async def test_typical_page_set_is_split_so_that_batches_can_run_in_parallel() -> None:
    # 4 страницы по ~40 кусков и вопрос: если всё уйдёт одним запросом,
    # параллелить нечего, и ответ ждёт самый длинный запрос целиком
    fake = FakeEmbeddings()

    await make_embedder(fake).embed(texts(1 + 4 * 40))

    assert len(fake.batches) >= 3


async def test_failed_batch_cancels_the_others_and_raises() -> None:
    class Broken(FakeEmbeddings):
        async def create(self, input: list[str], model: str) -> SimpleNamespace:  # noqa: A002
            if input[0] == "0":
                self.batches.append(input)
                raise RuntimeError("OpenAI недоступен")
            return await super().create(input, model)

    fake = Broken(delays={i: 5 for i in range(10)})

    with pytest.raises(RuntimeError, match="OpenAI недоступен"):
        await make_embedder(fake).embed(texts(EMBEDDING_BATCH_SIZE * 3))
    await asyncio.sleep(0)  # дать отменённым задачам отреагировать

    assert fake.cancelled == 2
    assert fake.in_flight == 0


async def test_cancelling_the_whole_call_cancels_running_batches() -> None:
    fake = FakeEmbeddings(delays={i: 5 for i in range(10)})
    call = asyncio.create_task(
        make_embedder(fake).embed(texts(EMBEDDING_BATCH_SIZE * 3))
    )
    await asyncio.sleep(0.01)  # батчи ушли в работу

    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    await asyncio.sleep(0)

    assert fake.cancelled == 3
