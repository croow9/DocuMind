import math
from collections.abc import Sequence
from uuid import UUID

# Сколько кандидатов берём из каждого поиска и сколько оставляем после слияния
CANDIDATES_PER_SEARCH = 20
TOP_K = 6

# Константа сглаживания из формулы RRF. Чем она больше, тем меньше разрыв между
# первым и вторым местом. 60 взято из статьи, где метод предложили
RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[UUID]], k: int = RRF_K
) -> list[UUID]:
    """Слить несколько упорядоченных списков в один (Reciprocal Rank Fusion).

    Каждое вхождение даёт элементу 1 / (k + место), места считаются с 1, а баллы
    из разных списков складываются. Сливаем по местам, а не по «оценкам»:
    у косинусной близости и у ts_rank_cd разные шкалы, их нельзя складывать.

    Элемент, замеченный в обоих списках, обгоняет элемент, который лидирует
    только в одном. При равных баллах порядок такой: кто раньше встретился,
    тот выше (для этого полагаемся на устойчивую сортировку). Внутри одного
    списка элементы не должны повторяться.
    """
    scores: dict[UUID, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position)
    return sorted(scores, key=lambda item: scores[item], reverse=True)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Косинус угла между векторами: 1 — одно направление, 0 — ничего общего.

    Длины считаем сами, а не полагаемся на единичные векторы: у OpenAI они
    единичные, но фейковый эмбеддер и другие модели этого не гарантируют.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def top_by_similarity(
    query: Sequence[float], vectors: Sequence[Sequence[float]], top_k: int
) -> list[int]:
    """Номера top_k векторов, самых близких к query, от ближайшего.

    Сортировка устойчивая: при равной близости раньше идёт тот, что был раньше
    в списке (страницы и разделы в порядке, который дал поиск).
    """
    scores = [cosine_similarity(query, vector) for vector in vectors]
    order = sorted(range(len(vectors)), key=lambda i: scores[i], reverse=True)
    return order[:top_k]
