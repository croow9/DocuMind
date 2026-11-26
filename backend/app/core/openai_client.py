from openai import AsyncOpenAI

from app.core.config import settings


def build_openai_client() -> AsyncOpenAI:
    """Клиент OpenAI с явным таймаутом.

    Таймаут SDK по умолчанию — 10 минут: зависшее соединение держало бы вопрос
    пользователя столько же. Для потока ответа это время ожидания очередного
    куска, а не всего ответа.
    """
    return AsyncOpenAI(
        api_key=settings.openai_api_key, timeout=settings.openai_timeout_seconds
    )
