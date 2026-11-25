from pathlib import Path

from pydantic import PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: PostgresDsn
    redis_url: str

    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Модель, которая отвечает на вопросы. Берём не «думающую» (gpt-5*, o*): им
    # лимит ответа расходуется ещё и на рассуждения, и ответ может оборваться
    chat_model: str = "gpt-4.1-mini"
    # Потолок длины ответа в токенах: защита от слишком длинных и дорогих ответов
    chat_max_tokens: int = 1000

    # Поиск по документации в интернете (включается флагом web_search в запросе).
    # Ходим только на эти сайты (и их поддомены); пустой список выключает поиск
    web_allowed_domains: list[str] = [
        "fastapi.tiangolo.com",
        "docs.python.org",
        "docs.pydantic.dev",
        "docs.sqlalchemy.org",
    ]
    # Модель, которая ищет страницы (её встроенный веб-поиск). Не «думающая»:
    # по той же причине, что и chat_model; ограничение по доменам делаем сами
    web_search_model: str = "gpt-4.1-mini"
    # Сколько страниц скачиваем на один вопрос
    web_max_pages: int = 4
    web_fetch_timeout_seconds: float = 10.0

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9094"
    kafka_consumer_group: str = "documind-worker"
    kafka_auto_offset_reset: str = "earliest"

    # Topics

    # App


settings = Settings()
