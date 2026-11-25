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

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9094"
    kafka_consumer_group: str = "documind-worker"
    kafka_auto_offset_reset: str = "earliest"

    # Topics

    # App


settings = Settings()
