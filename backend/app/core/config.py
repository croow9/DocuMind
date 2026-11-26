from pathlib import Path

from pydantic import Field, PostgresDsn
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
    # Сколько ждать ответа OpenAI (для потока: очередного куска), секунд
    openai_timeout_seconds: float = 60.0
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Модель, которая отвечает на вопросы. Берём не «думающую» (gpt-5*, o*): им
    # лимит ответа расходуется ещё и на рассуждения, и ответ может оборваться
    chat_model: str = "gpt-4.1-mini"
    # Потолок длины ответа в токенах: защита от слишком длинных и дорогих ответов
    chat_max_tokens: int = 1000

    # Поиск по документации в интернете (включается флагом web_search в запросе).
    # Какие сайты разрешены, описано профилями в app/rag/web_profiles.json; это
    # профиль, который берётся, если клиент не указал свой
    web_default_profile: str = "backend"
    # Поисковый API с фильтром по доменам и текстами страниц; пусто = запасной
    # поиск через OpenAI (менее надёжный на длинных списках сайтов)
    tavily_api_key: str = ""
    # Модель, которая ищет страницы (её встроенный веб-поиск). Не «думающая»:
    # по той же причине, что и chat_model; ограничение по доменам делаем сами
    web_search_model: str = "gpt-4.1-mini"
    # Сколько страниц скачиваем на один вопрос
    web_max_pages: int = 4
    web_fetch_timeout_seconds: float = 10.0
    # На сколько кэшируем найденные страницы документации (0 — не кэшировать).
    # Документация меняется редко, а поиск и скачивание стоят секунд и кредитов
    web_cache_ttl_seconds: int = 86_400

    # Ограничение частоты запросов на одного пользователя. Окна фиксированные
    # (минута, сутки), поэтому на стыке окон возможен всплеск до двойного лимита.
    # Не меньше 1: нулевой лимит молча запретил бы запросы всем.
    # Вопрос по своим документам стоит эмбеддинга и ответа модели
    rate_limit_chat_per_minute: int = Field(default=20, ge=1)
    # Вопрос по интернету дороже: поиск в Tavily (на бесплатном тарифе 1000
    # кредитов в месяц на всех), страницы и эмбеддинги многих кусков. Поэтому
    # строже и с дневным потолком. Дневной лимит одного пользователя общий
    # бюджет Tavily не защищает: пользователей много, лимит у каждого свой
    rate_limit_web_per_minute: int = Field(default=5, ge=1)
    rate_limit_web_per_day: int = Field(default=30, ge=1)
    # Загрузка документа запускает индексацию, то есть эмбеддинги всего текста
    rate_limit_upload_per_minute: int = Field(default=10, ge=1)

    # Защита от подбора паролей и массового создания учётных записей. Окно входа
    # 15 минут. Считаются все попытки, не только неудачные: счётчик увеличивается
    # до проверки пароля, и параллельные запросы не успевают проскочить мимо
    # него. Ключ по адресу клиента: если ключ только по аккаунту, злоумышленник
    # мог бы заблокировать чужой вход
    # Попыток входа в один аккаунт с одного адреса
    rate_limit_login_per_account: int = Field(default=10, ge=1)
    # Попыток входа с одного адреса вообще (перебор по многим аккаунтам)
    rate_limit_login_per_ip: int = Field(default=30, ge=1)
    # Регистраций с одного адреса за час: без этого лимиты на пользователя
    # обходятся созданием новых учётных записей
    rate_limit_register_per_hour: int = Field(default=10, ge=1)

    # Загрузка документов. Файл читается в память, а текст уходит на платные
    # эмбеддинги: без потолков один файл или сотни файлов дорого обойдутся
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    max_documents_per_user: int = Field(default=50, ge=1)

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9094"
    kafka_consumer_group: str = "documind-worker"
    kafka_auto_offset_reset: str = "earliest"

    # Topics

    # App


settings = Settings()
