import os

# Тесты не ходят в реальные сервисы — настройкам достаточно любых значений.
# setdefault не трогает переменные, уже заданные окружением или локальным .env,
# поэтому локально и в CI поведение одинаковое.
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test"
)
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-with-at-least-32-bytes")
